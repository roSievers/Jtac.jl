"""
    ticket_sizes(n, m)

Returns a list of `m` integers that sums up to `n`.
"""
function ticket_sizes(n, m)
  ns = zeros(Int, m)
  ns .+= floor(Int, n / m)
  ns[1:(n % m)] .+= 1
  ns
end

struct StopMatch <: Exception end

"""
    stop_match()
    stop_match(n)

Can be used in the `callback_move` argument for `record` in order to
stop recording a game. The method `stop_match(n)` is a function
that takes an argument `k` and calls `stop_match()` when `k > n`.

## Usage
The following call of `record` will cancel matches that make more
than 50 moves.

    Player.record(args...; callback_move = Player.stop_match(50), kwargs...)
"""
stop_match() = throw(StopMatch())
stop_match(n :: Int) = k -> (k > n && stop_match())

# -------- Generating Datasets: API ------------------------------------------ #

"""
    record(player [, n]; <keyword arguments>)

Record a dataset by letting `player` play against itself `n` times.

The value label for the dataset is derived from the outcome of the game (-1, 0,
or +1). The policy label is the policy proposed by the player for the respective
game state.

# Arguments
- `instance`: Function applied on the game type to obtain an initial state for
   a single match. Can be used to randomize the initial positions.
- `branch`: Function applied to each game state after a first selfplay. It can
   return `nothing` or a branched game state, which is then used as roots for
   new matches (*without* recursive branching). See also `Game.branch`.
- `augment = true`: Whether to apply augmentation on the generated dataset.
- `merge = true`: Whether to return one merged dataset or `n` seperate ones.
- `opt_targets`: Optional targets for which labels are recorded. Derived from
   the mcts player model by default.
- `callback`: Procedure that is called afer each completed match.
- `callback_move`: Procedure called after each individual move.
- `threads = :auto`: Whether to conduct the matches on background threads.
   If `true` or `:copy`, all available background threads are used.
   The option `:copy` lets each thread receive a copy of the player. If this
   is combined with GPU-based models, each thread receives its own device.
   If `:auto`, background threads are used if this is possible.

# Examples
```julia
# Record 20 self-matches of an classical MCTS player with power 250
G = Game.TicTacToe
player = Player.MCTSPlayer(power = 250)
dataset = Player.record(player, 20, game = G, branch = Util.branch(prob = 0.25))

# Record 10 self-matches of MCTS player with shallow predictor network and
# power 50
G = Game.TicTacToe
model = Model.NeuralModel(G, Model.@chain G Dense(50, "relu"))
player = Player.MCTSPlayer(model, power = 50)
dataset = Player.record(player, 10, augment = false)
```
"""
function record( p :: AbstractPlayer{G}
               , n :: Int = 1
               ; merge = true
               , threads = :auto
               , callback_move = _ -> nothing
               , opt_targets = Target.targets(p)[3:end]
               , kwargs... ) where {G}

  # Using only a single BLAS thread was beneficial for performance in all tests
  t = BLAS.get_num_threads()
  BLAS.set_num_threads(1)

  # Do we want to use threading?
  bgthreads = Threads.nthreads() - 1
  use_threads =
    threads == true || threads == :copy ||
    threads == :auto && bgthreads > 0 && is_async(p)

  if use_threads

    tickets = ticket_sizes(n, bgthreads)
    dss = _threaded([p]; threads) do idx, ps
      record( ps[1]
            , tickets[idx]
            ; threads = false
            , merge = false
            , opt_targets
            , callback_move
            , kwargs... )
    end
    ds = vcat(dss...)

  else

    # Function that plays a single match, starting at state `game`
    play = game -> begin

      H = typeof(game)
      targets = [Target.defaults(H); opt_targets] 

      dataset = DataSet(typeof(game), targets)

      moves = 0

      try

        while !is_over(game)

          # Get the improved policy from the player
          policy = think(p, game)

          # Record the current game state and policy target label
          # (we do not know the value label yet)
          push!(dataset.games, copy(game))
          push!(dataset.labels[2], policy)

          # Advance the game by randomly drawing from the policy
          apply_action!(game, choose_index(policy))

          moves += 1
          callback_move(moves)

        end

        # Push the final game state to the dataset. This game state is included
        # for calculating optional targets, but it has to be removed afterwards
        pl = policy_length(H)
        push!(dataset.games, copy(game))
        push!(dataset.labels[2], ones(Float32, pl) ./ pl)

        # Now we know how the game ended. Add value labels to the dataset.
        # Optional targets have to be recorded *after* augmentation
        s = status(dataset.games[end])
        for game in dataset.games
          push!(dataset.labels[1], Float32[current_player(game) * s])
        end

        dataset

      catch err
        if err isa StopMatch
          DataSet(typeof(game), targets)
        else
          throw(err)
        end
      end

    end

    # Record several matches with (optional) branching and augmentation
    ds = record_with_branching(G, play, n; ntasks = Model.ntasks(p), kwargs...)

  end

  BLAS.set_num_threads(t)

  merge ? Base.merge(ds) : ds

end

function record_with_branching( G :: Type{<: AbstractGame}
                              , play :: Function # maps game to dataset
                              , n :: Int
                              ; ntasks 
                              , augment = Game.is_augmentable(G)
                              , callback = () -> nothing
                              , instance = () -> Game.instance(G)
                              , branch = Game.branch(prob = 0, steps = 1) )

  # Extend the provided play function by random branching 
  play_with_branching = _ -> begin

    root = instance()

    @assert root isa G "Instance of $G not of correct type"

    # Play the game once without branching.
    # Then, create branchpoints and play them as well.
    dataset = play(root)
    bpoints = filter(!isnothing, branch.(dataset.games[1:end-1]))
    branches = play.(bpoints)
    filter!(x -> length(x) > 0, branches)

    datasets = [dataset; branches]

    # Augment the datasets and record optional targets
    augment && (datasets = mapreduce(Game.augment, vcat, datasets))
    foreach(record_opt_targets!, datasets)

    callback()
    Base.merge(datasets)

  end

  # Call play_with_branching n times
  if ntasks == 1 || n == 1
    ds = map(play_with_branching, 1:n)
  else
    ds = asyncmap(play_with_branching, 1:n, ntasks = ntasks)
  end

  ds

end


function record_opt_targets!(dataset :: DataSet{G}) where {G}

  length(dataset) == 0 && return

  opt_targets = Target.targets(dataset)[3:end]
  for (l, target) in enumerate(opt_targets)
    label = dataset.labels[l+2]
    for i = 1:(length(dataset.games)-1)
      game = dataset.games[i]
      push!(label, Target.evaluate(target, game, i, dataset))
    end
  end

  # Remove the finished game from the dataset
  pop!(dataset.games)
  pop!(dataset.labels[1])
  pop!(dataset.labels[2])

  # Hope that everything went well :)
  @assert Data.check_consistency(dataset)

  dataset
end




"""
Exception that is thrown as stop signal for the `record(player, channel)`
method.
"""
struct StopRecording <: Exception end

"""
    record(player, channel; ntasks, kwargs...)

Stream the games recorded by `player` to `channel`. The value
of `ntasks` determines the number of matches played asynchronously
per thread if `player` is based on an `Async` model.

The function returns when `channel` is closed.
"""
function record( p :: AbstractPlayer{H}
               , ch :: AbstractChannel{DataSet{G}}
               ; callback_move = () -> nothing
               , ntasks = is_async(p) ? 50 : 1
               , merge = true
               , threads = :auto
               , kwargs... ) where {H <: AbstractGame, G <: H}

  cb() = begin
    callback_move()
    isopen(ch) || throw(StopRecording())
  end

  main_loop(player) = begin
    asyncmap(1:ntasks; ntasks) do _
      while isopen(ch)
        ds = nothing
        try ds = record( player, 1; threads = false, merge = true
                       , callback_move = cb, kwargs... )
        catch err
          err isa StopRecording && break
          throw(err)
        end
        try put!(ch, ds)
        catch err
          err isa InvalidStateException && break
          throw(err)
        end
      end
    end
    nothing
  end

  t = BLAS.get_num_threads()
  BLAS.set_num_threads(1)

  bgthreads = Threads.nthreads() - 1
  use_threads =
    threads == true || threads == :copy ||
    threads == :auto && bgthreads > 0 && is_async(p)

  if use_threads
    _threaded([p]; threads) do idx, ps
      main_loop(ps[1])
    end
  else
    main_loop(p)
  end

  BLAS.set_num_threads(t)

end




"""
    evaluate(model or player, games [; augment])

Record a dataset by applying `model` or `player` to `games`.
"""
function evaluate( p :: Union{AbstractModel{G}, AbstractPlayer{G}}
                 , games :: Vector{T}
                 ; augment = false
                 , threads = :auto
                 ) where {G, T <: G}

  games = augment ? mapreduce(Game.augment, vcat, games) : games

  targets = Target.targets(p)
  labels = [Data.LabelData() for _ in targets]

  # Using only a single BLAS thread was beneficial for performance in all tests
  t = BLAS.get_num_threads()
  BLAS.set_num_threads(1)

  # Do we want to use threading?
  bgthreads = Threads.nthreads() - 1
  use_threads =
    threads == true || threads == :copy ||
    threads == :auto && bgthreads > 0 && is_async(p)

  if use_threads

    n = length(games)
    tickets = ticket_sizes(n, bgthreads)
    idxs = [0; cumsum(tickets)]
    ranges = [x+1:y for (x,y) in zip(idxs[1:end-1], idxs[2:end])]
    dss = _threaded([p]; threads) do idx, ps
      ds = evaluate( ps[1]
                   , games[ranges[idx]]
                   ; threads = false
                   , augment )
      ds
    end
    merge(dss)
  else

    # TODO: this can be made faster when specializing on NeuralModels, where many
    # games can be evaluated in parallel
    for game in games
      if p isa NeuralModel
        output = apply(p, game, true)
      else
        output = apply(p, game)
      end
      for l in 1:length(targets)
        if output[l] isa Float32
          val = Float32[output[l]]
        else
          val = output[l]
        end
        push!(labels[l], val)
      end
    end

    DataSet(games, labels, targets)

  end

end

"""
    evaluate(model or player, channel; instance, augment, threads)
"""
function evaluate( p :: Union{AbstractPlayer, AbstractModel}
                 , ch :: AbstractChannel
                 ; instance
                 , augment = false
                 , threads = :auto )

  main_loop(player) = begin
    while isopen(ch)
      games = nothing
      try
        games = instance()
        games = games isa AbstractGame ? [games] : games
        games = augment ? vcat(Game.augment.(games)...) : games
      catch err
        err isa StopRecording && break
        throw(err)
      end
      try
        for game in games
          if player isa NeuralModel
            output = apply(player, game, true)
          else
            output = apply(player, game)
          end
          put!(ch, (game, output))
        end
      catch err
        err isa InvalidStateException && break
        throw(err)
      end
    end

  end

  t = BLAS.get_num_threads()
  BLAS.set_num_threads(1)

  bgthreads = Threads.nthreads() - 1
  use_threads =
    threads == true || threads == :copy ||
    threads == :auto && bgthreads > 0 && is_async(p)

  if use_threads
    _threaded([p]; threads) do idx, ps
      main_loop(ps[1])
    end
  else
    main_loop(p)
  end

  BLAS.set_num_threads(t)

end

# -------- Threaded Recording --------------------------------------------- #

function _threaded(handle :: Function, ps :: Vector; threads)

  bgthreads = Threads.nthreads() - 1
  @assert bgthreads > 0 "No background threads available"

  # Not sure: Is this safe for any model?
  copy = threads == :copy
  #safe = all(is_async(p) for p in ps) || copy
  #@assert safe "Threading for non-async models is unsafe"

  # Having multiple threads work with different models on the same GPU created
  # problems for us, so we don't allow more threads than GPU devices
  ps_ = map(ps) do p
    gpu = on_gpu(training_model(p))
    if gpu && copy
      @assert bgthreads <= length(CUDA.devices()) "More threads than GPU devices"
      p = tune(p, gpu = false)
    end
    (p, gpu)
  end

  # Curious: if ps_ was named ps, the following code would produce random
  # errors... Julia bug related to closures?

  ThreadPools.bmap(1:bgthreads) do idx
    if any(gpu for (_, gpu) in ps_) && copy
      # switch CUDA device if a copy is brought to gpu
      CUDA.device!(idx - 1)
    end
    ps = map(ps_) do (p, gpu)
      if gpu && copy
        tune(p, gpu = true)
      elseif copy || !gpu && threads == :auto
        Base.copy(p)
      else
        p
      end
    end
    handle(idx, ps)
  end

end


