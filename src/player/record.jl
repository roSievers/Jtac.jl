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

"""
    with_BLAS_threads(f, n)

Call `f()` under the temporary setting `BLAS.set_num_threads(n)`.
"""
function with_BLAS_threads(f, n)
  t = BLAS.get_num_threads()
  BLAS.set_num_threads(n)
  try f()
  finally
    BLAS.set_num_threads(t)
  end
end

"""
Exception that can be used to stop the recording or evaluation of matches via
the callback or instance functions.
"""
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
- `merge = true`: Whether to return one merged dataset or `n` seperate ones.
- `augment`: Whether to apply augmentation on the generated dataset.
- `opt_targets`: Optional targets for which labels are recorded. Derived from
   the mcts player model by default.
- `callback`: Procedure that is called afer each completed match.
- `callback_move`: Procedure called after each individual move.

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
function record( p :: Union{P, Channel{P}}
               , n :: Int = 1
               ; merge :: Bool = true
               , callback_move :: Function = _ -> nothing
               , opt_targets = Target.targets(fetch(p))[3:end]
               , best_action_after :: Int = typemax(Int) 
               , kwargs...
               ) where {P <: AbstractPlayer}


  # Function that plays a single match, starting at state `game`
  play = (game, moves) -> begin

    H = typeof(game)
    game = copy(game)
    targets = [Target.defaults(H); opt_targets] 
    dataset = DataSet(H, targets)

    try
      while !is_over(game)

        # Get the improved policy from the player
        policy = think(fetch(p), game)

        # Record the current game state and policy target label
        # (we do not know the value label yet)
        push!(dataset.games, copy(game))
        push!(dataset.labels[2], policy)

        # Advance the game by randomly drawing from the policy
        if moves >= best_action_after
          action = findmax(policy)[2]
        else
          action = choose_index(policy)
        end
        apply_action!(game,  action)

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

  ds = with_BLAS_threads(1) do
    # Record several matches with (optional) branching and augmentation
    G = Model.gametype(fetch(p))
    ntasks = Model.ntasks(fetch(p))
    record_with_branching(G, play, n; ntasks, kwargs...)
  end

  merge ? Base.merge(ds) : ds
end

function record_with_branching( G :: Type{<: AbstractGame}
                              , play :: Function # maps game to dataset
                              , n :: Int
                              ; ntasks :: Int
                              , augment = Game.is_augmentable(G)
                              , callback = () -> nothing
                              , instance = () -> Game.instance(G)
                              , branch = Game.branch(prob = 0, steps = 1) )

  # Extend the provided play function by random branching 
  play_with_branching = _ -> begin

    root = instance()
    H = typeof(root)
    moves = Game.moves(root)

    @assert H <: G "Unexpected game instance type $H (expected $G)"

    # Play the game once without branching.
    # Then, create branchpoints and play them as well.
    dataset = play(root, moves)
    branches = DataSet{H}[]
    for (i, game) in enumerate(dataset.games[1:end-1])
      branchpoint = branch(game)
      if !isnothing(branchpoint) && !Game.is_over(branchpoint)
        ds = play(branchpoint, moves + i - 1)
        push!(branches, ds)
      end
    end

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
               ; callback_move = _ -> nothing
               , ntasks = Model.ntasks(p)
               , merge = true
               , kwargs... ) where {H <: AbstractGame, G <: H}

  cb(move) = begin
    isopen(ch) || throw(StopRecording())
    callback_move(move)
  end

  main_loop(player) = begin
    @sync for i in 1:ntasks
      @async begin
        while isopen(ch)
          ds = nothing
          try ds = record( player, 1; merge = true
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
    end
    nothing
  end

  with_BLAS_threads(1) do
      main_loop(p)
  end
end


"""
    evaluate(model or player, games [; augment])

Record a dataset by applying `model` or `player` to `games`.
"""
function evaluate( p :: Union{AbstractModel{G}, AbstractPlayer{G}}
                 , games :: Vector{T}
                 ; augment = Game.is_augmentable(T)
                 ) where {G, T <: G}

  games = augment ? mapreduce(Game.augment, vcat, games) : games

  targets = Target.targets(p)
  labels = [Data.LabelData() for _ in targets]

  with_BLAS_threads(1) do

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

  end # with_BLAS_threads(1)
end

"""
    evaluate(model or player, channel; instance, augment)
"""
function evaluate( p :: Union{AbstractPlayer, AbstractModel}
                 , ch :: AbstractChannel
                 ; instance :: Function )

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
      for game in games
        if player isa NeuralModel
          output = apply(player, game, true)
        else # Todo: Introduce target evaluation for players too?
          output = apply(player, game)
        end
        try
          put!(ch, (game, output))
          yield()
        catch err
          err isa InvalidStateException && break
          throw(err)
        end
      end
    end
  end

  with_BLAS_threads(1) do
    main_loop(p)
  end
end

