

function _record( play :: Function # maps game to dataset
                , root
                , n :: Int
                ; features 
                , ntasks 
                , augment = true
                , callback = () -> nothing
                , prepare = prepare(steps = 0)
                , branch = branch(prob = 0, steps = 1) )

  # Extend the provided play function by random branching 
  # and the call to callback
  play_with_branching = _ -> begin

    # Play the (prepared) game once without branching inbetween
    dataset = play(prepare(Game.instance(root)))

    # Create branchpoints and play them, too
    branchpoints = filter(!isnothing, branch.(dataset.games[1:end-1]))
    branches = play.(branchpoints)

    # Filter branches where we were directly lead to the end of the game
    filter!(x -> length(x) > 0, branches)

    # Collect the original and the branched datasets
    datasets = [dataset; branches]

    # Augment the datasets (if desired)
    augment && (datasets = mapreduce(Game.augment, vcat, datasets))

    # Calculate features (if there are any)
    datasets = map(d -> _record_features!(d, features), datasets)

    # Signal that we are done with one iteration
    callback()

    Base.merge(datasets...)

  end

  # Call play_with_branching n times
  if ntasks == 1 || n == 1

    # Asyncmap is assumed to be of disadvantage
    ds = map(play_with_branching, 1:n)

  else

    # Asyncmap is assumed to be of advantage
    ds = asyncmap(play_with_branching, 1:n, ntasks = ntasks)

  end

  # Mark the given features as activated
  for d in ds d.features = features end

  ds

end

function _record_move!( dataset :: DataSet{G}
                      , game
                      , player
                      ) where {G <: AbstractGame}

  pl = policy_length(G)

  # Record the current game state 
  push!(dataset.games, copy(game))

  # Get the improved policy from the player
  policy = think(player, game)

  # Advance the game randomly according to the policy
  apply_action!(game, choose_index(policy))

  # Prepare the label for this game state with policy filled in
  label = zeros(Float32, 1 + pl) 
  label[2:pl+1] .= policy

  push!(dataset.label, label)

end

# This functions adds the game state of the finished game with a dummy label
# This is done so that the features have the final game state available
function _record_final!( dataset :: DataSet{G}
                       , game ) where {G <: AbstractGame}

  # TODO: should put value here (while policy does not make sense)!
  pl = policy_length(G)
  push!(dataset.games, copy(game))
  push!(dataset.label, zeros(Float32, 1 + pl))

end


function _record_value!(dataset :: DataSet{G}) where {G <: AbstractGame}

  result = status(dataset.games[end])

  for i = 1:length(dataset.games)

    dataset.label[i][1] = current_player(dataset.games[i]) * result

  end

end

# Add feature label to an extended dataset (that contains the final game state)
function _record_features!( dataset :: DataSet{G}
                          , features
                          ) where {G <: AbstractGame}

  fl = feature_length(features, G)

  dataset.flabel = Array{Float32}[]

  # Iterate the unextended dataset
  for i = 1:(length(dataset.games)-1)

    j = 0
    flabel = zeros(Float32, fl)

    # Set all features
    for f in features
      l = feature_length(f, G)
      flabel[j+1:j+l] = f(dataset.games[i], dataset.games)
      j += l
    end

    push!(dataset.flabel, flabel)

  end

  # Remove the final game (and the corresponding zero-label) from the dataset
  # now that the feature labels are calculated.
  pop!(dataset.games)
  pop!(dataset.label)

  dataset

end


# -------- Generating Datasets: API ------------------------------------------ #

"""
    record(player [, n]; <keyword arguments>)

Record a dataset by letting `player` play against itself `n` times.

The value label for the dataset is derived from the outcome of the game (-1, 0,
or +1). The policy label is the policy proposed by the player for the respective
game state. The feature labels are calculated by applying the provided features
(see keyword arguments) to each stored game state and the corresponding history.

# Arguments
- `game`: Initial game type or state that is compatible with `player`. Derived by default.
- `features`: List of features for which labels are created. Derived by default.
- `augment = true`: Whether to apply augmentation on the generated dataset.
- `merge = true`: Whether to return one merged dataset or `n` seperate ones.
- `callback`: Procedure that is called afer each completed match.
- `callback_move`: Procedure called after each individual move.
- `prepare`: Function applied on `game` that returns the initial state for
   a single selfplay. Can be used to randomize the initial positions, for
   example by random actions. Default preparation maps can be generated by
   `prepare(; steps).
- `branch`: Function applied to each game state of a first selfplay. It can
   return `nothing` or branched game states, which are then used as roots for
   new matches (without recursive branching). Default branching maps can be
   generated by `Util.branch(; prob, steps)`.
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
dataset = Data.record(player, 20, game = G, branch = Util.branch(prob = 0.25))

# Record 10 self-matches of MCTS player with shallow predictor network and
# power 50
G = Game.TicTacToe
model = Model.NeuralModel(G, Model.@chain G Dense(50, "relu"))
player = Player.MCTSPlayer(model, power = 50)
dataset = Player.record(player, 10, augment = false)
```
"""
function record( p :: AbstractPlayer
               , n :: Int = 1
               ; game = Player.derive_gametype(p)
               , features = features(p)
               , merge = true
               , threads = :auto
               , callback_move = () -> nothing
               , kwargs... )

  # Using only a single BLAS thread was beneficial for performance in all tests
  t = BLAS.get_num_threads()
  BLAS.set_num_threads(1)

  bgthreads = Threads.nthreads()
  use_threads =
    threads == true || threads == :copy ||
    threads == :auto && bgthreads > 0 && is_async(p)

  if use_threads

    tickets = ticket_sizes(n, bgthreads)
    dss = _threaded([p]; threads) do idx, ps
      record(ps[1], tickets[idx]; threads = false, merge = false, kwargs...)
    end
    ds = vcat(dss...)

  else

    # Function that plays a single game of player against itself
    play = game -> begin

      dataset = DataSet{typeof(game)}()

      while !is_over(game)

        # Play one move, prepare the label, and record the game and player policy
        _record_move!(dataset, game, p)
        callback_move()

      end

      # Here, we push the final game state to the dataset.
      # This game is needed for calculating the features later, but it has to be
      # removed again after that. See _record_features!
      _record_final!(dataset, game)

      # Complete the label value information now that we know the total game history
      _record_value!(dataset)

      # Note: the features are calculated later, after augmentation steps took
      # place in _record below

      # Return the value-policy labeled dataset
      dataset

    end

    # Record several games, potentially with branching
    ds = _record(play, game, n; features, ntasks = Model.ntasks(p), kwargs...)

  end

  BLAS.set_num_threads(t)

  merge ? Base.merge(ds...) : ds

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

  run_loops(player) = begin
    asyncmap(1:ntasks; ntasks) do _
      while isopen(ch)
        ds = nothing
        try ds = record(player, 1; threads = false, merge = true, callback_move = cb, kwargs...)
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
      run_loops(ps[1])
    end
  else
    run_loops(p)
  end

  nothing
end




"""
    record_model(model, games [; use_features, augment])

Record a dataset by applying `model` to `games`. The features enabled for
`model` are recorded if `use_features` is true. `callback` is a function that is
called after each application of `model` to a game state.
"""
function record_model( model :: AbstractModel{G}
                     , games :: Vector{T}
                     ; use_features = true
                     , augment = false
                     ) where {G, T <: G}

  games = augment ? mapreduce(Game.augment, vcat, games) : games

  vplabel = Vector{Float32}[]
  flabel  = Vector{Float32}[]

  v, p, f = apply_features(model, game)

  for i in 1:length(games)
    push!(vplabel, vcat(v[i], p[:,i]))
    use_features ? push!(flabel, f[:,i]) : push!(flabel, similar(f[:,i], 0))
  end

  features = use_features ? Model.features(model) : Feature[]

  DataSet(games, vplabel, flabel; features)

end

# -------- Threaded Recording --------------------------------------------- #

function _threaded(handle :: Function, ps :: Vector; threads)

  bgthreads = Threads.nthreads() - 1
  @assert bgthreads > 0 "No background threads available"

  # TODO: Is this safe for any model?
  # Using threads is only safe for async models or if each thread
  # gets its own copy of the model
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


