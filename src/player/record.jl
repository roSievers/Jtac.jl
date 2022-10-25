

function _record( play :: Function # maps game to dataset
                , root
                , n :: Int
                ; features 
                , ntasks 
                , augment = true
                , callback = () -> nothing
                , merge = true
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

  # Return one big dataset or the different playings separately?
  merge ? Base.merge(ds...) : ds

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

  # Go over the unextended dataset
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
- `callback`: Procedure that is called afer each completed playing.
- `callback_move`: Procedure called after each individual move.
- `prepare`: Function applied on `game` that returns the initial state for
   a single selfplay. Can be used to randomize the initial positions, for
   example by random actions. Default preparation maps can be generated by
   `prepare(; steps).
- `branch`: Function applied to each game state of a first selfplay. It can
   return `nothing` or branched game states, which are then used as roots for
   new playings (without recursive branching). Default branching maps can be
   generated by `Util.branch(; prob, steps)`.
- `distributed = false`: Whether to conduct the self-playings on several
   processes in parallel. If `true`, all available workers are used. Alternatively,
   a list of worker ids can be passed.
- `tickets`: Number of chunks in which the workload is distributed if
   `distributed != false`. By default, it is set to the number of workers.

# Examples
```julia
# Record 20 self-playings of an classical MCTS player with power 250
G = Game.TicTacToe
player = Player.MCTSPlayer(power = 250)
dataset = Data.record(player, 20, game = G, branch = Util.branch(prob = 0.25))

# Record 10 self-playings of MCTS player with shallow predictor network and
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
               , distributed = false
               , tickets = nothing
               , callback_move = () -> nothing
               , kwargs... )

  # If self-playing is to be distributed, get the list of workers and
  # cede to the corresponding distributed function, defined below
  if distributed != false

    workers = distributed == true ? Distributed.workers() : distributed
    tickets = isnothing(tickets) ? length(workers) : tickets

    return record_distributed( p, n
                                  ; game = game
                                  , features = features
                                  , workers = workers
                                  , tickets = tickets
                                  , callback_move = callback_move
                                  , kwargs... )

  end

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
  _record(play, game, n; features = features, ntasks = Model.ntasks(p), kwargs...)

end


"""
    record_against(player, enemy [, n]; <keyword arguments>)

Record a dataset by letting `player` play against `enemy` a number of `n` times.
The datasets are recorded similarly to `record`.

# Arguments
The function takes the following arguments:
- `game`: Initial game type or state that is compatible with `player` and `enemy`.
- `features`: List of features for which feature labels are created.
- `start`: Function that determines the starting player (-1: enemy, 1: player).
- `augment`: Whether to apply symmetry augmentation on the generated dataset.
- `merge = true`: Whether to return one merged dataset or seperate playings.
- `callback`: Function that is called afer each finished game.
- `prepare`: Function applied on `game` that returns the initial state for
   a single playing. Can be used to randomize the initial positions, for
   example by random actions. Default preparation maps can be generated by
   `prepare(; steps).
- `branch`: Function applied to each game state of a first playing. It can
   return `nothing` or branched game states, which are then used as roots for
   new playings (without recursive branching). Default branching maps can be
   generated by `Util.branch(; prob, steps)`.
- `distributed = false`: Whether to conduct the playings on several processes in
   parallel. If `true`, all available workers are used. Alternatively, a list of
   worker ids can be passed.
- `tickets`: Number of chunks in which the workload is distributed if
   `distributed != false`. By default, it is set to the number of workers.
"""
function record_against( p :: AbstractPlayer
                       , enemy :: AbstractPlayer
                       , n :: Int = 1
                       ; game = Player.derive_gametype(p, enemy)
                       , start :: Function = () -> rand([-1, 1])
                       , features = features(p)
                       , distributed = false
                       , tickets = nothing
                       , kwargs... )

  # If playing is to be distributed, get the list of workers and
  # cede to the corresponding function in distributed.jl
  if distributed != false

    workers = distributed == true ? Distributed.workers() : distributed
    tickets = isnothing(tickets) ? length(workers) : tickets

    return record_against_distributed( p, enemy, n
                                     ; game = game
                                     , start = start
                                     , features = features
                                     , workers = workers
                                     , tickets = tickets
                                     , kwargs... )

  end

  play = game -> begin

    dataset = DataSet{typeof(game)}()

    # Roll the dice to see who starts. Our player (i = 1) or the enemy (i = -1)?
    s = start()

    while !is_over(game)

      # If it is our turn, record the move. Otherwise, let the enemy play
      play = (current_player(game) == s)
      play ? _record_move!(dataset, game, p) : turn!(game, enemy)

    end

    # Finish the dataset (except for features)
    _record_final!(dataset, game)
    _record_value!(dataset)

    dataset

  end

  ntasks = Model.ntasks(p) + Model.ntasks(enemy) - 1

  _record(play, game, n; features, ntasks, kwargs...)

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

# -------- Distributed Recording --------------------------------------------- #

function record_distributed( p :: AbstractPlayer
                                , n :: Int = 1
                                ; game = Player.derive_gametype(p)
                                , workers = workers()
                                , merge = true
                                , callback_move = () -> nothing
                                , kwargs... )

  # Create the record function
  f = (ps, n; game, callback_error, kwargs...) -> begin
    # Check for errors after each move
    callback_move = () -> (callback_error(); callback_move())
    game = Game.unfreeze(game)
    ds = record(ps[1], n; game, merge, callback_move, kwargs...) 
    Game.freeze(ds)
  end

  # Use the with_workers function defined in src/player/distributed.jl
  game = Game.freeze(game)
  ds = Player.with_workers(f, [p], n; workers, game, kwargs...)
  ds = vcat(ds...)

  # After transferring the datasets from other processes, make sure that they
  # are unfrozen
  ds = Game.unfreeze.(ds)

  merge ? Base.merge(ds...) : ds
end


function record_against_distributed( p :: AbstractPlayer
                                   , enemy :: AbstractPlayer
                                   , n :: Int = 1
                                   ; game = Player.derive_gametype(p, enemy)
                                   , workers = workers()
                                   , merge = true
                                   , callback_move = () -> nothing
                                   , kwargs... )

  # Create the record function
  f = (ps, n; game, callback_error, kwargs...) -> begin
    # Check for errors after each move
    callback_move = () -> (callback_error(); callback_move())
    game = Game.unfreeze(game)
    ds = record_against(ps[1], ps[2], n; game, merge, callback_move, kwargs...)
    Game.freeze(ds)
  end

  # Use the with_workers function defined in src/distributed.jl
  game = Game.freeze(game)
  ds = Player.with_workers(f, [p, enemy], n; workers, game, kwargs...)
  ds = vcat(ds...)

  # After transferring the datasets from other processes, make sure that they
  # are unfrozen
  ds = Game.unfreeze.(ds)

  merge ? Base.merge(ds) : ds
end

