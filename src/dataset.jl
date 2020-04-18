
# -------- Datasets ---------------------------------------------------------- #

"""
Structure that holds a list of games and labels, i.e., targets for learning.

Dataset are usually created through playing games from start to finish by an
MCTSPlayer. The value label corresponds to the results of the game, and the
policy label to the improved policy in the single MCTS steps. Furthermore,
features can be enabled for a dataset and are stored as part of the label, as
well.
"""
mutable struct DataSet{G <: Game}

  games  :: Vector{G}                 # games saved in the dataset
  label  :: Vector{Vector{Float32}}   # target value/policy labels
  flabel :: Vector{Vector{Float32}}   # target feature values

  features :: Vector{Feature}         # with which features was the ds created

end

"""
      DataSet{G}([; features])

Initialize an empty dataset for concrete game type `G` and `features` enabled.
"""
function DataSet{G}(; features = Feature[]) where {G <: Game}

  DataSet( Vector{G}()
         , Vector{Vector{Float32}}()
         , Vector{Vector{Float32}}()
         , features )

end

function DataSet(games, label, flabel; features = Feature[])

  DataSet(games, label, flabel, features)

end

features(ds :: DataSet) = ds.features
Base.length(d :: DataSet) = length(d.games)

# -------- Dataset Operations ------------------------------------------------ #

function Base.merge(d :: DataSet{G}, ds...) where {G <: Game}

  # Make sure that all datasets have compatible features
  features = d.features

  if !all(x -> x.features == features, ds) 

    error("Merging databases with incompatible features")

  end

  # Create and return the merged dataset
  dataset = DataSet{G}()
  dataset.features = features

  dataset.games  = vcat([d.games,  (x.games  for x in ds)...]...)
  dataset.label  = vcat([d.label,  (x.label  for x in ds)...]...)
  dataset.flabel = vcat([d.flabel, (x.flabel for x in ds)...]...)

  dataset

end

function Base.split(d :: DataSet{G}, size :: Int; shuffle = true) where {G}

  n = length(d)
  @assert size <= n "Cannot split dataset of length $n at position $size."

  idx = shuffle ? randperm(n) : 1:n
  idx1, idx2 = idx[1:size], idx[size+1:end]

  d1 = DataSet( d.games[idx1], d.label[idx1]
              , d.flabel[idx1], features = d.features)
  d2 = DataSet( d.games[idx2], d.label[idx2]
              , d.flabel[idx2], features = d.features)

  d1, d2

end

function augment(d :: DataSet{G}) :: Vector{DataSet{G}} where G <: Game

  # NOTE: Augmentation will render flabel information useless.
  # Therefore, one must recalculate them after applying this function!

  # Problem: we must augment d in such a way that playthroughs are still
  # discernible after augmentation. I.e., for each symmetry transformation, one
  # DataSet will be returned.

  gs, ls = unzip([augment(g, l) for (g, l) in zip(d.games, d.label)])

  map(1:length(gs[1])) do j

    games = map(x -> x[j], gs)
    label = map(x -> x[j], ls)

    DataSet(games, label, copy(d.flabel), features = d.features)

  end

end

# -------- Raw DataSet Representation: Caches -------------------------------- #

struct DataCache{G <: Game, GPU}

  data        # game representation data

  vlabel      # value labels
  plabel      # policy labels
  flabel      # feature labels

end

function DataCache( ds :: DataSet{G}
                  ; gpu = false
                  , use_features = false
                  ) where {G <: Game}

  # Preparation
  at = atype(gpu)
  vplabel = hcat(ds.label...)

  # Convert to at
  data = convert(at, representation(ds.games))
  vlabel = convert(at, vplabel[1, :])
  plabel = convert(at, vplabel[2:end, :])
  flabel = use_features ? convert(at, hcat(ds.flabel...)) : nothing

  DataCache{G, gpu}(data, vlabel, plabel, flabel)

end

Base.length(c :: DataCache) = size(c.data)[end]

# -------- Iterating Datasets: Batches --------------------------------------- #

struct Batches{G <: Game, GPU}

  cache :: DataCache{G, GPU}

  batchsize :: Int
  shuffle :: Bool
  partial :: Bool

  indices :: Vector{Int}

end

function Batches( d :: DataSet{G}
                , batchsize
                ; shuffle = false
                , partial = true
                , gpu = false 
                , use_features = false
                ) where {G <: Game}

  indices = collect(1:length(d))

  cache = DataCache(d, gpu = gpu, use_features = use_features)

  Batches{G, gpu}(cache, batchsize, shuffle, partial, indices)

end

function Base.length(b :: Batches)
  n = length(b.cache) / b.batchsize
  b.partial ? ceil(Int, n) : floor(Int, n)
end

function Base.iterate(b :: Batches{G, GPU}, start = 1) where {G <: Game, GPU}

  # Preparations
  l = length(b.cache)
  b.shuffle && start == 1 && (b.indices .= randperm(l))

  # start:stop is the range in b.indices that selected
  stop = min(start + b.batchsize - 1, l)

  # Check for end of iteration
  if start > l || !b.partial && stop - start < b.batchsize - 1
    return nothing
  end

  # Build the data cache
  idx = b.indices[start:stop]

  data = b.cache.data[:, :, :, idx]
  vlabel = b.cache.vlabel[idx]
  plabel = b.cache.plabel[:, idx]
  flabel = isnothing(b.cache.flabel) ? nothing : b.cache.flabel[:, idx]

  cache = DataCache{G, GPU}(data, vlabel, plabel, flabel)

  # Return the (cache, new_start) state tuple
  cache, stop + 1

end

# -------- Generating Datasets: Helpers -------------------------------------- #

function _record( play :: Function # maps game to dataset
                , root :: G
                , n :: Int
                ; features 
                , ntasks 
                , augment = true
                , callback = () -> nothing
                , merge = true
                , prepare = prepare(steps = 0)
                , branch = branch(prob = 0, steps = 1)
                ) where {G <: Game}


  # Extend the provided play function by random branching 
  # and the call to callback
  play_with_branching = _ -> begin

    # Play the (prepared) game once without branching inbetween
    dataset = play(prepare(root))
    
    # Create branchpoints and play them, too
    branchpoints = filter(!isnothing, branch.(dataset.games[1:end-1]))
    branches = play.(branchpoints)

    # Filter branches where we were directly lead to the end of the game
    filter!(x -> length(x) > 0, branches)

    # Collect the original and the branched datasets
    datasets = [dataset; branches]

    # Augment the datasets (if desired)
    augment && (datasets = mapreduce(Jtac.augment, vcat, datasets))
    
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
                      ) where {G <: Game}

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

function _record_final!( dataset :: DataSet{G}
                      , game ) where {G <: Game}

  pl = policy_length(G)
  push!(dataset.games, copy(game))
  push!(dataset.label, zeros(Float32, 1 + pl))

end


function _record_value!(dataset :: DataSet{G}) where {G <: Game}

  result = status(dataset.games[end])

  for i = 1:length(dataset.games)

    dataset.label[i][1] = current_player(dataset.games[i]) * result

  end

end

# Add feature label to an extended dataset (that contains the final game state)
function _record_features!( dataset :: DataSet{G}
                          , features
                          ) where {G <: Game}

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
    record_self(player [, n]; <keyword arguments>)

Record a dataset by letting `player` play against itself `n` times.

The value label for the dataset is derived from the outcome of the game (-1, 0,
or +1). The policy label is the policy proposed by the player for the respective
game state. The feature labels are calculated by applying the provided features
(see keyword arguments) to each stored game state and the corresponding history.

# Arguments
- `game`: Initial game state that is compatible with `player`. Derived by default.
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
   generated by `branch(; prob, steps)`.
- `distributed = false`: Whether to conduct the self-playings on several
   processes in parallel. If `true`, all available workers are used. Alternatively,
   a list of worker ids can be passed.
- `tickets`: Number of chunks in which the workload is distributed if
   `distributed != false`. By default, it is set to the number of workers.

# Examples
```julia
# Record 20 self-playings of an classical MCTS player with power 250
player = MCTSPlayer(power = 250)
dataset = record_self(player, 20, game = TicTacToe(), branch = branch(0.25))

# Record 10 self-playings of MCTS player with shallow predictor network and
# power 50
model = NeuralModel(TicTacToe, @chain TicTacToe Dense(50))
player = MCTSPlayer(model, power = 50)
dataset = record_self(player, 10, augment = false)
```
"""
function record_self( p :: Player{G}
                    , n :: Int = 1
                    ; game :: T = G()
                    , features = features(p)
                    , distributed = false
                    , tickets = nothing
                    , callback_move = () -> nothing
                    , kwargs...
                    ) where {G, T <: G}

  # If self-playing is to be distributed, get the list of workers and
  # cede to the corresponding distributed function, defined below
  if distributed != false

    workers = distributed == true ? Distributed.workers() : distributed
    tickets = isnothing(tickets) ? length(workers) : tickets

    return record_self_distributed( p, n
                                  ; game = game
                                  , features = features
                                  , workers = workers
                                  , tickets = tickets
                                  , callback_move = callback_move
                                  , kwargs... )

  end

  # Function that plays a single game of player against itself
  play = game -> begin

    dataset = DataSet{T}()

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
  _record(play, game, n; features = features, ntasks = ntasks(p), kwargs...)

end


"""
    record_against(player, enemy [, n]; <keyword arguments>)

Record a dataset by letting `player` play against `enemy` a number of `n` times.
The datasets are recorded similarly to `record_self`.

# Arguments
The function takes the following arguments:
- `game`: Initial game state that is compatible with `player`.
- `features`: List of features for which feature labels are created.
- `start`: Function that determines the starting player (-1: enemy, 1: player).
- `augment`: Whether to apply symmetry augmentation on the generated dataset.
- `merge = false`: Whether to return one merged dataset or seperate playings.
- `callback`: Function that is called afer each finished game.
- `prepare`: Function applied on `game` that returns the initial state for
   a single playing. Can be used to randomize the initial positions, for
   example by random actions. Default preparation maps can be generated by
   `prepare(; steps).
- `branch`: Function applied to each game state of a first playing. It can
   return `nothing` or branched game states, which are then used as roots for
   new playings (without recursive branching). Default branching maps can be
   generated by `branch(; prob, steps)`.
- `distributed = false`: Whether to conduct the playings on several processes in
   parallel. If `true`, all available workers are used. Alternatively, a list of
   worker ids can be passed.
- `tickets`: Number of chunks in which the workload is distributed if
   `distributed != false`. By default, it is set to the number of workers.
"""
function record_against( p :: Player{G}
                       , enemy :: Player{H}
                       , n :: Int = 1
                       ; game :: T = G()
                       , start :: Function = () -> rand([-1, 1])
                       , features = features(p)
                       , distributed = false
                       , tickets = nothing
                       , kwargs...
                       ) where {G, H, T <: typeintersect(G, H)}

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

    dataset = DataSet{T}()

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

  ntasks = Jtac.ntasks(p) + Jtac.ntasks(enemy) - 1

  _record(play, game, n; features = features, ntasks = ntasks, kwargs...)

end


"""
    record_model(model, games [; use_features])

Record a dataset by applying `model` to `games`. The features enabled for
`model` are recorded if `use_features` is true. `callback` is a function that is
called after each application of `model` to a game state.
"""
function record_model( model :: Model{G}
                     , games :: Vector{T}
                     ; use_features = true
                     , augment = false
                     ) where {G, T <: G}

  games = augment ? mapreduce(Jtac.augment, vcat, games) : games

  vplabel = Vector{Float32}[]
  flabel  = Vector{Float32}[]

  v, p, f = apply_features(model, game)

  for i in 1:length(games)
    push!(vplabel, vcat(v[i], p[:,i]))
    use_features ? push!(flabel, f[:,i]) : push!(flabel, similar(f[:,i], 0))
  end

  features = use_features ? Jtac.features(model) : Feature[]

  DataSet(games, vplabel, flabel, features = features)

end

# -------- Distributed Recording --------------------------------------------- #

function record_self_distributed( p :: Player
                                , n :: Int = 1
                                ; workers = workers()
                                , merge = false
                                , kwargs... )

  # Create the record function
  record = (ps, n; kwargs...) -> begin
    record_self(ps[1], n; merge = merge, kwargs...)
  end

  # Use the with_workers function defined in src/distributed.jl
  ds = with_workers(record, [p], n; workers = workers, kwargs...)
  ds = vcat(ds...)

  merge ? Base.merge(ds) : ds

end


function record_against_distributed( p :: Player
                                   , enemy :: Player
                                   , n :: Int = 1
                                   ; workers = workers()
                                   , merge = false
                                   , kwargs... )

  # Create the record function
  record = (ps, n; kwargs...) -> begin
    record_against(ps[1], ps[2], n; merge = merge, kwargs...)
  end

  # Use the with_workers function defined in src/distributed.jl
  ds = with_workers(record, [p, enemy], n; workers = workers, kwargs...)
  ds = vcat(ds...)

  merge ? Base.merge(ds) : ds

end


