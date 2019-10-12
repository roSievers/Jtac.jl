
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
  cache                               # cache that can store prepared data

end

"""
      DataSet{G}([; features])

Initialize an empty dataset for concrete game type `G` and `features` enabled.
"""
function DataSet{G}(; features = Feature[]) where {G <: Game}

  DataSet( Vector{G}()
         , Vector{Vector{Float32}}()
         , Vector{Vector{Float32}}()
         , features
         , nothing )

end

function DataSet(games, label, flabel; features = Feature[])

  DataSet(games, label, flabel, features, nothing)

end

features(ds :: DataSet) = ds.features
Base.length(d :: DataSet) = length(d.games)

# Cache binary data blocks used to swiftly calculate the loss
function prepare_data( ds :: DataSet{G}
                     ; gpu :: Bool
                     , use_features :: Bool
                     ) where {G}

  # Check if we have to update the cache
  if isnothing(ds.cache) || ds.cache.gpu != gpu ||
     ds.cache.use_features != use_features

    at = atype(gpu)

    data = convert(at, representation(ds.games))

    vplabel = hcat(ds.label...)

    vlabel = convert(at, vplabel[1, :])
    plabel = convert(at, vplabel[2:end, :])
    flabel = use_features ? convert(at, hcat(ds.flabel...)) : nothing

    ds.cache = ( gpu = gpu
               , use_features = use_features
               , data = data
               , vlabel = vlabel
               , plabel = plabel
               , flabel = flabel )

  end

  ds.cache

end


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

function Knet.minibatch( d :: DataSet{G}
                       , batchsize
                       ; shuffle = false
                       , partial = true
                       ) where {G}

  l = length(d)
  indices = shuffle ? Random.shuffle(1:l) : collect(1:l)
  batches = []

  i, j = 1, batchsize

  while max(i, j) <= l

    sel = indices[i:j]
    ds  = DataSet( d.games[sel], d.label[sel]
                 , d.flabel[sel], features = d.features )

    push!(batches, ds)

    i += batchsize
    j += partial ? min(batchsize, l - j) : batchsize

  end

  batches

end


# -------- Saving and Loading Datasets --------------------------------------- #

function save_dataset(fname :: String, d :: DataSet)
  
  # Temporarily disable the cache for saving
  cache = d.cache
  d.cache = nothing

  # Save the file
  BSON.bson(fname * ".jtd", dataset = dataset) 
  d.cache = cache

  nothing

end

load_dataset(fname :: String) = BSON.load(fname * ".jtd")[:dataset]


# -------- Generating Datasets: Helpers -------------------------------------- #

function _record( play :: Function # maps game to dataset
                , root :: G
                , n :: Int
                ; features 
                , ntasks 
                , augment = true
                , callback = () -> nothing
                , branching = 0.
                ) where {G <: Game}

  # Extend the provided play function by random branching 
  # and the call to callback
  play_with_branching = _ -> begin

    # Play through the game once without branching
    dataset = play(copy(root))
    
    # Create branchpoints and play them
    branchpoints = randsubseq(dataset.games[1:end-1], branching)
    branches = map(branchpoints) do game
      play(random_turn!(copy(game)))
    end

    # Filter branches where the random turn lead us to the end of the game
    filter!(x -> length(x) > 0, branches)

    # Collect the original and the branched datasets
    datasets = [dataset; branches]

    # Augment the datasets (if desired)
    if augment
      datasets = mapreduce(Jtac.augment, vcat, datasets)
    end
    
    # Calculate features (if there are any)
    datasets = map(d -> _record_features!(d, features), datasets)

    # Signal that we are done with one iteration
    callback()

    merge(datasets...)

  end

  # Call play_with_branching n times and merge all datasets
  if ntasks == 1 || n == 1

    # Asyncmap is assumed to be of disadvantage
    ds = merge(map(play_with_branching, 1:n)...)

  else

    # Asyncmap is assumed to be of advantage
    ds = merge(asyncmap(play_with_branching, 1:n, ntasks = ntasks)...)

  end

  # Mark the given features as activated
  ds.features = features

  ds

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
- `game`: Initial game that is compatible with `player`. Derived by default.
- `features`: List of features for which feature labels are created.
- `augment = true`: Whether to apply augmentation on the generated dataset.
- `branching = 0.`: Probability for random branching during the playthrough.
- `callback`: Function that is called afer each finished playing.

# Examples
```julia
# Record 20 self-playings of an classical MCTS player with power 250
player = MCTSPlayer(power = 250)
dataset = record_self(player, 20, game = TicTacToe(), branching = 0.25)

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
                    , kwargs...
                    ) :: DataSet{T} where {G, T <: G}

  play = game -> begin

    dataset = DataSet{T}()

    while !is_over(game)

      # Play one move, prepare the label, and record the game and player policy
      _record_move!(dataset, game, p)

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
- `branching`: Probability for random branching during the playthrough.
- `callback`: Function that is called afer each finished game.
"""
function record_against( p :: Player{G}
                       , enemy :: Player{H}
                       , n :: Int = 1
                       ; game :: T = G()
                       , start :: Function = () -> rand([-1, 1])
                       , features = features(p)
                       , kwargs...
                       ) :: DataSet{T} where {G, H, T <: typeintersect(G, H)}

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
    record_model(model, games [; use_features, callback])

Record a dataset by applying `model` to `games`. The features enabled for
`model` are recorded if `use_features` is true. `callback` is a function that is
called after each application of `model` to a game state.
"""
function record_model( model :: Model{G}
                     , games :: Vector{T}
                     ; use_features = true
                     , augment = false
                     , callback = () -> nothing
                     ) where {G, T <: G}

  games = augment ? mapreduce(Jtac.augment, vcat, games) : games

  label = map(games) do game
    callback()
    model(game, use_features)
  end

  features = use_features ? model.features : Features[]

  DataSet{T}(games, label, features = features)

end

