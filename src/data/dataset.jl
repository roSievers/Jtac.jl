
# -------- Datasets ---------------------------------------------------------- #

"""
Structure that holds a list of games and labels, i.e., targets for learning.

Dataset are usually created through playing games from start to finish by an
MCTSPlayer. The value label corresponds to the results of the game, and the
policy label to the improved policy in the single MCTS steps. Furthermore,
features can be enabled for a dataset and are stored as part of the label, as
well.
"""
mutable struct DataSet{G <: AbstractGame}

  games  :: Vector{G}                 # games saved in the dataset
  label  :: Vector{Vector{Float32}}   # target value/policy labels
  flabel :: Vector{Vector{Float32}}   # target feature values

  features :: Vector{Feature}         # with which features was the ds created

end

Pack.register(DataSet)
Pack.@mappack DataSet

function Pack.destruct(ds :: Data.DataSet)
  label = vcat(ds.label...)
  flabel = vcat(ds.flabel...)
  bytes = reinterpret(UInt8, label)
  fbytes = reinterpret(UInt8, flabel)
  Dict{String, Any}(
      "games" => Pack.Bytes(Pack.pack(ds.games))
    , "label" => Pack.Bytes(bytes)
    , "flabel" => Pack.Bytes(fbytes)
    , "features" => ds.features
  )
end

function Pack.construct(:: Type{Data.DataSet{G}}, d :: Dict) where {G}
  games = Pack.unpack(d["games"], Vector{G})

  data = reinterpret(Float32, d["label"])
  data = reshape(data, :, length(games))
  label = [ col[:] for col in eachcol(data) ]

  fdata = reinterpret(Float32, d["flabel"])
  fdata = reshape(fdata, :, length(games))
  flabel = [ col[:] for col in eachcol(fdata) ]

  features = Pack.from_msgpack(Vector{Model.Feature}, d["features"])
  Data.DataSet(games, label, flabel, features)
end

"""
      DataSet{G}([; features])

Initialize an empty dataset for concrete game type `G` and `features` enabled.
"""
function DataSet{G}(; features = Feature[]) where {G <: AbstractGame}

  DataSet( Vector{G}()
         , Vector{Vector{Float32}}()
         , Vector{Vector{Float32}}()
         , features )

end

function DataSet(games, label, flabel; features = Feature[])

  DataSet(games, label, flabel, features)

end

Model.features(ds :: DataSet) = ds.features
Base.length(d :: DataSet) = length(d.games)
Base.lastindex(d :: DataSet) = length(d)

# -------- Serialization and IO of Datasets ---------------------------------- #


"""
    save(name, dataset)

Save `dataset` under filename `name`. Dataset caches are not saved.
"""
function save(fname :: String, d :: DataSet)
  open(io -> Pack.pack_compressed(io, d), fname, "w")
end

"""
    load(name)

Load a dataset for games from file `name`.
"""
function load(fname :: String) where G <: AbstractGame
  open(io -> Pack.unpack_compressed(io, DataSet), fname)
end

# -------- DataSet Operations ------------------------------------------------ #

function Base.getindex(d :: DataSet{G}, I) where {G <: AbstractGame}
  DataSet{G}(d.games[I], d.label[I], d.flabel[I], d.features)
end

function Base.append!(d :: DataSet{G}, dd :: DataSet{G}) where {G <: AbstractGame}

  if d.features != dd.features

    error("Appending dataset with incompatible features")

  end

  append!(d.games, dd.games)
  append!(d.label, dd.label)
  append!(d.flabel, dd.flabel)

end

function Base.merge(d :: DataSet{G}, ds...) where {G <: AbstractGame}

  # Make sure that all datasets have compatible features
  features = d.features

  if !all(x -> x.features == features, ds) 

    error("Merging datasets with incompatible features")

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

function Game.augment(d :: DataSet{G}) :: Vector{DataSet{G}} where G <: AbstractGame

  # NOTE: Augmentation will render flabel information useless.
  # Therefore, one must recalculate them after applying this function!

  # Problem: we must augment d in such a way that playthroughs are still
  # discernible after augmentation. I.e., for each symmetry transformation, one
  # DataSet will be returned.

  aug = [augment(g, l) for (g, l) in zip(d.games, d.label)]
  gs, ls = map(x -> x[1], aug), map(x -> x[2], aug)

  map(1:length(gs[1])) do j

    games = map(x -> x[j], gs)
    label = map(x -> x[j], ls)

    DataSet(games, label, copy(d.flabel), features = d.features)
  end
end

function Pack.freeze(d :: DataSet)
  DataSet(Pack.freeze.(d.games), d.label, d.flabel, d.features)
end

#function Pack.freeze(ds :: Vector{D}) where D <: DataSet
#  Pack.freeze.(ds)
#end

function Pack.unfreeze(d :: DataSet)
  DataSet(Pack.unfreeze.(d.games), d.label, d.flabel, d.features)
end

function Base.show(io :: IO, d :: DataSet{G}) where G <: AbstractGame
  n = length(d.features)
  features = n == 1 ? "1 feature" : "$n features"
  print(io, "DataSet{$(Game.name(G))}($(length(d)) elements, $features)")
end

function Base.show(io :: IO, :: MIME"text/plain", d :: DataSet{G}) where G <: AbstractGame
  n = length(d.features)
  features = n == 1 ? "1 feature" : "$n features"
  print(io, "DataSet{$(Game.name(G))} with $(length(d)) elements and $features")
end

function Random.rand(rng, d :: DataSet, n :: Int)
  indices = Random.rand(rng, 1:length(d), n)
  d[indices]
end



