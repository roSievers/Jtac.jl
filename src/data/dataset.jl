
# -------- Datasets ---------------------------------------------------------- #

const LabelData = Vector{Vector{Float32}}

"""
Structure that holds a list of games and labels, i.e., prediction targets for
learning.

Dataset are usually created through playing games from start to finish by an
MCTSPlayer. The value label corresponds to the results of the game, and the
policy label to the improved policy in the single MCTS steps.
Enabled prediction targets are stored as well.
"""
mutable struct DataSet{G <: AbstractGame}
  games  :: Vector{G}
  labels :: Vector{LabelData}
  targets :: Vector{PredictionTarget{G}}
end

Pack.register(DataSet)
Pack.@mappack DataSet

function Pack.destruct(ds :: DataSet)
  label = reduce(vcat, reduce.(vcat, ds.labels))
  bytes = reinterpret(UInt8, label)
  Dict{String, Any}(
      "games" => Pack.Bytes(Pack.pack(ds.games))
    , "labels" => Pack.Bytes(bytes)
    , "targets" => Pack.Bytes(Pack.pack(ds.targets))
  )
end

function split_array(x, lengths)
  res = LabelData()
  start = firstindex(x)
  for len in lengths
    push!(res, x[start:(start+len-1)])
    start += len
  end
  res
end

function Pack.construct(:: Type{DataSet{G}}, d :: Dict) where {G}
  games = Pack.unpack(d["games"], Vector{G})
  targets = Pack.unpack(d["targets"], Vector{PredictionTarget{G}})

  data = reinterpret(Float32, d["labels"])
  lengths = length.(targets) .* length(games)

  @assert length(data) == sum(lengths)

  labels = map(split_array(data, lengths)) do arr
    arr = reshape(arr, :, length(games))
    collect.(eachcol(arr))
  end

  Data.DataSet(games, labels, targets)
end

function Pack.freeze(d :: DataSet{G}) where {G}
  DataSet{G}(Pack.freeze.(d.games), d.labels, d.targets)
end

function Pack.unfreeze(d :: DataSet{G}) where {G}
  DataSet{G}(Pack.unfreeze.(d.games), d.labels, d.targets)
end


"""
      DataSet(G, targets)
      DataSet(model)

Initialize an empty dataset for concrete game type `G` and `targets` enabled.
The second method constructs a DataSet that is compatible to `model`.
"""
DataSet(G :: Type{<: AbstractGame}, targets = []) =
  DataSet( Vector{G}()
         , [LabelData() for _ in 1:length(targets)]
         , convert(Vector{PredictionTarget{G}}, targets) )

function DataSet(m :: AbstractModel{G}) where {G}
  targets = Target.targets(training_model(m))
  DataSet(G, targets)
end

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

Base.getindex(d :: DataSet{G}, I) where {G <: AbstractGame} =
  DataSet{G}(d.games[I], [l[I] for l in d.labels], d.targets)

function Base.append!(d :: DataSet{G}, dd :: DataSet{G}) where {G <: AbstractGame}

  td = Target.targets(d)
  tdd = Target.targets(dd)
  compatible =
    length(td) == length(tdd) && all(map(Target.compatible, td, tdd))

  @assert compatible "Cannot append dataset with incompatible targets"

  append!(d.games, dd.games)
  foreach(d.labels, dd.labels) do l, ll
    append!(l, ll)
  end

  @assert check_consistency(d)

end

function Base.merge(d :: DataSet{G}, ds...) where {G <: AbstractGame}

  ts = [Target.targets(d), Target.targets.(ds)...]

  compatible =
    all(length(ts[1]) .== length.(ts)) && all(map(Target.compatible, ts...))

  # Create and return the merged dataset
  dataset = DataSet(G)
  dataset.targets = Target.targets(d)

  dataset.games  = vcat([d.games,  (x.games  for x in ds)...]...)
  for i in 1:length(dataset.targets)
    label = vcat((d.labels[i],  (x.labels[i] for x in ds)...)...)
    push!(dataset.labels, label)
  end

  @assert check_consistency(dataset)

  dataset

end

function Base.split(d :: DataSet{G}, size :: Int; shuffle = true) where {G}

  n = length(d)
  @assert size <= n "Cannot split dataset of length $n at position $size."

  idx = shuffle ? randperm(n) : 1:n
  idx1, idx2 = idx[1:size], idx[size+1:end]

  d1 = DataSet{G}(d.games[idx1], d.labels[idx1], d.targets)
  d2 = DataSet{G}(d.games[idx2], d.labels[idx2], d.targets)

  d1, d2

end

function Game.augment(d :: DataSet{G}) :: Vector{DataSet{G}} where G <: AbstractGame


  # This function will usually be called on not yet fully constructed datasets:
  # *after* value and policy targets have been recorded but *before* all
  # optional targets are recorded. Therfore, we only care about value and policy
  # targets and *reset* the rest of the label data
  @assert d.targets[1] isa ValueTarget "Cannot augment dataset with optional targets"
  @assert d.targets[2] isa PolicyTarget "Cannot augment dataset with optional targets"

  # We must augment d in such a way that playthroughs are still discernible
  # after augmentation. For each symmetry transformation, one DataSet will be
  # returned.

  values = d.labels[1]
  policies = d.labels[2]
  gps = [augment(g, p) for (g, p) in zip(d.games, policies)]
  gs, ps = first.(gps), last.(gps)

  map(1:length(gs[1])) do j

    games = map(x -> x[j], gs)
    policies = map(x -> x[j], ps)
    values = copy(values)

    labels = [values, policies, (LabelData() for _ in 1:length(d.labels)-2)...]

    DataSet(games, labels, d.targets)
  end
end

function check_consistency(d :: DataSet)
  length(d.targets) == length(d.labels) &&
  all(map(length, d.labels) .== length(d.games)) &&
  all(1:length(d.targets)) do i
    all(length(d.targets[i]) .== length.(d.labels[i]))
  end
end

Target.targets(ds :: DataSet) = ds.targets

function Target.adapt(d :: DataSet, targets)
  idx = Target.adapt(d.targets, targets)
  DataSet(d.games, d.labels[idx], d.targets[idx])
end


function Base.show(io :: IO, d :: DataSet{G}) where G <: AbstractGame
  if isempty(d.targets)
    targets = "[]"
  else
    targets = Target.name.(d.targets) |> string
  end
  print(io, "DataSet{$(Game.name(G))}($(length(d)), targets = $targets)")
end

Base.show(io :: IO, :: MIME"text/plain", d :: DataSet) = show(io, d)

function Random.rand(rng, d :: DataSet, n :: Int)
  indices = Random.rand(rng, 1:length(d), n)
  d[indices]
end

