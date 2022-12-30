
# -------- Datasets ---------------------------------------------------------- #

struct LabelData
  data :: Vector{Vector{Float32}}
end

Pack.@untyped LabelData

Pack.fieldnames(:: Type{LabelData}) = [:length, :bytes]
Pack.fieldtypes(:: Type{LabelData}) = [Int, Pack.Bytes]
Pack.fieldvalues(l :: LabelData) = [length(l.data), Pack.Bytes(reduce(vcat, l.data))]

function Pack.construct(:: Type{LabelData}, length, bytes)
  data = reinterpret(Float32, bytes.data)
  data = reshape(data, :, Int(length))
  LabelData(collect.(eachcol(data)))
end

LabelData() = LabelData([])

Base.length(ld :: LabelData) = Base.length(ld.data)
Base.copy(ld :: LabelData) = LabelData(copy(ld.data))

Base.append!(ld :: LabelData, ld2 :: LabelData) = append!(ld.data, ld2.data)
Base.push!(ld :: LabelData, args...) = push!(ld.data, args...)
Base.pop!(ld :: LabelData) = pop!(ld.data)
Base.deleteat!(ld :: LabelData, idx) = deleteat!(ld.data, idx)

Base.iterate(ld :: LabelData, args...) = iterate(ld.data, args...)
Base.getindex(ld :: LabelData, args...) = getindex(ld.data, args...)

Base.convert(:: Type{LabelData}, data :: Vector{Vector{Float32}}) = LabelData(data)
Base.convert(:: Type{Vector{Vector{Float32}}}, ld :: LabelData) = ld.data

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

Pack.@typed DataSet

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
  targets = Target.targets(m)
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
function load(fname :: String)
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

function Base.merge(ds :: Vector{DataSet{G}}) where {G <: AbstractGame}

  @assert length(ds) > 0 "Cannot merge empty DataSets"

  d = ds[1]

  compatible = all(ds) do x
    td = Target.targets(d)
    tx = Target.targets(x)
    length(d) == length(x) && all(Target.compatible.(td, tx))
  end

  # Create and return the merged dataset
  dataset = DataSet(G, Target.targets(d))

  for d in ds
    append!(dataset.games, d.games)
    for i in 1:length(dataset.targets)
      append!(dataset.labels[i], d.labels[i])
    end
  end

  @assert check_consistency(dataset)

  dataset

end

function Base.split(d :: DataSet{G}, maxsize :: Int; shuffle = false) where {G}

  n = length(d)
  maxsize >= n && return [d]

  idx = shuffle ? randperm(n) : 1:n
  pos = [collect(1:maxsize:n); n+1]
  idxs = [idx[i:j-1] for (i,j) in zip(pos[1:end-1], pos[2:end])]

  map(idxs) do idx
    labels = map(1:length(d.targets)) do i
      d.labels[i][idx]
    end
    DataSet{G}(d.games[idx], labels, d.targets)
  end
end

function Base.deleteat!(d :: DataSet, idx)
  deleteat!(d.games, idx)
  foreach(d.labels) do label
    deleteat!(label, idx)
  end
end

function Game.augment(d :: DataSet{G}) :: Vector{DataSet{G}} where G <: AbstractGame

  length(d) == 0 && return [d]

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

    labels = [values, LabelData(policies), (LabelData() for _ in 1:length(d.labels)-2)...]

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

# replace different labels for same state by average labels
# TODO
#function sanitize(d :: DataSet)
#end


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

