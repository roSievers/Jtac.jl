
"""
Auxiliary structure that holds the label data of one target for several games.
"""
struct LabelData
  data :: Vector{Vector{Float32}}
end

"""
  LabelData()

Create an empty label data object. Can be populated and modified via
[`Base.append!`](@ref), [`Base.push!`](@ref), [`Base.pop!`](@ref), and
[`Base.deleteat!`](@ref).
"""
LabelData() = LabelData([])

Pack.format(:: Type{LabelData}) = Pack.BinArrayFormat()
Pack.destruct(l :: LabelData, :: Pack.BinArrayFormat) = reduce(hcat, l.data)

function Pack.construct(:: Type{LabelData}, val, :: Pack.BinArrayFormat)
  data = reinterpret(Float32, val.data)
  data = reshape(data, val.size...)
  LabelData(collect.(eachcol(data)))
end

# Base methods
Base.length(l :: LabelData) = Base.length(l.data)
Base.copy(l :: LabelData) = LabelData(copy(l.data))

Base.append!(ld :: LabelData, ld2 :: LabelData) = append!(ld.data, ld2.data)
Base.push!(ld :: LabelData, args...) = push!(ld.data, args...)
Base.pop!(ld :: LabelData) = pop!(ld.data)
Base.deleteat!(ld :: LabelData, idx) = deleteat!(ld.data, idx)

Base.iterate(ld :: LabelData, args...) = iterate(ld.data, args...)
Base.getindex(ld :: LabelData, args...) = LabelData(getindex(ld.data, args...))

Base.convert(:: Type{LabelData}, data :: Vector{Vector{Float32}}) = LabelData(data)
Base.convert(:: Type{Vector{Vector{Float32}}}, ld :: LabelData) = ld.data


"""
Structure that contains a list of games and prediction target labels for
learning.

Dataset are usually created when an [`Player.MCTSPlayer`](@ref) plays matches against
itself, during which the target labels are evaluated on the resulting game
states (see [`Target.label`](@ref)).
"""
mutable struct DataSet{G <: AbstractGame}
  games  :: Vector{G}
  targets :: Vector{AbstractTarget{G}}
  target_names :: Vector{Symbol}
  target_labels :: Vector{LabelData}
end

@pack {<: DataSet} in TypedFormat{MapFormat}

"""
    DataSet(G)
    DataSet(G, targets)
    DataSet(G, names, targets)

Initialize an empty dataset. `G` is the supported game type and `targets` a
named tuple of the supported targets (defaults to `Target.defaulttargets(G)`).
If the target `names` are passed explicitly, `targets` can be any iterable with
elements [`AbstractTarget`](@ref).
"""
function DataSet(G :: Type{<: AbstractGame}, targets = Target.defaulttargets(G))
  names = collect(keys(targets))
  targets = AbstractTarget{G}[t for t in values(targets)]
  DataSet(G[], targets, names, [LabelData() for _ in 1:length(targets)])
end

function DataSet(G :: Type{<: AbstractGame}, names, targets)
  names = Symbol.(names)
  targets = AbstractTarget{G}[t for t in targets]
  @assert length(names) == length(targets) "Inconsistent number of targets"
  DataSet(G[], targets, names, [LabelData() for _ in 1:length(targets)])
end

"""
    DataSet(model)

Initialize an empty dataset, where the game type and supported targets are
derived from `model`.
"""
function DataSet(m :: AbstractModel{G}) where {G}
  DataSet(G, Target.targets(m))
end

Base.length(d :: DataSet) = length(d.games)
Base.lastindex(d :: DataSet) = length(d)

"""
    isconsistent(dataset)

Check whether a dataset is consistent, i.e., if all of its members have the
expected dimensions.
"""
function isconsistent(d :: DataSet)
  all([
    length(d.targets) == length(d.target_labels),
    all(map(length, d.target_labels) .== length(d.games)),
    all(1:length(d.targets)) do index
      all(length(d.targets[index]) .== length.(d.target_labels[index]))
    end
  ])
end

"""
    save(io, dataset)
    save(name, dataset)

Save `dataset` to the iostream `io`. If a path `name` is given, save it as a
file (with preferred extension ".jtd").
"""
function save(io :: IO, d :: DataSet) :: Nothing
  Pack.pack(io, d, Pack.StreamFormat(ZstdCompressorStream))
end

function save(fname :: AbstractString, d :: DataSet) :: Nothing
  open(io -> save(io, d), fname, "w")
end

"""
    load(io)
    load(name)

Load a dataset from the iostream `io` or file `name`.
"""
function load(io :: IO) :: DataSet
  Pack.unpack(io, DataSet, Pack.StreamFormat(ZstdDecompressorStream))
end

function load(fname :: String) :: DataSet
  open(io -> load(io), fname)
end

function Base.getindex(d :: DataSet, I)
  labels = [l[I] for l in d.target_labels]
  DataSet(d.games[I], d.targets, d.target_names, labels)
end

function Base.append!( a :: DataSet{G}
                     , b :: DataSet{G}
                     ) where {G <: AbstractGame}
  @assert all(a.targets .== b.targets) "Incompatible targets"
  @assert a.target_names == b.target_names "Incompatible target names"
  append!(a.games, b.games)
  foreach((al, bl) -> append!(al, bl), a.target_labels, b.target_labels)
end

function Base.merge(ds :: AbstractVector{DataSet{G}}) where {G <: AbstractGame}
  # @assert length(ds) > 0 "Cannot merge empty vector of datasets"
  dataset = DataSet(G, ds[1].target_names, ds[1].targets)
  foreach(d -> append!(dataset, d), ds)
  dataset
end

function Base.split(d :: DataSet{G}, maxsize :: Int; shuffle = false) where {G}
  n = length(d)
  if maxsize >= n
    [d]
  else
    idx = shuffle ? randperm(n) : (1:n)
    pos = [collect(1:maxsize:n); n+1]
    Is = [idx[i:j-1] for (i,j) in zip(pos[1:end-1], pos[2:end])]
    map(I -> d[I], Is)
  end
end

function Base.deleteat!(d :: DataSet, idx)
  deleteat!(d.games, idx)
  foreach(d.target_labels) do label
    deleteat!(label, idx)
  end
end

"""
    targets(dataset)

Return the targets stored in `dataset`.
"""
targets(args...; kwargs...) = Target.targets(args...; kwargs...)

Target.targets(ds :: DataSet) = (; zip(ds.target_names, ds.targets)...)

"""
    targetnames(dataset)

Return the names of the targets stored in `dataset`.
"""
targetnames(args...; kwargs...) = Target.targetnames(args...; kwargs...)

Target.targetnames(ds :: DataSet) = ds.target_names

# TODO: operation to replace different labels for same state by average labels

function Base.show(io :: IO, d :: DataSet{G}) where G <: AbstractGame
  if isempty(d.targets)
    targets = "[]"
  else
    targets = "[" * join(d.target_names, ", ") * "]"
  end
  print(io, "DataSet{$(Game.name(G))}($(length(d)), targets = $targets)")
end

Base.show(io :: IO, :: MIME"text/plain", d :: DataSet) = show(io, d)

function Random.rand(rng, d :: DataSet, n :: Int)
  indices = Random.rand(rng, 1:length(d), n)
  d[indices]
end

"""
    diversity(dataset :: DataSet)

Print out information about the diversity of data set `dataset`
"""
function diversity(d :: DataSet{G}) where {G}
  dict = Dict{G, Any}()
  for index in eachindex(d.games)
    game = d.games[index]
    targets = [x.data[index] for x in d.target_labels]
    if game in keys(dict)
      entry = dict[game]
      dict[game] = (;
        count = entry.count + 1,
        targets = push!(entry.targets, targets),
      )
    else
      dict[game] = (;
        count = 1,
        targets = [targets],
      )
    end
  end

  @printf "states total: %d\n" length(d)
  @printf "states distinct: %d\n" length(dict)
  counts = map(x -> x.count, values(dict))
  freqs = map(c -> count(counts .== c), 1:maximum(counts))
  print("count: ")
  for count in maximum(counts):-1:1
    freqs[count] > 0 && @printf "%5d" count
  end
  print("\nstates:")
  for count in maximum(counts):-1:1
    freqs[count] > 0 && @printf "%5d" freqs[count]
  end
  println()
end


"""
Structure that stores tensorized [`DataSet`](@ref)s.

Used during the training process of [`NeuralModel`](@ref)s. Data sets are first
converted to data caches, then split into batches (see [`DataBatches`](@ref))
for gradient descent steps.
"""
struct DataCache{G <: AbstractGame, T}
  data :: T
  targets :: Vector{AbstractTarget}
  target_names :: Vector{Symbol}
  target_labels :: Vector{T}
end

"""
    DataCache{G}(T, data, labels)

Create a [`DataCache`](@ref) for games of type `G` with array type `T`, game
data `data`, and label data `labels`.
"""
function DataCache{G}(T, data, targets, names, labels) where {G}
  @assert length(labels) == length(targets) == length(labels) """
  Inconsistent target data.
  """
  data = convert(T, data)
  labels = map(labels) do label
    convert(T, label)
  end
  DataCache{G, T}(data, targets, names, labels)
end

"""
    DataCache(T, dataset)

Convert a data set `dataset` to a [`DataCache`](@ref) with array type `T`.
"""
function DataCache(T, ds :: DataSet{G}) where {G <: AbstractGame}
  data = Game.array(ds.games)
  labels = map(ds.target_labels) do label
    hcat(label...)
  end
  DataCache{G}(T, data, ds.targets, ds.target_names, labels)
end

Base.length(c :: DataCache) = size(c.data)[end]

function Model.releasememory!(c :: DataCache)
  Model.releasememory!(c.data)
  foreach(Model.releasememory!, c.target_labels)
end

Target.targets(c :: DataCache) = (; zip(c.target_names, c.targets)...)

"""
Wrapper of a [`DataSet`](@ref) that splits the data set into
[`DataCache`](@ref)s of a given batchsize. Can be iterated over.
"""
struct DataBatches{G <: AbstractGame, T}
  dataset :: DataSet{G}
  batchsize :: Int
  shuffle :: Bool
  partial :: Bool
end

"""
State object that keeps track of interna during iterations over
[`DataBatches`](@ref).
"""
struct DataBatchesState{G <: AbstractGame, T}
  indices :: Vector{Int}
  startindex :: Int
  prev_cache :: Union{Nothing, DataCache{G, T}}
end

"""
    DataBatches(T, dataset, batchsize; shuffle = false, partial = true)

Create a batch iterator over `dataset`, yielding [`DataCache{G, T}`](@ref)
objects of length `batchsize` on iteration, where `G` is the game type of
`dataset`. If `shuffle = true`, each iteration draws data in random order.
If `partial = true`, the final batch may be smaller than `batchsize`.
"""
function DataBatches( T
                    , d :: DataSet{G}
                    , batchsize
                    ; shuffle = false
                    , partial = true
                    ) where {G <: AbstractGame}

  DataBatches{G, T}(d, batchsize, shuffle, partial)
end

function Base.length(b :: DataBatches)
  n = length(b.dataset) / b.batchsize
  b.partial ? ceil(Int, n) : floor(Int, n)
end

function Base.iterate( b :: DataBatches{G, T}
                     , state = nothing
                     ) where {G <: AbstractGame, T}

  len = length(b.dataset)

  if isnothing(state)
    indices = b.shuffle ? randperm(len) : 1:len
    state = DataBatchesState{G, T}(indices, 1, nothing)
  else
    Model.releasememory!(state.prev_cache)
  end
  
  startindex = state.startindex
  stopindex = min(startindex + b.batchsize - 1, len)

  stopiter = startindex > len
  if !b.partial
    stopiter |= stopindex - startindex < b.batchsize - 1
  end

  if !stopiter
    I = state.indices[startindex:stopindex]
    subset = b.dataset[I]
    cache = DataCache(T, subset)
    cache, DataBatchesState(state.indices, stopindex + 1, cache)
  end
end
