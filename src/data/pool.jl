
"""
Data structure for a finite capacity pool of training data that can be tagged
with metadata, like an age or a usage index. Provides functionality to select
and discard training data based on a user defined quality criterion. A quality
value smaller or equal to 0 flags the corresponding game state for removal
"""
mutable struct Pool{G <: AbstractGame, M <: NamedTuple}
  data :: DataSet{G}
  meta :: Vector{M}

  capacity :: Int

  criterion :: Function # meta :: M -> quality :: Float64
end

function Pool( :: Type{G}
             , meta :: NamedTuple
             , criterion :: Function
             ; capacity :: Int = 100000
             , targets = Target.defaults(G)
             ) where {G <: AbstractGame}

  @assert all(x -> x isa DataType, meta)
  M = NamedTuple{keys(meta), Tuple{values(meta)...}}
  data = DataSet(G, targets)
  Pool{G, M}(data, M[], capacity, criterion)
end

Pool(m :: AbstractModel{G}, args...; capacity) where {G} =
  Pool(G, args...; targets = Model.targets(m), capacity)

Base.length(dp :: Pool) = length(dp.data)

function Base.getindex(dp :: Pool{G, M}, I) where {G <: AbstractGame, M}
  Pool{G, M}(dp.data[I], dp.meta[I], dp.capacity, dp.criterion)
end

"""
    Base.append!(pool, pool2)
    Base.append!(pool, dataset, meta)

Add `dataset` with metadata `meta` to `pool`. This action does not respect
the pool's capacity. To trim the pool, see `Jtac.Data.trim!`.
"""
function Base.append!( dp :: Pool{G, M}
                     , dp2 :: Pool{G, M}
                     ) where {G <: AbstractGame, M <: NamedTuple}

  append!(dp.data, dp2.data)
  append!(dp.meta, dp2.meta)

  nothing
end

function Base.append!( dp :: Pool{G, M}
                     , ds :: DataSet{G}
                     , meta :: Vector{M}
                     ) where {G <: AbstractGame, M <: NamedTuple}

  @assert length(ds) == length(meta)

  append!(dp.data, ds)
  append!(dp.meta, meta)

  nothing
end

function Base.append!( dp :: Pool{G, M}
                     , ds :: DataSet{G}
                     , meta :: M
                     ) where {G <: Game.AbstractGame, M <: NamedTuple}

  meta = repeat([meta], length(ds))
  Base.append!(dp, ds, meta)
end


"""
    criterion!(crit, pool)

Canges the quality criterion of `pool` to `crit`.
"""
criterion!(f :: Function, dp :: Pool) = (dp.criterion = f)


"""
    capacity(pool)

Returns the capacity of `pool`
"""
capacity(dp :: Pool) = dp.capacity

"""
    capacity!(pool, capacity)

Set the capacity of `pool`
"""
capacity!(dp :: Pool, capacity) = (dp.capacity = capacity)

"""
    occupation(pool)

Returns the fraction of occupied states of `pool`
"""
occupation(dp :: Pool) = length(dp) / dp.capacity


"""
    quality(pool)
    quality(pool, sel)

Calculate the quality vector of `pool` based on the pool's `criterion` function.
If the argument `sel` is provided, only the qualities in the respective
selection are calculated.
"""
quality(dp :: Pool) = dp.criterion.(dp.meta)
quality(dp :: Pool, sel) = dp.criterion(dp.meta[sel])

"""
    sample(pool, n) -> (sample, selection)

Quality-weighted sampling of `n` elements from `pool` without replacement.
"""
function sample(dp :: Pool, n)
  # The algorithm is based on the exponential sort trick documented here:
  #   https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/
  @assert n < length(dp.data)
  w = quality(dp)
  k = length(dp.data)
  r = rand(length(dp.data))
  sel = partialsortperm(-log.(r) ./ w, 1:n)
  (dp.data[sel], sel)
end

"""
    update!(f, pool [, sel])

Update `pool` via the function `f`. This function must map
meta-information to new meta-information.
"""
function update!(f, dp :: Pool)
  dp.meta .= f.(dp.meta)
  nothing
end

function update!(f, dp :: Pool, sel)
  dp.meta[sel] .= f.(dp.meta[sel])
  nothing
end

"""
    trim!(pool; criterion)

Remove data entries from `pool`. Entries with `criterion <= 0` are always
removed. If the capacity of the pool is still exceeded, entries with `criterion
> 0` are removed as well.
"""
function trim!(dp :: Pool)
  q = dp.criterion.(dp.meta)
  indices = findall(x -> x > 0, q)
  total = length(dp)
#  flagged = length(indices)


  while length(indices) > dp.capacity
    # Get the lowest quality value in the pool that is not yet flagged for
    # removal
    minq = findmin(q[indices])[1]

    # Find all indices with better quality
    indices = findall(x -> x > minq, q)

    # If these indices do not exhaust the capacity of the pool, we add a random
    # selection of data with quality = minq
    if length(indices) < dp.capacity
      l = dp.capacity - length(indices)
      sel = findall(isequal(minq), q)
      Random.shuffle!(sel)
      append!(indices, sel[1:l])
    end
  end

  dp.data = dp.data[indices]
  dp.meta = dp.meta[indices]

  total - length(indices)
end

