
"""
Data structure for a finite capacity pool of training data that can be tagged
with metadata, like an age or a usage index. Provides functionality to select
and discard training data based on user defined quality criteria.
"""
mutable struct Pool{G <: Game.AbstractGame, M <: NamedTuple}
  data :: Dataset{G}
  meta :: Vector{M}
  capacity :: Int
end

function Pool( :: Type{G}
             , capacity :: Int
             ; meta :: NamedTuple = (;)
             , features = Model.Feature[]
             ) where {N, G <: Game.AbstractGame}

  @assert all(x -> x isa DataType, meta)
  M = NamedTuple{keys(meta), Tuple{values(meta)...}}
  data = Dataset{G}(; features = features)
  Pool{G, M}(data, M[], capacity)
end

Base.length(dp :: Pool) = length(dp.data)

function Base.getindex(dp :: Pool{G, M}, I) where {G <: Game.AbstractGame, M}
  Pool{G, M}(dp.data[I], dp.meta[I], dp.capacity)
end

"""
    Base.append!(pool, pool)
    Base.append!(pool, dataset, meta)
    Base.append!(pool, dataset; meta...)

Add `dataset` with metadata `meta` to `pool`. This action does not respect
the pool's capacity. To trim the pool, `see Jtac.Data.trim!`.
"""
function Base.append!( pool :: Pool{G, M}
                     , pool2 :: Pool{G, M}
                     ) where {G <: Game.AbstractGame, M <: NamedTuple}
  append!(pool.data, pool2.data)
  append!(pool.meta, pool2.meta)

  nothing
end

function Base.append!( pool :: Pool{G, M}
                     , data :: Dataset{G}
                     , meta :: M
                     ) where {G <: Game.AbstractGame, M <: NamedTuple}

  append!(pool.data, data)
  append!(pool.meta, meta)

  nothing
end

function Base.append!( pool :: Pool{G, M}
                     , data :: Dataset{G}
                     ; kwargs...
                     ) where {G <: Game.AbstractGame, M <: NamedTuple}

  kwargs = collect(kwargs)
  getvals = map(kwargs) do (key, val)
    @assert key in fieldnames(M)
    T = fieldtype(M, key)
    if typeof(val) == T
      (key, idx -> val)
    elseif typeof(val) == Vector{T}
      (key, idx -> val[idx])
    else
      F = typeof(val) 
      msg = "Metadata with key $key has type $F (expected was $T)"
      throw(ArgumentError(msg))
    end
  end

  meta = map(1:length(data)) do idx
    (; map(x -> (x[1] => x[2](idx)), getvals)...) :: M
  end

  append!(pool.data, data)
  append!(pool.meta, meta)

  nothing
end

"""
    capacity(pool)

Returns the capacity of `pool`
"""
capacity(pool :: Pool) = pool.capacity

"""
    occupation(pool)

Returns the fraction of occupied states of `pool`
"""
occupation(pool :: Pool) = length(pool) / pool.capacity


"""
    dataquality(pool; criterion)
    dataquality(pool, sel; criterion)

Calculate the quality vector of `pool` based on a `criterion` function.
This function takes metadata and returns a quality value. If the argument `sel`
is provided, only the qualities in the respective selection are calculated.
"""
dataquality(pool :: Pool; criterion) = map(criterion, pool.meta)
dataquality(pool :: Pool, sel; criterion) = map(idx -> criterion(pool.meta[idx]), sel)

"""
    sample(pool, n; criterion)

Weighted sampling of `n` elements from `pool` without replacement. The
weights are calculated by `criterion`.
"""
function sample(pool :: Pool, n; criterion)
  # The algorithm is based on the exponential sort trick documented here:
  #   https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/
  @assert n < length(pool.data)
  w = dataquality(pool; criterion)
  k = length(pool.data)
  r = rand(length(pool.data))
  partialsortperm(-log.(r) ./ w, 1:n)
end

function update!(f, pool :: Pool)
  pool.meta .= f.(pool.meta)
  nothing
end

function update!(f, pool :: Pool, sel)
  h(idx) = f(pool.meta[idx])
  pool.meta[sel] .= h.(sel)
  nothing
end

# TODO: these update functions cause a recompilation of f every time they run...
function update!(pool :: Pool{G, M}, sel; kwargs...) where {G, M}
  mapper = map(fieldnames(M)) do key
    (key, key in keys(kwargs) ? kwargs[key] : identity)
  end
  ntmapper = (; mapper...)
  f(x) = map((g, v) -> g(v), ntmapper, x)
  update!(f, pool, sel)
  nothing
end

function update!(pool :: Pool; kwargs...)
  update!(pool, 1:length(pool.data); kwargs...)
  nothing
end


"""
    trim!(pool; criterion)

Remove data entries from `pool`. Entries with `criterion <= 0` are always
removed. If the capacity of the pool is still exceeded, entries with `criterion
> 0` are removed as well.
"""
function trim!(pool :: Pool; criterion)
  q = dataquality(pool; criterion)
  indices = findall(x -> x > 0, q)
  l = length(indices)

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
      idx = findall(isequal(minq), q)
      Random.shuffle!(idx)
      append!(indices, idx[1:l])
    end
  end

  dp.data = dp.data[indices]
  dp.meta = dp.meta[indices]

  nothing #length(indices), l - length(indices)
end
