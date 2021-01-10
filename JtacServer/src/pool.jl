
"""
Pool of training data.
"""
mutable struct DataPool{G <: Jtac.Game}
  data :: Jtac.DataSet{G}
  age  :: Vector{Int}
  use  :: Vector{Int}
  reqid :: Vector{Int}
  capacity :: Int
end

function DataPool{G}(c :: Int, features = Jtac.Feature[]) where {G <: Jtac.Game}
  data = Jtac.DataSet{G}(; features = features)
  DataPool{G}(data, Int[], Int[], Int[], c)
end

Base.length(dp :: DataPool) = length(dp.age)

function Base.getindex(dp :: DataPool{G}, I) where {G <: Jtac.Game}
  DataPool{G}(dp.data[I], dp.age[I], dp.use[I], dp.reqid[I], dp.capacity)
end

"""
    add!(datapool, dataset, id, latest_id)

Add `dataset` generated under the data request `id` to `datapool`. The
`latest_id` of requests has to be provided in order to calculate the age of the
dataset.
"""
function add!( dp :: DataPool{G}
             , ds :: Jtac.DataSet{G}
             , reqid :: Int
             , latest :: Int
             ) where {G <: Jtac.Game}

  k = length(ds)
  age = latest - reqid

  append!(dp.data, ds)
  append!(dp.use, fill(0, k))
  append!(dp.age, fill(age, k))
  append!(dp.reqid, fill(reqid, k))

  nothing
end

"""
    quality(datapool, context)
    quality(datapool, context, sel)

Calculate the quality vector of `datapool` in a given training `context`.
If the argument `sel` is provided, only the quality in the respective selection
is calculated.
"""
function quality(dp :: DataPool, ctx :: Context)
  age = max.(0, ctx.max_age .- dp.age) ./ ctx.max_age
  use = max.(0, ctx.max_use .- dp.use) ./ ctx.max_use
  (age .* use).^2
end

# TODO: inefficient selection?
quality(dp :: DataPool, ctx :: Context, sel) = quality(dp[sel], ctx)

"""
    random_selection(datapool, context, n)

Randomly select (on average) `n` entries from `datapool`, weighted by their
quality under `context`.
"""
function random_selection(dp :: DataPool, ctx :: Context, n)
  q = quality(dp, ctx)
  k = length(dp)
  r = n / sum(q)
  
  indices = findall(rand(k) .<= r .* q)
  pool = dp[indices]
  pool.data, indices, Statistics.mean(quality(pool, ctx))
end

"""
    dataload(datapool)

Calculate the fraction of occupied states in `datapool`.
"""
dataload(dp :: DataPool) = length(dp) / ctx.capacity


"""
    update_age!(datapool, latest_id)

Update the age entries of a `datapool` for the `latest_id` of requests.
"""
update_age!(dp :: DataPool, latest) = (dp.age .= latest .- dp.reqid; nothing)

"""
    update_use!(datapool, sel)

Increment the use counter for the selection `sel` in `datapool`.
"""
update_use!(dp :: DataPool, indices :: Vector{Int}) = (dp.use[indices] .+= 1; nothing)

"""
    update_capacity!(datapool, capacity)

Adjust the capacity of `datapool` to `capacity`.
"""
update_capacity!(dp :: DataPool, c :: Int) = (dp.capacity = c; nothing)


# TODO: may also be super inefficient due to indexing of datapool
"""
    cleanse!(datapool, context)

Clear entries with quality = 0 in a given `context` from `datapool`. If the
capacity of the pool is exceeded, entries with quality > 0 may also be removed.
"""
function cleanse!(dp :: DataPool, ctx :: Context)

  # Select all indices with positive quality
  q = quality(dp, ctx)
  indices = findall(x -> x > 0, q)

  l = length(indices)

  # In this loop, we iteratively remove entries from 'indices' if the capacity of the
  # dataset is exceeded
  while length(indices) > dp.capacity
    
    # Get the minimal quality value in the pool that is not yet flagged for
    # removal
    mq = findmin(q[indices])[1]

    # Find all indices with better quality
    indices = findall(x -> x > mq, q)

    # If these indices do not exhaust the capacity of the pool, we add a random
    # selection of data with quality = mq
    if length(indices) < dp.capacity
      l = dp.capacity - length(indices)
      idx = findall(isequal(mq), q)
      Random.shuffle!(idx)
      append!(indices, idx[1:l])
    end

  end

  dp.data  = dp.data[indices]
  dp.age   = dp.age[indices]
  dp.use   = dp.use[indices]
  dp.reqid = dp.reqid[indices]

  length(indices), l - length(indices)
end

