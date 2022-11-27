
mutable struct Batches{G <: AbstractGame, GPU}

  cache :: Cache{G}

  last_cache :: Union{Nothing, Cache{G}}

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
                , store_on_gpu = gpu
                ) where {G <: AbstractGame}

  indices = collect(1:length(d))
  cache = Cache(d, gpu = store_on_gpu)
  Batches{G, gpu}(cache, nothing, batchsize, shuffle, partial, indices)
end

function Base.length(b :: Batches)
  n = length(b.cache) / b.batchsize
  b.partial ? ceil(Int, n) : floor(Int, n)
end

function Base.iterate(b :: Batches{G, GPU}, start = 1) where {G <: AbstractGame, GPU}

  # Preparations
  l = length(b.cache)
  b.shuffle && start == 1 && (b.indices .= randperm(l))

  # start:stop is the range in b.indices that selected
  stop = min(start + b.batchsize - 1, l)

  # Check for end of iteration
  if start > l || !b.partial && stop - start < b.batchsize - 1
    Model.release_gpu_memory!(b.last_cache)
    return nothing
  end

  # Build the data cache
  idx = b.indices[start:stop]

  data = b.cache.data[:, :, :, idx]
  labels = map(b.cache.labels) do label
    label[:, idx]
  end

  Model.release_gpu_memory!(b.last_cache)

  cache = Cache{G}(data, labels, gpu = GPU)

  b.last_cache = cache

  # Return the (cache, new_start) state tuple
  cache, stop + 1
end
