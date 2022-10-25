
struct Batches{G <: AbstractGame, GPU}

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
                ) where {G <: AbstractGame}

  indices = collect(1:length(d))
  cache = DataCache(d, gpu = gpu, use_features = use_features)
  Batches{G, gpu}(cache, batchsize, shuffle, partial, indices)
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
