

"""
Model wrapper that enables game state caching. This wrapper is intended for
accelerating MCTS playouts where previously evaluated game states are queried
repeatedly.
"""
mutable struct CachingModel{G <: AbstractGame} <: AbstractModel{G}
  model :: AbstractModel{G}
  cachesize :: Int

  cache :: Dict{UInt64, Tuple{Float32, Vector{Float32}}}

  calls_cached :: Int
  calls_uncached :: Int
end

Pack.freeze(m :: CachingModel) = switchmodel(m, Pack.freeze(m.model))
Pack.@only CachingModel [:model, :cachesize]

CachingModel{G}(model, cachesize) where {G} = CachingModel(model; cachesize)


"""
    CachingModel(model; cachesize)

Wraps `model` in a caching layer that checks if a game to be evaluated has
already been cached. If so, the cached result is reused. If it has not been
cached before, and if fewer game states than `cachesize` are currently
stored, it is added to the cache.
"""
function CachingModel(model :: AbstractModel; cachesize = 100000)
  # It does not make sense to wrap Caching models
  @assert !(model isa CachingModel) "Cannot wrap CachingModel model in CachingModel"

  cache = Dict{UInt64, Tuple{Float32, Vector{Float32}}}()
  sizehint!(cache, cachesize)
  CachingModel(model, cachesize, cache, 0, 0)
end

function apply( m :: CachingModel{G}
              , game :: G
              ; targets = [:value, :policy]
              ) where {G <: AbstractGame}
  @assert issubset(targets, targetnames(m))
  m.calls_cached += 1
  (v, p) = get(m.cache, Game.hash(game)) do
    m.calls_cached -= 1
    m.calls_uncached += 1
    (v, p) = apply(m.model, game)
    if length(m.cache) < m.cachesize
      m.cache[Game.hash(game)] = (v, p)
    end
    (v, p)
  end
  (value = v, policy = p)
end

function clear_cache!(m :: CachingModel{G}) where {G <: AbstractGame}
  m.cache = Dict{UInt64, Tuple{Float32, Vector{Float32}}}()
  sizehint!(m.cache, m.cachesize)
  m.calls_cached = 0
  m.calls_uncached = 0
  nothing
end

function switchmodel(m :: CachingModel{G}, model :: AbstractModel{G}) where {G <: AbstractGame}
  CachingModel(model; cachesize = m.cachesize)
end

adapt(backend, m :: CachingModel) = switchmodel(m, adapt(backend, m.model))

isasync(m :: CachingModel) = isasync(m.model)
ntasks(m :: CachingModel) = ntasks(m.model)
childmodel(m :: CachingModel) = m.model
basemodel(m :: CachingModel) = basemodel(m.model)
trainingmodel(m :: CachingModel) = trainingmodel(m.model)

Base.copy(m :: CachingModel) = switchmodel(m, copy(m.model))

function Base.show(io :: IO, m :: CachingModel{G}) where {G <: AbstractGame}
  print(io, "Caching($(length(m.cache)), $(m.cachesize), ")
  show(io, m.model)
  print(io, ")")
end

function Base.show(io :: IO, mime :: MIME"text/plain", m :: CachingModel{G}) where {G <: AbstractGame}
  print(io, "Caching($(length(m.cache)), $(m.cachesize)) ")
  show(io, mime, m.model)
end

