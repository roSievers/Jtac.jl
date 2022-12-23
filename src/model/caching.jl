

"""
Model wrapper that enables game state caching. This wrapper is intended for
accelerating MCTS playouts, since previously evaluated game states might be
queried repeatedly.
"""
mutable struct Caching{G <: AbstractGame} <: AbstractModel{G, false}
  model :: AbstractModel{G}
  max_cachesize :: Int

  cache :: Dict{UInt64, Tuple{Float32, Vector{Float32}}}

  calls_cached :: Int
  calls_uncached :: Int
end

Pack.@onlyfields Caching [:model, :max_cachesize]

Caching{G}(model, max_cachesize) where {G} =
  Caching(model; max_cachesize) :: Caching{G}

Pack.freeze(m :: Caching) = switch_model(m, Pack.freeze(m.model))

"""
    Caching(model; max_cachesize)

Wraps `model` in a caching layer that checks if a game to be evaluated has
already been cached. If so, the cached result is reused. If it has not been
cached before, and if fewer game states that `max_cachesize` are currently
stored, it is added to the cache.
"""
function Caching(model :: AbstractModel; max_cachesize = 100000)

  # It does not make sense to wrap Caching models
  @assert !(model isa Caching) "Cannot wrap Caching model in Caching"

  cache = Dict{UInt64, Tuple{Float32, Vector{Float32}}}()
  sizehint!(cache, max_cachesize)
  Caching(model, max_cachesize, cache, 0, 0)
end

function apply(m :: Caching{G}, game :: G) where {G <: AbstractGame}
  m.calls_cached += 1
  (v, p) = get(m.cache, Game.hash(game)) do
    m.calls_cached -= 1
    m.calls_uncached += 1
    (v, p) = apply(m.model, game)
    if length(m.cache) < m.max_cachesize
      m.cache[Game.hash(game)] = (v, p)
    end
    (v, p)
  end
  (value = v, policy = p)
end


function clear_cache!(m :: Caching{G}) where {G <: AbstractGame}
  m.cache = Dict{UInt64, Tuple{Float32, Vector{Float32}}}()
  sizehint!(m.cache, m.max_cachesize)
  m.calls_cached = 0
  m.calls_uncached = 0
  nothing
end

function switch_model(m :: Caching{G}, model :: AbstractModel{G}) where {G <: AbstractGame}
  Caching(model; max_cachesize = m.max_cachesize)
end

swap(m :: Caching) = @warn "Caching models cannot be swapped"
Base.copy(m :: Caching) = switch_model(m, copy(m.model))

ntasks(m :: Caching) = ntasks(m.model)
base_model(m :: Caching) = base_model(m.model)
training_model(m :: Caching) = training_model(m.model)

is_async(m :: Caching) = is_async(m.model)

function tune( m :: Caching
             ; gpu = on_gpu(base_model(m))
             , async = is_async(m) ? m.model.max_batchsize : false
             , cache = m.max_cachesize )

  tune(m.model; gpu, async, cache)
end

function Base.show(io :: IO, m :: Caching{G}) where {G <: AbstractGame}
  print(io, "Caching($(length(m.cache)), $(m.max_cachesize), ")
  show(io, m.model)
  print(io, ")")
end

function Base.show(io :: IO, mime :: MIME"text/plain", m :: Caching{G}) where {G <: AbstractGame}
  print(io, "Caching($(length(m.cache)), $(m.max_cachesize)) ")
  show(io, mime, m.model)
end

