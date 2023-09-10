
# -------- Async Wrapper ----------------------------------------------------- #

"""
Asynchronous model wrapper that allows a model to be called on a batch of games
in parallel when the single calls take place in an async context. Note that an
`Async` model always returns CPU arrays, even if the worker model acts on the GPU.
"""
mutable struct Async{G <: AbstractGame} <: AbstractModel{G, false}
  model      :: NeuralModel{G}
  ch         :: Channel
  task       :: Task

  batchsize  :: Int
  buffersize :: Int     # Must not be smaller than batchsize!

  spawn      :: Bool
  dynamic    :: Bool

  ch_dynamic :: Channel{Bool}
  profile
end

Pack.@onlyfields Async [:model, :batchsize, :buffersize, :spawn, :dynamic]

Async{G}(models, batchsize, buffersize, spawn, dynamic) where {G} =
  Async(models; batchsize, buffersize, spawn, dynamic)

Pack.freeze(m :: Async) = switch_model(m, Pack.freeze(m.model))

"""
    Async(model; kwargs...)

Wraps `model` to become an asynchronous model with maximal batchsize
`batchsize` for evaluation in parallel and a buffer of size `buffersize` for
queuing.

## Keyword Arguments
* `batchsize = 50`: Maximal number of games evaluated in parallel.
* `buffersize = 10batchsize`: Size of the channel that buffers games to be \
evaluated.
* `spawn = false`: If true, the worker is spawned in its own thread.
* `dynamic = true`: If true, the model does not have to wait until `batchsize` \
games are buffered before evaluation. This can prevent blocking (if games stop \
being evaluated) but may lead to situations where the effective batchsize is \
systematically smaller than `batchsize`, especially if `spawn = true`.
"""
function Async( model :: NeuralModel{G}
              ; batchsize = 50
              , buffersize = 10batchsize
              , spawn = false
              , dynamic = true
              ) where {G <: AbstractGame}

  # Make sure that the buffer is larger than the maximal allowed batchsize
  @assert buffersize >= batchsize "buffersize must be larger than batchsize"

  # Open the channel in which the inputs are dumped
  ch = Channel(buffersize)
  ch_dynamic = Channel{Bool}(1)
  put!(ch_dynamic, dynamic)

  # For debugging and profiling, record some information
  profile = (batchsize = Int[], delay = Float64[], latency = Float64[])

  # Start the worker task in the background
  if spawn
    task = Threads.@spawn worker(ch, model, batchsize, ch_dynamic, profile)
  else
    task = @async worker(ch, model, batchsize, ch_dynamic, profile)
  end

  # Create the instance
  amodel = Async{G}( model, ch, task, batchsize, buffersize
                   , spawn, dynamic, ch_dynamic, profile )

  # Register finalizer
  finalizer(m -> close(m.ch), amodel)

  # Return the instance
  amodel
end


function apply(m :: Async{G}, game :: G) where {G <: AbstractGame}
  c = Threads.Condition()
  put!(m.ch, (copy(game), c))
  lock(c) do
    wait(c) # notify(ticket) in the worker will provide the value
  end
end

dynamic_mode!(m :: AbstractModel, :: Bool) = nothing

function dynamic_mode!(m :: Async{G}, dynamic :: Bool) where {G}
  take!(m.ch_dynamic)
  put!(m.ch_dynamic, dynamic)
  m.dynamic = dynamic
  try Model.apply(m, G()) catch _ end # this prevents blocking
  nothing
end

#Base.close(m :: Async) = close(m.ch)

function switch_model(m :: Async{G}, model :: NeuralModel{G}) where {G <: AbstractGame}
  Async( model
       , batchsize = m.batchsize
       , buffersize = m.buffersize )
end

swap(m :: Async) = @warn "Async cannot be swapped."
Base.copy(m :: Async) = switch_model(m, copy(m.model))

ntasks(m :: Async) = m.buffersize
base_model(m :: Async) = base_model(m.model)
training_model(m :: Async) = training_model(m.model)

is_async(m :: Async) = true

function tune( m :: Async
             ; gpu = on_gpu(base_model(m))
             , async = m.batchsize
             , cache = false )

  tune(m.model; gpu, async, cache)
end

function Base.show(io :: IO, m :: Async{G}) where {G <: AbstractGame}
  print(io, "Async($(m.batchsize), $(m.buffersize), ")
  show(io, m.model)
  print(io, ")")
end

function Base.show(io :: IO, mime :: MIME"text/plain", m :: Async{G}) where {G <: AbstractGame}
  print(io, "Async($(m.batchsize), $(m.buffersize)) ")
  show(io, mime, m.model)
end

# -------- Async Worker ------------------------------------------------------ #

function closed_and_empty(ch, profile)
  try
    push!(profile.delay, @elapsed fetch(ch))
    false
  catch _
    true
  end
end

function notify_error(conds, msg)
  for c in conds
    lock(c) do
      notify(c, InvalidStateException(msg), error = true)
    end
  end
end

function worker(ch, model, batchsize, ch_dynamic, profile)

  @debug "Async worker started"

  buf = Game.array_buffer(model, batchsize)
  G = Model.gametype(model)

  # worker task has to run under the same CUDA device that the model has been
  # created in
  adapt_gpu_device!(model)

  expect_more_games = games -> begin
    if fetch(ch_dynamic)
      isready(ch) && length(games) < batchsize
    else
      length(games) < batchsize
    end
  end

  try
    while !closed_and_empty(ch, profile)

      games = Vector{G}()
      conds = Vector{Threads.Condition}()

      dt = @elapsed begin

        while expect_more_games(games)
          game, c = try take!(ch) catch _ break end
          push!(games, game)
          push!(conds, c)
          yield()
        end

        # Actual batchsize
        batchsize = length(games)
        push!(profile.batchsize, batchsize)

        v, p = try
          Game.array!(buf, games)
          v, p = model(buf[:, :, :, 1:batchsize])
          to_cpu(v), to_cpu(p)
        catch err
          notify_error(conds, "Async worker error: $err")
          close(ch)
          rethrow()
        end

        for i in 1:length(conds)
          c = conds[i]
          lock(c) do
            notify(c, (value = v[i], policy = p[:,i]))
          end
        end
      end

      push!(profile.latency, dt)
    end

  catch err
    close(ch)
    throw(err)
  end

  @debug "Async worker shutted down"
end

