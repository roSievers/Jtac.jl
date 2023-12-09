
# """
# Asynchronous model wrapper that allows for a model to be called on a batch of
# games in parallel when the single calls take place in an async context.
# """
mutable struct AsyncModel{G <: AbstractGame} <: AbstractModel{G}
  model      :: NeuralModel{G}
  ch         :: Channel
  task       :: Task

  batchsize  :: Int
  buffersize :: Int     # Must not be smaller than batchsize!

  spawn      :: Bool
  dynamic    :: Bool

  cdynamic :: Channel{Bool}
  nqueue :: Threads.Atomic{Int}
  profile
end

@pack {<: AsyncModel} AsyncModel(model; batchsize, buffersize, spawn, dynamic)

"""
Asynchronous model wrapper that allows for a model to be called on a batch of
games in parallel when the single calls take place in an async context.

---

    AsyncModel(model; kwargs...)

Wrap `model` into an `AsyncModel` model.

## Arguments
- `batchsize = 64`: Maximal number of games evaluated in parallel.
- `buffersize = 10batchsize`: Size of the channel that buffers games to be \
evaluated.
- `spawn = false`: If true, the worker is spawned in a thread.
- `dynamic = true`: If true, the model is not forced to wait until `batchsize` \
games are buffered before evaluation. This can prevent blocking (if games stop \
being evaluated) but may lead to situations where the effective batchsize is \
systematically smaller than `batchsize`, especially if `spawn = true`.
"""
function AsyncModel( model :: NeuralModel{G}
                   ; batchsize = 64
                   , buffersize = 10batchsize
                   , spawn = false
                   , dynamic = true
                   ) where {G <: AbstractGame}

  @assert buffersize >= batchsize "buffersize must be larger than batchsize"

  ch = Channel(buffersize)
  cdynamic = Channel{Bool}(1)
  put!(cdynamic, dynamic)

  # For debugging and profiling, record some information
  profile = (batchsize = Int[], delay = Float64[], latency = Float64[])

  # Start the worker task in the background
  if spawn
    task = Threads.@spawn worker(model, ch, batchsize, cdynamic, profile)
  else
    task = @async worker(model, ch, batchsize, cdynamic, profile)
  end

  nqueue = Threads.Atomic{Int}(0)

  # Create the instance
  amodel = AsyncModel{G}( model, ch, task, batchsize, buffersize
                        , spawn, dynamic, cdynamic, nqueue, profile )

  # Register finalizer
  finalizer(m -> close(m.ch), amodel)

  # Return the instance
  amodel
end


function apply( m :: AsyncModel{G}
              , game :: G
              ; targets = [:value, :policy]
              ) where {G <: AbstractGame}

  @assert issubset(targets, targetnames(m))
  @assert !istaskdone(m.task) "Worker task of async model has stopped"
  c = Threads.Condition()
  # Threads.atomic_add!(m.nqueue, 1)
  put!(m.ch, (copy(game), c))
  val = lock(() -> wait(c), c) # notify(c) in the worker will provide the value
  # Threads.atomic_sub!(m.nqueue, 1)
  val
end

nqueue(m :: AsyncModel) = m.nqueue[]

dynamicmode!(:: AbstractModel, :: Bool) = nothing

function dynamicmode!(m :: AsyncModel{G}, dynamic :: Bool) where {G}
  take!(m.cdynamic)
  put!(m.cdynamic, dynamic)
  m.dynamic = dynamic
  try apply(m, G()) catch _ end # prevent blocking
  nothing
end

function switchmodel( m :: AsyncModel{G}
                    , model :: NeuralModel{G}
                    ) where {G <: AbstractGame}
  AsyncModel(model; m.batchsize, m.buffersize, m.spawn, m.dynamic)
end

adapt(backend, m :: AsyncModel) = switchmodel(m, adapt(backend, m.model))
getbackend(m :: AsyncModel) = getbackend(m.model)

isasync(m :: AsyncModel) = true
ntasks(m :: AsyncModel) = 2 * m.batchsize
basemodel(m :: AsyncModel) = basemodel(m.model)
childmodel(m :: AsyncModel) = m.model
trainingmodel(m :: AsyncModel) = trainingmodel(m.model)

Base.copy(m :: AsyncModel) = switchmodel(m, copy(m.model))

function Base.show(io :: IO, m :: AsyncModel{G}) where {G <: AbstractGame}
  print(io, "Async($(m.batchsize), $(m.buffersize), ")
  show(io, m.model)
  print(io, ")")
end

function Base.show( io :: IO
                  , mime :: MIME"text/plain"
                  , m :: AsyncModel{G}
                  ) where {G <: AbstractGame}
  print(io, "Async($(m.batchsize), $(m.dynamic ? "dynamic" : "static")) ")
  show(io, mime, m.model)
end

# -------- Async Worker ------------------------------------------------------ #

function closedandempty(ch, profile)
  try
    push!(profile.delay, @elapsed fetch(ch))
    false
  catch _
    true
  end
end

function notifyerror(conds, msg)
  for c in conds
    lock(c) do
      notify(c, InvalidStateException(msg), error = true)
    end
  end
end

function worker( model :: NeuralModel{G, B}
               , ch
               , max_batchsize
               , cdynamic
               , profile ) where {G, B}

  @debug "Worker of AsyncModel started"

  # Worker task should run on the same device that the model has been
  # created in
  aligndevice!(model)

  # Prepare buffer to make game array creation more efficient
  buf = Game.arraybuffer(G, max_batchsize)
  buf = convert(arraytype(B()), buf)


  expectmore = games -> begin
    if fetch(cdynamic)
      isready(ch) && length(games) < max_batchsize
    else
      length(games) < max_batchsize
    end
  end

  try
    while !closedandempty(ch, profile)

      games = Vector{G}()
      conds = Vector{Threads.Condition}()

      dt = @elapsed begin

        while expectmore(games)
          game, c = try take!(ch) catch _ break end
          push!(games, game)
          push!(conds, c)
          yield()
        end

        # Actual size of the batch to evaluate
        batchsize = length(games)
        push!(profile.batchsize, batchsize)

        # TODO: Benchmark buf[:,:,:,1:batchsize] ! Could be worse than resizing
        # the buffer such that no temp copies are created. Alternatively, work
        # with views?
        v, p = try
          Game.array!(buf, games)
          vp = model(buf[:, :, :, 1:batchsize])
          vp = convert.(Array{Float32}, vp)
          vp
        catch err
          notifyerror(conds, "Worker of AsyncModel erred: $err")
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

  @debug "Worker of AsyncModel exited"
end

