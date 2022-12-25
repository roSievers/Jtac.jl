
# -------- Async Wrapper ----------------------------------------------------- #

"""
Asynchronous model wrapper that allows a model to be called on a batch of games
in parallel when the single calls take place in an async context. Note that an
Async model always returns CPU arrays, even if the worker model acts on the GPU.
"""
mutable struct Async{G <: AbstractGame} <: AbstractModel{G, false}
  model          :: AbstractModel{G}
  channel        :: Channel
  task           :: Task

  max_batchsize  :: Int
  buffersize     :: Int     # Must not be smaller than max_batchsize!

  profile

end

Pack.@onlyfields Async [:model, :max_batchsize, :buffersize]

Async{G}(models, max_batchsize, buffersize) where {G} =
  Async(models; max_batchsize, buffersize) :: Async{G}

Pack.freeze(m :: Async) = switch_model(m, Pack.freeze(m.model))

"""
    Async(model; max_batchsize, buffersize)

Wraps `model` to become an asynchronous model with maximal batchsize
`max_batchsize` for evaluation in parallel and a buffer of size `buffersize` for
queuing.
"""
function Async( model :: AbstractModel{G};
                max_batchsize = 50, 
                buffersize = 10max_batchsize ) where {G <: AbstractGame}

  # It does not make sense to wrap Async or Caching models, since these
  # are not suited for batch evaluation
  @assert !(model isa Async) "Cannot wrap Async model in Async"
  @assert !(model isa Caching) "Cannot wrap Caching model in Async"

  # Make sure that the buffer is larger than the maximal allowed batchsize
  @assert buffersize >= max_batchsize "Buffersize must be larger than max_batchsize"

  # Open the channel in which the inputs are dumped
  channel = Channel(buffersize)

  # For debugging and profiling, record some information
  batchsize = Int[]
  delay = Float64[]
  latency = Float64[]
  profile = (batchsize = batchsize, delay = delay, latency = latency)

  # Start the worker task in the background
  task = @async worker_task(channel, model, max_batchsize, profile)

  # Create the instance
  amodel = Async{G}(model, channel, task, max_batchsize, buffersize, profile)

  # Register finalizer
  finalizer(m -> close(m.channel), amodel)

  # Return the instance
  amodel

end


function apply(m :: Async{G}, game :: G) where {G <: AbstractGame}
  out_channel = Channel(1)
  put!(m.channel, (copy(game), out_channel))
  take!(out_channel)
end


function switch_model(m :: Async{G}, model :: AbstractModel{G}) where {G <: AbstractGame}
  Async( model
       , max_batchsize = m.max_batchsize
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
             , async = m.max_batchsize
             , cache = false )

  tune(m.model; gpu, async, cache)
end

function Base.show(io :: IO, m :: Async{G}) where {G <: AbstractGame}
  print(io, "Async($(m.max_batchsize), $(m.buffersize), ")
  show(io, m.model)
  print(io, ")")
end

function Base.show(io :: IO, mime :: MIME"text/plain", m :: Async{G}) where {G <: AbstractGame}
  print(io, "Async($(m.max_batchsize), $(m.buffersize)) ")
  show(io, mime, m.model)
end


# -------- Async Worker ------------------------------------------------------ #

function closed_and_empty(channel, profile)
  try
    push!(profile.delay, @elapsed fetch(channel))
    false
  catch _
    true
  end
end

function worker_task(channel, model, max_batchsize, profile)

  @debug "Async worker started"

  try

    while !closed_and_empty(channel, profile)

      dt = @elapsed begin
        inputs = Vector()
        # If we arrive here, there is at least one thing to be done.
        while isready(channel) && length(inputs) < max_batchsize
          push!(inputs, take!(channel))
          yield()
        end

        push!(profile.batchsize, length(inputs))

        try
          v, p = model(first.(inputs))
          v = to_cpu(v)
          p = to_cpu(p)
        catch
          for i in 1:length(inputs)
            # Notify all callers that something went wrong
            # Otherwise, this task fails silently and the
            # callers hang
            try close(inputs[i][2]) catch end
          end
          rethrow()
        end

        for i in 1:length(inputs)
          put!(inputs[i][2], (value = v[i], policy = p[:,i]))
        end
      end

      push!(profile.latency, dt)

    end

  catch err

    close(channel)
    throw(err)

  end

  @debug "Async worker shutted down"

end

