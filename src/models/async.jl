
# -------- Async Wrapper ----------------------------------------------------- #

"""
Asynchronous model wrapper that allows a model to be called on a batch of games
in parallel when the single calls take place in an async context. Note that
an Async model always returns CPU arrays, even if the model it is based on works
on the GPU.
"""
mutable struct Async{G} <: Model{G, false}
  model          :: Model{G}
  channel        :: Channel
  thread

  max_batchsize  :: Int
  buffersize     :: Int     # Must not be smaller than max_batchsize!
end

"""
    Async(model; max_batchsize, buffersize)

Wraps `model` to an asynchronous model with maximal batchsize `max_batchsize`
for evaluation in parallel and a buffer of size `buffersize` for queuing.
"""
function Async( model :: Model{G}; 
                max_batchsize = 50, 
                buffersize = 2*max_batchsize ) where {G <: Game}

  # Make sure that the buffer is larger than the maximal allowed batchsize
  @assert buffersize >= max_batchsize "buffersize must be larger than max_batchsize"

  # Open the channel in which the inputs are dumped
  channel = Channel(buffersize)

  # Start the worker thread in the background
  thread = @async worker_thread(channel, model, max_batchsize)

  # Create the instance
  amodel = Async{G}(model, channel, thread, max_batchsize, buffersize)

  # Register finalizer
  finalizer(m -> close(m.channel), amodel)

  # Return the instance
  amodel

end


function (m :: Async{G})( game :: G
                        , use_features = false
                        ) where {G <: Game}

  @assert !use_features "Features should never be used in Async."

  out_channel = Channel(1)
  put!(m.channel, (game, out_channel))
  take!(out_channel)

end

function (m :: Async{G})( games :: Vector{G}
                        , use_features = false
                        ) where {G <: Game}

  @assert !use_features "Features should never be used in Async. "
  @warn "Calling Async model in batched mode is not recommended." maxlog=1

  outputs = asyncmap(x -> m(x, args...), games, ntasks = m.buffersize)
  cat_outputs(outputs)

end

function switch_model(m :: Async{G}, model :: Model{G}) where {G <: Game}
  Async( model
       , max_batchsize = m.max_batchsize
       , buffersize = m.buffersize )
end

swap(m :: Async) = @warn "Async cannot be swapped. Remaining on CPU."
Base.copy(m :: Async) = switch_model(m, copy(m.model))

ntasks(m :: Async) = m.buffersize
base_model(m :: Async) = m.model
training_model(m :: Async) = m.model
worker_model(m :: Async) = m.model

# Async networks cannot calculate features. To get the features of the
# network on which it is based, access them via training_model(...)
features(m :: Async) = Feature[]

worker_model_to_cpu(m :: Async) = switch_model(m, to_cpu(m.model))
worker_model_to_gpu(m :: Async) = switch_model(m, to_gpu(m.model))


# -------- Async Worker ------------------------------------------------------ #

closed_and_empty(channel) = try fetch(channel); false catch _ true end

function worker_thread(channel, model, max_batchsize)

  try

    while !closed_and_empty(channel)

      inputs = Vector()
      # If we arrive here, there is at least one thing to be done.
      while isready(channel) && length(inputs) < max_batchsize
        push!(inputs, take!(channel))
        yield()
      end

      v, p, f = model(first.(inputs))
      v = v |> to_cpu
      p = p |> to_cpu
      f = f |> to_cpu

      for i = 1:length(inputs)
        put!(inputs[i][2], (v[i], p[:,i], f[:,i]))
      end

    end

  catch err

    error("Worker died: $err")

  end

end

