
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
                        , activate_features = false
                        ) where {G <: Game}

  @assert !activate_features "Features should never be activated in Async."

  out_channel = Channel(0)
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

swap(m :: Async) = @warn "Async models cannot be swapped to GPU mode."

function Base.copy(m :: Async)

  Async( copy(m.model)
       , max_batchsize = m.max_batchsize
       , buffersize = m.buffersize )

end

ntasks(m :: Async) = m.buffersize
training_model(m :: Async) = m.model

# Async networks cannot calculate features. To get the features of the
# network on which it is based, access them via training_model(...)
features(m :: Async) = []


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


#      if length(inputs) == 1
#        put!(inputs[1][2], model(inputs[1][1]))
#      else
#        outputs = model(first.(inputs))
#        for i = 1:length(inputs)
#          put!(inputs[i][2], outputs[:,i])
#        end
#      end

    end

  catch err

    println("Worker died: $err")

  end

end

