

mutable struct Async{G, GPU} <: Model{G, GPU}
  model          :: Model{G, GPU}
  channel        :: Channel
  thread

  max_batchsize  :: Int
  buffersize     :: Int     # Must not be smaller than max_batchsize!
end

function Async( model :: Model{G, GPU}; 
                max_batchsize = 50, 
                buffersize = 2*max_batchsize ) where {G <: Game, GPU}

  # Make sure that the buffer is larger than the maximal allowed batchsize
  @assert buffersize >= max_batchsize "buffersize must be larger than max_batchsize"

  # Open the channel in which the inputs are dumped
  channel = Channel(buffersize)

  # Start the worker thread in the background
  thread = @async worker_thread(channel, model, max_batchsize)

  # Create the instance
  amodel = Async{G, GPU}(model, channel, thread, max_batchsize, buffersize)

  # Register finalizer
  finalizer(m -> close(m.channel), amodel)

  # Return the instance
  amodel
end


function (m :: Async{G})(game :: G) where {G <: Game}
  out_channel = Channel(0)
  put!(m.channel, (game, out_channel))
  take!(out_channel)
end

(m :: Async{G})(games :: Vector{G}) where {G <: Game} = map(m, games)

function swap(m :: Async)
  Async( swap(m.model), max_batchsize = m.max_batchsize, buffersize = m.buffersize )
end

function Base.copy(m :: Async)
  Async( copy(m.model), max_batchsize = m.max_batchsize, buffersize = m.buffersize )
end


# Helper functions

closed_and_empty(channel) = try fetch(channel); true catch _ false end

function worker_thread(channel, model, max_batchsize)
  while !closed_and_empty(channel)
    inputs = Vector()
    # If we arrive here, there is at least one thing to be done.
    while isready(channel) && length(inputs) < max_batchsize
      push!(inputs, take!(channel))
    end
    #println("Processing $(length(inputs)) inputs at the same time.")
    if length(inputs) == 1
      put!(inputs[1][2], model(inputs[1][1]))
    else
      outputs = model(first.(inputs))
      for i = 1:length(inputs)
        put!(inputs[i][2], outputs[:,i])
      end
    end
  end
end

