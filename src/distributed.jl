
# -------- Serializing (Async) Models ---------------------------------------- #

function _serialize(p :: Union{IntuitionPlayer, MCTSPlayer})

  m = playing_model(p)
  m = isa(m, Async) ? worker_model_to_cpu(m) : to_cpu(m)
  p = switch_model(p, training_model(p)) # the training model is serializable

  (p, decompose(m))

end

function _deserialize(p :: Union{IntuitionPlayer, MCTSPlayer}, dm, gpu)
  
  m = compose(dm)

  gpu && (m = isa(m, Async) ? worker_model_to_gpu(m) : to_gpu(m))

  switch_model(p, m)

end

# -------- Distributed Recording --------------------------------------------- #

function record_self_distributed( p :: Player{G}
                                , n :: Int = 1
                                ; game :: T = G()
                                , kwargs...
                                ) :: DataSet{T} where {G, T <: G}

  # Number of available workers
  m = nworkers()

  # Deconstruct the player such that it is cpu-based contains no tasks 
  sp = _serialize(p)

  # Check if the player's training model lives on the GPU
  if on_gpu(training_model(p))

    # See how many graphics cards there are
    devices = Knet.cudaGetDeviceCount()

    m > devices && @info "Multiple workers will share one GPU device" maxlog = 1

    # Distribute the tasks to all workers
    ds = pmap(1:m) do worker

      # Set the gpu device for the worker
      device = worker % devices
      gpu(device)

      # Number of games to be played by this worker
      n = ceil(Int, n / m)

      # Reconstruct the player and bring it on the selected GPU
      player = _deserialize(sp..., true)

      # Record the games
      record_self(player, n; game = game, kwargs...)

    end

  # If on the CPU, act differently
  else

    ds = pmap(1:m) do worker

      # Get the number of corres on the machine
      cores = length(Sys.cpu_info())

      # Set number of BLAS threads 
      threads = max(floor(Int, m/cores), 1)
      LinearAlgebra.BLAS.set_num_threads(threads)

      # Number of games to be played by this worker
      n = ceil(Int, n / m)

      # Reconstruct the player and leave it on the CPU
      player = _deserialize(sp..., false)

      # Record the games
      record_self(player, n; game = game, kwargs...)

    end

  end

  merge(ds...)

end

