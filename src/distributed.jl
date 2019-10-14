
# -------- Serializing (Async) Models ---------------------------------------- #

function _serialize(p :: Union{IntuitionPlayer, MCTSPlayer})

  m = playing_model(p)
  m = isa(m, Async) ? worker_model_to_cpu(m) : to_cpu(m)
  p = switch_model(p, training_model(p))

  (p, decompose(m))

end

function _deserialize(p :: Union{IntuitionPlayer, MCTSPlayer}, dm)
  
  m = compose(dm)
  m = isa(m, Async) ? worker_model_to_gpu(m) : to_gpu(m)

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

  # Check if the player's model has memory that lies on the GPU
  if isa(training_model(p), Model{G, true})

    # See how many graphics cards there are
    devices = Knet.cudaGetDeviceCount()

    m > devices && @info "Multiple workers will share one GPU device" maxlog = 1

    # Deconstruct the player to be cpu-based and task-free
    sp = _serialize(p)

    # Distribute the tasks to all workers
    ds = pmap(1:m) do worker

      # Set the gpu device for the worker
      device = worker % devices
      gpu(device)

      # Reconstruct the player and bring it on the selected GPU
      player = _deserialize(sp...)

      # Number of games to be played by this worker
      n = ceil(Int, n / m)

      # Record the games!
      record_self(p, n; game = game, kwargs...)

    end

  # If on the CPU, act a bit differently
  else

    ds = pmap(1:m) do worker

      # Get the number of corres on the machine
      cores = length(Sys.cpu_info())

      # Set number of BLAS threads 
      threads = max(floor(Int, m/cores), 1)
      LinearAlgebra.BLAS.set_num_threads(threads)

      # Number of games to be played by this worker
      n = ceil(Int, n / m)

      # Record the games
      record_self(p, n; game = game, kwargs...)

    end

  end

  merge(ds...)

end

