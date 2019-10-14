
# -------- Remove or bring Players from or to GPU # -------------------------- #

_cpu_player(p :: Player) = p

function _cpu_player(p :: Union{IntuitionPlayer, MCTSPlayer})

  m = playing_model(p)
  mcpu = isa(m, Async) ? worker_model_to_cpu(m) : to_cpu(m)
  switch_model(p, mcpu)

end

# Helper that brings a player on the cpu back

function _gpu_player(p :: Union{IntuitionPlayer, MCTSPlayer})

  m = playing_model(p)
  mgpu = isa(m, Async) ? worker_model_to_gpu(m) : to_gpu(m)
  switch_model(p, mgpu)

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

    # Create a CPU-based copy of player, that can then be converted to the
    # individual GPUs
    cpu_player = _cpu_player(p)

    # Distribute the tasks to all workers
    ds = pmap(1:m) do worker

      # Set the gpu device for the worker
      device = worker % devices
      gpu(device)

      # Get the player on the GPU
      player = _gpu_player(cpu_player)

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

