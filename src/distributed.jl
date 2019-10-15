
# -------- Serializing (Async) Models ---------------------------------------- #

function _serialize(p :: Union{IntuitionPlayer, MCTSPlayer})

  # Convert the playing model to cpu
  m = playing_model(p)
  m = isa(m, Async) ? worker_model_to_cpu(m) : to_cpu(m)

  # Temporarily replace the model by DummyModel
  p = switch_model(p, DummyModel()) 

  (p, decompose(m))

end

function _deserialize(p :: Union{IntuitionPlayer, MCTSPlayer}, dm, gpu)
  
  # Reconstruct the model and bring it to the GPU if wanted
  m = compose(dm)
  gpu && (m = isa(m, Async) ? worker_model_to_gpu(m) : to_gpu(m))

  # Replace DummyModel by the reconstructed model
  switch_model(p, m)

end

# -------- Distributed Recording --------------------------------------------- #

function record_self_distributed( p :: Player{G}
                                , n :: Int = 1
                                ; game :: T = G()
                                , workers = Distributed.workers()
                                , kwargs...
                                ) :: DataSet{T} where {G, T <: G}

  # Number of available workers
  m = length(workers)

  # Deconstruct the player such that it is cpu-based and contains no tasks 
  sp = _serialize(p)

  # Get the player's training model
  tm = training_model(p)

  # Check if it lives on the GPU
  if !isnothing(tm) && on_gpu(tm)

    # See how many graphics cards there are
    devices = Knet.cudaGetDeviceCount()

    m > devices && @info "Multiple workers will share one GPU device" maxlog = 1

    # Distribute the tasks to all workers
    ds = asyncmap(1:m, ntasks = m) do i

      # TODO: Investigate why pmap would stall after 1 call
      @spawnat workers[i] begin

        # Set the gpu device for the worker
        device = (i-1) % devices
        Knet.gpu(device)

        # Number of games to be played by this worker
        n = ceil(Int, n / m)

        # Reconstruct the player and bring it on the selected GPU
        gpup = _deserialize(sp..., true)

        # Record the games
        record_self(gpup, n; game = game, kwargs...)

      end

    end

    ds = asyncmap(fetch, ds, ntasks = m)


  # If on the CPU, act differently
  else

    ds = asyncmap(1:m, ntasks = m) do i

      @spawnat workers[i] begin

        # Get the number of corres on the machine
        cores = length(Sys.cpu_info())
  
        # Set number of BLAS threads 
        threads = max(floor(Int, m/cores), 1)
        LinearAlgebra.BLAS.set_num_threads(threads)
  
        # Number of games to be played by this worker
        n = ceil(Int, n / m)
  
        # Reconstruct the player and leave it on the CPU
        cpup = _deserialize(sp..., false)
  
        # Record the games
        record_self(cpup, n; game = game, kwargs...)

      end

    end
   
    ds = asyncmap(fetch, ds, ntasks = m)

  end

  merge(ds...)

end

# -------- Distributed / Async stable Progress Maps -------------------------- #

struct ProgressBar

  length :: Int
  progress :: ProgressMeter.Progress

  channel :: RemoteChannel
  thread

end

function ProgressBar(n :: Int, description)

  glyphs = ProgressMeter.BarGlyphs("[=>⋅]")
  progress = ProgressMeter.Progress( n + 1
                                   , dt = dt
                                   , desc = description
                                   , barglyphs = glpyhs )

  # Remote channel that can be used to signal a step on any process
  channel = RemoteChannel(() -> Channel{Bool}(10))

  # Thread that manages progressing the progress bar
  thread = @async begin

    i = 1

    while i <= n && take!(channel)
      next!(progress)
      i += 1
    end

    ProgressMeter.printover(progress.output, "")

  end

  ProgressBar(n, progress, channel, thread)

end

stepper(pb :: ProgressBar) = () -> put!(pb.channel, true)
wait(pb :: ProgressBar) = @sync pb.thread

