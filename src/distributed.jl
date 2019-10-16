
# -------- Serializing (Async) Models ---------------------------------------- #

function _serialize(p :: Union{IntuitionPlayer, MCTSPlayer})

  # Convert the playing model to cpu
  m = playing_model(p)
  m = isa(m, Async) ? worker_model_to_cpu(m) : to_cpu(m)

  # Temporarily replace the model by DummyModel
  p = switch_model(p, DummyModel()) 

  (p, decompose(m), on_gpu(training_model(p)))

end

function _deserialize(p :: Union{IntuitionPlayer, MCTSPlayer}, dm, gpu)
  
  # Reconstruct the model and bring it to the GPU if wanted
  m = compose(dm)
  gpu && (m = isa(m, Async) ? worker_model_to_gpu(m) : to_gpu(m))

  # Replace DummyModel by the reconstructed model
  switch_model(p, m)

end

_on_gpu(splayers) = any(p[3] for p in splayers)

# -------- Distributed Calculations ------------------------------------------ #


function set_defaults(i, worker, gpu)

  if gpu

    devices = Knet.cudaGetDeviceCount()
    m > devices && @info "Multiple workers will share one GPU device" maxlog = 1

    # Set the gpu device for the worker
    Knet.gpu((i-1) % devices)

  else

    # TODO: Setting the number of BLAS threads here kills the process

    # Get the number of cores on the machine
    # cores = Sys.CPU_THREADS

    # Set number of BLAS threads 
    # threads = max(floor(Int, m/cores), 1)
    # LinearAlgebra.BLAS.set_num_threads(threads)

  end

end

function ticket_sizes(n, m)
  ns = zeros(Int, m)
  ns .+= floor(Int, n / m)
  ns[1:(n % m)] .+= 1
  ns
end

take_ticket!(tic) = try take!(tic) catch _ 0 end

# TODO: We currently assume that we stay on one machine. Should not be too hard
# to generalize this to work on quite arbitrary cluster-systems.
# For this, we should provide the 'with_workers' function with more information,
# e.g., which workers will land on nodes with how many GPUs.
function with_workers( f :: Function
                     , players
                     , n
                     , args...
                     ; workers = workers()
                     , tickets = length(workers)
                     , kwargs... )

  # Count the number of workers
  m = length(workers)

  # Generate channels to provide worker tickets and receive results
  tic = RemoteChannel(() -> Channel(tickets))
  res = RemoteChannel(() -> Channel(m))

  # Divide the n tasks as evenly as possible and fill the ticket supply
  foreach(s -> put!(tic, s), ticket_sizes(n, tickets))

  # Fill the buffer asynchronously
  task = @async map(_ -> take!(res), 1:tickets)

  # Serialize the players
  splayers = _serialize.(players)

  # Start let the workers work on the tickets
  map(1:m) do i

    @spawnat workers[i] begin

      # Make adjustments to the environment, like number of threads or GPU id
      set_defaults(i, workers[i], _on_gpu(splayers))

      # Reconstruct the players
      players = [_deserialize(p...) for p in splayers]

      # Wait for a ticket. If we get one, carry it out. If there are no tickets
      # left, the ticket channel is closed and we end the loop.
      while (n = take_ticket!(tic)) != 0

        put!(res, f(players, n, args...; kwargs...))

      end

    end

  end

  # Fetch the data with error handling if something unexpected happens
  data = try

    fetch(task)

  catch exn

    close(res); close(tic)
    interrupt(workers)
    rethrow(exn)

  end

  # Close the channels
  close(res); close(tic)

  # Return the data
  vcat(data...)

end


# -------- Distributed Recording --------------------------------------------- #

function record_self_distributed( p :: Player
                                , n :: Int = 1
                                ; workers = workers()
                                , tickets = length(workers)
                                , merge = false
                                , kwargs... )

  # Create the record function
  record = (ps, n; kwargs...) -> begin
    record_self(ps[1], n; merge = merge, kwargs...)
  end

  ds = with_workers(record, [p], n; workers = workers, kwargs...)

  merge ? Base.merge(ds) : ds

end


function record_against_distributed( p :: Player
                                   , enemy :: Player
                                   , n :: Int = 1
                                   ; workers = workers()
                                   , merge = false
                                   , kwargs... )

  # Create the record function
  record = (ps, n; kwargs...) -> begin
    record_against(ps[1], ps[2], n; merge = merge, kwargs...)
  end

  ds = with_workers(record, [p, enemy], n; workers = workers, kwargs...)

  merge ? Base.merge(ds) : ds

end

# -------- Distributed Contesting -------------------------------------------- #
