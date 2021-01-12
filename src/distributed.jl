
# -------- Serializing (Async) Players --------------------------------------- #

function pack(p :: Union{IntuitionPlayer, MCTSPlayer})

  # Convert the playing model to cpu
  m = playing_model(p)
  m = isa(m, Async) ? worker_model_to_cpu(m) : to_cpu(m)

  # Temporarily replace the model by DummyModel
  pt = switch_model(p, DummyModel()) 

  (pt, decompose(m), on_gpu(training_model(p)))

end

function unpack(p :: Union{IntuitionPlayer, MCTSPlayer}, dm, gpu)
  
  # Reconstruct the model and bring it to the GPU if wanted
  m = compose(dm)
  gpu && (m = isa(m, Async) ? worker_model_to_gpu(m) : to_gpu(m))

  # Replace DummyModel by the reconstructed model
  switch_model(p, m)

end

models_on_gpu(splayers) = any(p[3] for p in splayers)

# -------- Distributed Calculations ------------------------------------------ #

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
  splayers = pack.(players)

  # Count the GPUs
  gpu = models_on_gpu(splayers)
  devices = Knet.cudaGetDeviceCount()
  m > devices && gpu && @info "Multiple workers will share one GPU device" maxlog = 1

  # Start let the workers work on the tickets
  map(1:m) do i

    @spawnat workers[i] begin

      # Make adjustments to the environment, like number of threads or GPU id
      # TODO: This does not seem to work, as Knet.gpu is not known on the process
      # if this call is hidden in set_defaults.
      # set_defaults(i, devices, _on_gpu(splayers))
      if gpu
        CUDA.device!((i-1) % devices)
      end

      # Reconstruct the players
      ps = [unpack(p...) for p in splayers]

      # Wait for a ticket. If we get one, carry it out. If there are no tickets
      # left, the ticket channel is closed and we end the loop.
      while (n = take_ticket!(tic)) != 0

        put!(res, f(ps, n, args...; kwargs...))

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
  data

end

