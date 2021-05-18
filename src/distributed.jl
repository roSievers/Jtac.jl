
"""
Specification of an MCTS or Intuition player used to bring players to other
processes.
"""
struct PlayerSpec

  # Reference model
  model :: AbstractModel{G, false} where {G <: AbstractGame}

  # Player parameters
  power       :: Int      # power = 0 is used for IntuitionPlayers
  temperature :: Float32
  exploration :: Float32
  dilution    :: Float32
  name        :: String

end

function PlayerSpec(player :: MCTSPlayer)
  model = Model.base_model(player) |> Model.to_cpu
  PlayerSpec( model, player.power, player.temperature
            , player.exploration, player.dilution, player.name )
end

function PlayerSpec(player :: IntuitionPlayer)
  model = Model.base_model(player) |> Model.to_cpu
  PlayerSpec(model, 0, player.temperature, 0., 0., player.name)
end

"""
    build_player(spec; gpu = false, async = false)

Derive a player from a specification `spec`. The model of the player is
transfered to the gpu or brought in async mode if the respective flags are set.
"""
function build_player(spec :: PlayerSpec; gpu = false, async = false)
  model = spec.model
  if model isa Model.NeuralModel 
    model = gpu   ? Model.to_gpu(model) : model
    model = async ? Model.Async(model)  : model
  end
  if spec.power <= 0
    Player.IntuitionPlayer( model
                          , temperature = spec.temperature
                          , name = spec.name )
  else
    Player.MCTSPlayer( model
                     , power = spec.power
                     , temperature = spec.temperature
                     , exploration = spec.exploration
                     , dilution = spec.dilution
                     , name = spec.name )
  end
end

# -------- Packing (Async) Players ------------------------------------------- #

function pack(p :: Union{IntuitionPlayer, MCTSPlayer})
  (PlayerSpec(p), isa(playing_model(p), Async), on_gpu(training_model(p)))
end

function unpack(p :: Tuple{PlayerSpec, Bool, Bool})
  build_player(p[1]; async = p[2], gpu = p[3])
end

models_on_gpu(packs) = count(p[3] for p in packs)

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

  # Make players serializable (no gpu memory and async worker threads allowed)
  splayers = pack.(players)

  # Count the GPUs
  gpu = models_on_gpu(splayers) > 0
  devices = length(CUDA.devices())
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

