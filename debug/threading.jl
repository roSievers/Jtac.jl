import ThreadPools
using Jtac
using Statistics
using Printf

function record_threaded2(player, n; gpu = false, copy = false, augment = false, kwargs...)
  @assert Model.is_async(Model.playing_model(player)) "Threaded self plays only work with Async models"
  @assert Threads.nthreads() > 1 "record_threaded requires at least two threads"
  @assert Threads.threadid() == 1 "record_threaded can only be called from the master thread"

  # count moves and games
  moves = Threads.Atomic{Int}(0)
  games = Threads.Atomic{Int}(0)

  move_cb = () -> Threads.atomic_add!(moves, 1)
  game_cb = () -> Threads.atomic_add!(games, 1)

  # recording times to measure moves / second
  start_time = time()
  peak = 0

  update = t -> begin
    dt = t - start_time
    mps = moves[] / dt
    if mps > peak
      peak = mps
    end
    @printf "\e[2K\e[1G%.2f m/s (%d / %.2f)  |  %d game(s) finished" mps moves[] dt games[]
  end

  dss = nothing
  async_stats = []
  l = ReentrantLock()
  if !copy
    player = Model.tune(player; gpu)
  end

  @sync begin
    @async begin
      tickets = Player.ticket_sizes(n, Threads.nthreads() - 1)
      @show tickets
      dss = ThreadPools.bmap(tickets) do ticket
        if copy && gpu
          p = Model.tune(Base.copy(player), gpu = true)
        elseif copy
          p = Base.copy(player)
        else
          p = player
        end
        ds = Player.record( p, ticket
                     , callback = game_cb, callback_move = move_cb
                     , merge = false, distributed = false
                     , augment = augment, kwargs...)
        if copy
          lock(l) do
            avg = mean(p.model.history)
            std = Statistics.std(p.model.history)
            push!(async_stats, (Threads.threadid(), avg, std))
          end
        end
        ds
      end
    end
    @async while games[] < n
      sleep(0.25)
      update(time())
    end
  end

  dss = vcat(dss...)

  states_per_game = length.(dss)
  avg = mean(states_per_game)
  std = var(states_per_game) |> sqrt
  min = minimum(states_per_game)
  max = maximum(states_per_game)
  @printf "\n%d states created in %.2f seconds\n" sum(states_per_game) (time() - start_time)
  @printf "peak: %.2f m/s, avg: %.2f m/s\n" peak (sum(states_per_game) / (time() - start_time))
  @printf "%.2f ± %.2f states per match (min: %d, max: %d)\n" avg std min max

  println()

  if copy
    for (id, avg, std) in async_stats
      @printf "%.2f ± %.2f parallel state evaluations on thread %d\n" avg std id
    end
  else
    avg = mean(player.model.history)
    std = Statistics.std(player.model.history)
    @printf "%.2f ± %.2f parallel state evaluations\n" avg std
  end

  dss
end


G = Game.MetaTac
model = Model.NeuralModel(G, Model.@chain G Conv(128, "relu", padding = 1) Dense(128, "relu"))
model = Model.tune(model, async = true)
player = Player.MCTSPlayer(model, power = 100)


