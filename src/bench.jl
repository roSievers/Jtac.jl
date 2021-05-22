
using ..Jtac
using Distributed
using Statistics
using Printf

function record_self(player, n; augment = false, kwargs...)

  start_time = time()
  moves = 0
  games = 0
  peak = 0

  # dataset variable
  dss = nothing

  move_ch = RemoteChannel(() -> Channel{Bool}(1000))
  game_ch = RemoteChannel(() -> Channel{Bool}(1000))
  
  move_cb = () -> (put!(move_ch, true); yield())
  game_cb = () -> (put!(game_ch, true); yield())

  update = t -> begin
    dt = t - start_time
    mps = moves / dt
    if mps > peak
      peak = mps
    end
    @printf "\e[2K\e[1G%.2f m/s (%d / %.2f)  |  %d game(s) finished" mps moves dt games
  end

  @sync begin
    @async while take!(move_ch)
      moves += 1
    end
    @async while take!(game_ch)
      games += 1
    end
    @async while games < n
      sleep(0.25)
      update(time())
    end
    @async begin
      dss = Training.record_self(player, n; callback = game_cb, callback_move = move_cb
                       , merge = false, augment = augment, kwargs...)
      put!(move_ch, false)
      put!(game_ch, false)
    end
  end

  states_per_game = length.(dss)
  avg = mean(states_per_game)
  std = var(states_per_game) |> sqrt
  min = minimum(states_per_game)
  max = maximum(states_per_game)

  # TODO: print peak rate and average rate
  @printf "\n%d states created in %.2f seconds\n" sum(states_per_game) (time() - start_time)
  @printf "peak: %.2f m/s, avg: %.2f m/s\n" peak (sum(states_per_game) / (time() - start_time))
  @printf "%.2f ± %.2f states per game (min: %d, max: %d)\n" avg std min max
  dss
end

# Note: Threading like done in this function works as intended, but it does not
# seem to yield significant performance advantages for NeuralModels.  For small
# networks, multiprocessing (and occupying the GPU with several independently
# working copies of the network) seems to perform better; for large networks,
# the threaded version is only marginally faster than the non-threaded version,
# since most time is (probably) spent working for the Gpu.
function record_self_threaded(player, n; augment = false, kwargs...)
  @assert Model.playing_model(player) isa Model.Async "Threaded self plays only work with Async models"
  @assert Threads.nthreads() > 1 "record_self_threaded requires at least two threads"
  @assert Threads.threadid() == 1 "record_self_threaded can only be called from the master thread"

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

  # variable to save datasets
  dss = nothing

  # We only want to work on non-master threads
  nworkerthreads = Threads.nthreads() - 1
  tickets = Player.ticket_sizes(n, nworkerthreads)
  tasks = map(1:nworkerthreads) do i
    Threads.@spawn begin
      @assert Threads.threadid() != 1 "master thread received workload"
      println("Thread $(Threads.threadid()) with index $i starts working")
      ds = Training.record_self( player, tickets[i]
                 , callback = game_cb, callback_move = move_cb
                 , merge = false, distributed = false, augment = augment, kwargs...)
      ds
    end
  end

  @sync begin
    @async begin
      dss = vcat(fetch.(tasks)...)
    end
    @async while games[] < n
      sleep(0.25)
      update(time())
    end
  end

  states_per_game = length.(dss)
  avg = mean(states_per_game)
  std = var(states_per_game) |> sqrt
  min = minimum(states_per_game)
  max = maximum(states_per_game)
  @printf "\n%d states created in %.2f seconds\n" sum(states_per_game) (time() - start_time)
  @printf "peak: %.2f m/s, avg: %.2f m/s\n" peak (sum(states_per_game) / (time() - start_time))
  @printf "%.2f ± %.2f states per game (min: %d, max: %d)\n" avg std min max
  dss
end

