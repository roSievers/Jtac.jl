
import ThreadPools
using LinearAlgebra

function record(player, n; augment = false, kwargs...)

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
      dss = Player.record( player, n
                       ; callback = game_cb, callback_move = move_cb
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

  @printf "\n%d states created in %.2f seconds\n" sum(states_per_game) (time() - start_time)
  @printf "peak: %.2f m/s, avg: %.2f m/s\n" peak (sum(states_per_game) / (time() - start_time))
  @printf "%.2f ± %.2f states per game (min: %d, max: %d)\n" avg std min max
  dss
end

# Note: Threading like done in this function works as intended, but it does not
# seem to yield significant performance advantages for NeuralModel.  For small
# networks, multiprocessing (and occupying the GPU with several independently
# working copies of the network) seems to perform better; for large networks,
# the threaded version is only marginally faster than the non-threaded version,
# since most time is (probably) spent in the GPU anyway.
function record_threaded(player, n; copy_model = false, augment = false, kwargs...)
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

  # We only want to work on background threads
  nworkers = Threads.nthreads() - 1
  tickets = Player.ticket_sizes(n, nworkers)
  dss = nothing

  @sync begin
    # map to background threads
    @async begin
      dss = ThreadPools.bmap(tickets) do ticket
        p = copy_model ? copy(player) : player
        Player.record( p, ticket
                     , callback = game_cb, callback_move = move_cb
                     , merge = false, distributed = false
                     , augment = augment, kwargs...)
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
  @printf "%.2f ± %.2f states per game (min: %d, max: %d)\n" avg std min max
  dss
end

function train(player, ds = nothing; batchsize = 512, loss = Training.Loss(), epochs = 20, kwargs...)
  if isnothing(ds)
    print("generating dummy dataset...")
    G = Model.gametype(player)
    p = Player.MCTSPlayer(Model.RandomModel(), power = 10)
    ds = Player.record(p, 200, game = G)
    println(" done")
  end
  println("dataset size ", length(ds))

  model = Model.training_model(player)
  Training.set_optimizer!(model)
  gpu = Model.on_gpu(model)

  batches = Data.Batches( ds
                        , batchsize
                        , shuffle = true
                        , partial = true
                        , gpu = gpu )

  total = 0
  time = 0.
  for epoch in 1:epochs
    for (i, cache) in enumerate(batches)
      dt = @elapsed Training.train_step!(loss, model, cache)
      games = length(cache)
      total += games
      time += dt
      gps = total / time
      @printf "\e[2K\e[1G%.2f g/s (%d / %.2f)  |  %d epoch(s) finished" gps total time epoch
    end
  end
  n = length(batches)
  println()
  @printf "trained on %d batches with batchsize %d in %.2f seconds\n" (n * epochs) batchsize time
  @printf "%.2f seconds per batch on average\n" time/n

end


function benchmark_cpu(; threads = false, async = 25)
  t = BLAS.get_num_threads()
  BLAS.set_num_threads(1) 

  println("\nMetaTac (very simple):")
  println("-- recording --")
  model = Model.Zoo.ZeroConv(Game.MetaTac, blocks = 1, filters = 64)
  model = Model.tune(model, async = async)
  player = Player.MCTSPlayer(model, power = 250)
  if threads && Threads.nthreads() > 1
    ds = record_threaded(player, 100, copy_model = true)
  else
    ds = record(player, 100)
  end
  println()
  println("-- training --")
  train(player, merge(ds...))

  println("\nMetaTac (simple):")
  println("-- recording --")
  model = Model.Zoo.ZeroConv(Game.MetaTac, blocks = 2, filters = 64)
  model = Model.tune(model, async = async)
  player = Player.MCTSPlayer(model, power = 250)
  if threads && Threads.nthreads() > 1
    ds = record_threaded(player, 100, copy_model = true)
  else
    ds = record(player, 100)
  end
  println()
  println("-- training --")
  train(player, merge(ds...))


  BLAS.set_num_threads(t) 
  nothing
end

function benchmark_gpu(; threads = false, async = 100)

  println("\nMetaTac (simple):")
  println("-- recording --")
  model = Model.Zoo.ZeroConv(Game.MetaTac, blocks = 2, filters = 64)
  model = Model.tune(model, gpu = true, async = async)
  player = Player.MCTSPlayer(model, power = 250)
  if threads && Threads.nthreads() > 1
    ds = record_threaded(player, 100)
  else
    ds = record(player, 100)
  end
  println()
  println("-- training --")
  train(player, merge(ds...))

  println("\nMetaTac (medium):")
  println("-- recording --")
  model = Model.Zoo.ZeroConv(Game.MetaTac, blocks = 4, filters = 128)
  model = Model.tune(model, gpu = true, async = async)
  player = Player.MCTSPlayer(model, power = 250)
  if threads && Threads.nthreads() > 1
    ds = record_threaded(player, 100)
  else
    ds = record(player, 100)
  end
  println()
  println("-- training --")
  train(player, merge(ds...))

  println("\nMetaTac (large):")
  println("-- recording --")
  model = Model.Zoo.ZeroConv(Game.MetaTac, blocks = 8, filters = 256)
  model = Model.tune(model, gpu = true, async = async)
  player = Player.MCTSPlayer(model, power = 250)
  if threads && Threads.nthreads() > 1
    ds = record_threaded(player, 100)
  else
    ds = record(player, 100)
  end
  println()
  println("-- training --")
  train(player, merge(ds...))

  nothing
end


