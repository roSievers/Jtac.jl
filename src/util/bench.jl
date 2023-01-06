
using LinearAlgebra

import ThreadPools
import CUDA

function timeit(f, trials)
  dt = @elapsed CUDA.@sync for _ in 1:trials
    f()
  end
end

function layer_data(sz, batchsize)
  a = "relu"
  chain = Model.Chain(
    [ Model.Conv(sz, sz, a, padding = 1)
    , Model.Batchnorm(sz, a)
    , Model.Conv(sz, sz, a, padding = 1)
    , Model.Batchnorm(sz, a)
    ]...
  )
  [ "conv$(sz)x$(sz)" => (() -> rand(Float32, (9, 9, sz, batchsize)), () -> Model.Conv(sz, sz, a, padding = 1))
  , "dense$(8sz)x$(sz)" => (() -> rand(Float32, (8sz, batchsize)), () -> Model.Dense(8sz, sz, a))
  , "batchnorm$sz" => (() -> rand(Float32, (9, 9, sz, batchsize)), () -> Model.Batchnorm(sz, a))
  , "chain$sz" => (() -> rand(Float32, (9, 9, sz, batchsize)), () -> chain)
  , "residual$sz" => (() -> rand(Float32, (9, 9, sz, batchsize)), () -> Model.Residual(chain))
  ]
end

function layer_cpu(sz = 256; batchsize = sz, trials = 100)
  layers = layer_data(sz, batchsize)
  @printf "%15s  %7s  %11s\n" "operation" "runtime" "inputs / ms"
  println("  ", join(repeat("-", 38)))
  for (key, (get_data, get_layer)) in layers
    @printf "%15s  " key
    flush(stdin)
    data = get_data()
    layer = get_layer()
    # precompile
    layer(data)
    # measure
    dt = @elapsed for _ in 1:trials layer(data) end
    dt = dt / batchsize / trials * 1000
    # report
    @printf "%7.2g  %11.2f\n" dt 1/dt
  end
end

function layer_gpu(sz = 256; batchsize = sz, trials = 100)
  @assert CUDA.functional() "No CUDA support detected"
  layers = layer_data(sz, batchsize)
  @printf "%22s  %7s  %10s\n" "operation (backend)" "runtime" "inputs / ms"
  println("  ", join(repeat("-", 43)))

  upload_data = layers[1][2][1]()
  for at in ["knet", "cuda"]

    # set gpu backend
    Model.atype_gpu!(at)

    # upload task
    @printf "%15s (%4s)  " "upload" at
    flush(stdin)
    Model.to_gpu(upload_data)
    dt = @elapsed CUDA.@sync for _ in 1:trials Model.to_gpu(upload_data) end
    dt = dt / batchsize / trials * 1000
    @printf "%7.2g  %11.2f\n" dt 1/dt
    GC.gc(); CUDA.reclaim()

    # download task
    @printf "%15s (%4s)  " "download" at
    flush(stdin)
    download_data = Model.to_gpu(upload_data)
    Model.to_cpu(download_data)
    dt = @elapsed CUDA.@sync for _ in 1:trials Model.to_cpu(download_data) end
    dt = dt / batchsize / trials * 1000
    @printf "%7.2g  %11.2f\n" dt 1/dt
    GC.gc(); CUDA.reclaim()
  end

  println()

  for (key, (get_data, get_layer)) in layers
    for at in ["knet", "cuda"]
      # set gpu backend
      Model.atype_gpu!(at)

      @printf "%15s (%4s)  " key at
      flush(stdin)
      data = get_data() |> Model.to_gpu
      layer = get_layer() |> Model.to_gpu

      # precompile
      layer(data)

      # measure
      dt = @elapsed CUDA.@sync for _ in 1:trials layer(data) end
      dt = dt / batchsize / trials * 1000
      @printf "%7.2g  %11.2f\n" dt 1/dt
      GC.gc(); CUDA.reclaim()
    end
  end
end

function model_gpu(model; trials = 1000)
  @assert model isa Model.NeuralModel
  cpumodel = Model.to_cpu(model)
  G = Model.gametype(model)
  @printf "     mode (backend)  bsize     games/s  moves/s (@power 250)\n"
  println(" ", repeat("-", 60))
  for batchsize in [16, 32, 64, 128, 256, 512]
    games = [Game.random_instance(G) for _ in 1:batchsize]

    for at in ["cuda", "knet"]
      # set gpu backend
      Model.atype_gpu!(at)
      data = Game.array(games) |> Model.to_gpu
      model = cpumodel |> Model.to_gpu

      # raw model throughput
      model(data)
      dt = @elapsed for _ in 1:trials
        res = model(data)
        res = Model.to_cpu.(res)
      end
      dt = dt / batchsize / trials
      sps = 1/dt
      mps = sps / 250
      @printf "%12s (%4s)    %3d  %10.2f  %7.2f\n" "raw" at batchsize sps mps

      # model throughput with prior conversion and upload
      model(games)
      dt = @elapsed for _ in 1:trials
        res = model(games)
        res = Model.to_cpu.(res)
      end
      dt = dt / batchsize / trials
      sps = 1/dt
      mps = sps / 250
      @printf "%12s (%4s)    %3d  %10.2f  %7.2f\n" "raw+upload" at batchsize sps mps

      # async
      amodel = Model.Async(model, max_batchsize = batchsize)

      # async model throughput
      res = Vector(undef, batchsize)
      @sync for (i, game) in enumerate(games)
        @async res[i] = Model.apply(amodel, game)
      end

      dt = @elapsed for _ in 1:trials
        res = Vector(undef, batchsize)
        foreach_async(1:length(games)) do i
          res[i] = Model.apply(amodel, games[i])
        end
      end
      dt = dt / batchsize / trials
      sps = 1/dt
      mps = sps / 250
      bs = sum(amodel.profile.batchsize) / length(amodel.profile.batchsize)
      @printf "%12s (%4s)    %3d  %10.2f  %7.2f (avg. batchsize %.2f)\n" "async" at batchsize sps mps bs

      GC.gc(); CUDA.reclaim()
      println()
    end
  end

end

function foreach_async(f, it)
  @sync for i in it
    @async f(i)
  end
end

function throughput(model, trials = 1000)
  @assert model isa Model.NeuralModel
  G = Model.gametype(model)
  for batchsize in [16, 32, 64, 128, 256, 512]
    games = [Game.random_instance(G) for _ in 1:batchsize]
    data = Game.array(games)
    if Model.on_gpu(model)
      data = Model.to_gpu(data)
    end
    println("batchsize $batchsize")

    # direct application with and without conversion
    for (descr, input) in [("without conversion:", data), ("with conversion:   ", games)]
      dt = @elapsed for _ in 1:trials
        res = model(input)
        res = Model.to_cpu.(res)
      end
      sps = batchsize * trials/dt
      mps = sps / 250
      @printf "  %s %.2f states/s" descr sps
      @printf " (%.2f m/s at power 250)\n" mps
    end

    # asyncmap model
    descr = "asyncmap model:    "
    amodel = Model.Async(model; max_batchsize = batchsize)
    dt = @elapsed for _ in 1:trials
      asyncmap(games, ntasks = batchsize) do game
        res = Model.apply(amodel, game)
      end
    end
    sps = batchsize * trials/dt
    mps = sps / 250
    bsize = sum(amodel.profile.batchsize) / length(amodel.profile.batchsize)
    @printf "  %s %.2f states/s" descr sps
    @printf " (%.2f m/s at power 250)" mps
    @printf " (avg. batchsize %d)\n" bsize
    amodel = nothing
    GC.gc()

    # @async model
    descr = "@async model:      "
    amodel = Model.Async(model; max_batchsize = batchsize)
    dt = @elapsed for _ in 1:trials
      ts = map(games) do game
        @async Model.apply(amodel, game)
      end
      res = fetch.(ts)
    end
    sps = batchsize * trials/dt
    mps = sps / 250
    bsize = sum(amodel.profile.batchsize) / length(amodel.profile.batchsize)
    @printf "  %s %.2f states/s" descr sps
    @printf " (%.2f m/s at power 250)" mps
    @printf " (avg. batchsize %d)\n" bsize
    amodel = nothing
    GC.gc()

    # async-like 
    descr = "async-like model:  "
    dt = @elapsed async_like(trials, model, games)
    sps = batchsize * trials/dt
    mps = sps / 250
    @printf "  %s %.2f states/s" descr sps
    @printf " (%.2f m/s at power 250)\n" mps
    println()

  end
end

function async_like(trials, model, games)
  bs = length(games)
  ch = Channel(bs)

  t = @async begin
    G = Model.gametype(model)
    while true
      games = G[]
      conds = Threads.Condition[]
      while length(games) < bs
        game, c = take!(ch)
        push!(games, game)
        push!(conds, c)
        yield()
      end
      @assert length(games) == bs
      val, pol = model(games)
      v, p = Model.to_cpu(val), Model.to_cpu(pol)
      for i in 1:length(conds)
        c = conds[i]
        lock(c) do
          notify(c, (value = v[i], policy = p[:, i]))
        end
      end
    end
  end

  n = length(games)
  gamesets = [games[16i-15:16i] for i in 1:div(n, 16)]
  for _ in 1:trials
    i = 1
    res = Vector(undef, n)
#    @show res
    @sync begin
      map(gamesets) do games
        @async begin
          map(games) do game
            @async begin
              c = Threads.Condition()
              put!(ch, (game, c))
              lock(c) do
                res[i] = wait(c)
                i += 1
              end
            end
          end
        end
      end
    end
  end
  close(ch)
  yield()
  @assert Base.istaskdone(t)
end


function record(player, n; augment = false, kwargs...)

  start_time = time()
  moves = 0
  games = 0
  peak = 0

  # dataset variable
  dss = nothing

  move_ch = RemoteChannel(() -> Channel{Bool}(1000))
  game_ch = RemoteChannel(() -> Channel{Bool}(1000))
  
  move_cb = _ -> (put!(move_ch, true); yield())
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

  move_cb = _ -> Threads.atomic_add!(moves, 1)
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
                     , merge = false
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

function train(player, ds = nothing; batchsize = 512, epochs = 20, kwargs...)
  if isnothing(ds)
    print("generating dummy dataset...")
    G = Model.gametype(player)
    p = Player.MCTSPlayer(Model.RandomModel(), power = 10)
    ds = Player.record(p, 200, instance = G)
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
      dt = @elapsed Training.train_step!(model, cache)
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
  train(player, merge(ds))

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
  train(player, merge(ds))


  BLAS.set_num_threads(t) 
  nothing
end

function benchmark_gpu(G = Game.MetaTac; threads = false, async = 64, matches = async)

  println("\nMetaTac (simple):")
  println("-- recording --")
  model = Model.Zoo.ZeroConv(G, blocks = 2, filters = 64)
  model = Model.tune(model, gpu = true, async = async)
  player = Player.MCTSPlayer(model, power = 250)
  if threads && Threads.nthreads() > 1
    ds = record_threaded(player, matches)
  else
    ds = record(player, matches)
  end
  println()
  println("-- training --")
  train(player, merge(ds))

  println("\nMetaTac (medium):")
  println("-- recording --")
  model = Model.Zoo.ZeroConv(G, blocks = 6, filters = 128)
  model = Model.tune(model, gpu = true, async = async)
  player = Player.MCTSPlayer(model, power = 250)
  if threads && Threads.nthreads() > 1
    ds = record_threaded(player, matches)
  else
    ds = record(player, matches)
  end
  println()
  println("-- training --")
  train(player, merge(ds))

  println("\nMetaTac (large):")
  println("-- recording --")
  model = Model.Zoo.ZeroConv(G, blocks = 8, filters = 256)
  model = Model.tune(model, gpu = true, async = async)
  player = Player.MCTSPlayer(model, power = 250)
  if threads && Threads.nthreads() > 1
    ds = record_threaded(player, matches)
  else
    ds = record(player, matches)
  end
  println()
  println("-- training --")
  train(player, merge(ds))

  nothing
end


