
using LinearAlgebra

function layerdata(sz, batchsize)
  a = "relu"
  chain = Chain(
    [ Conv(sz, sz, a, pad = 1)
    , Batchnorm(sz, a)
    , Conv(sz, sz, a, pad = 1)
    , Batchnorm(sz, a)
    ]
  )
  [ "conv$(sz)x$(sz)" => (() -> rand(Float32, (9, 9, sz, batchsize)), () -> Conv(sz, sz, a, pad = 1))
  , "dense$(8sz)x$(sz)" => (() -> rand(Float32, (8sz, batchsize)), () -> Dense(8sz, sz, a))
  , "batchnorm$sz" => (() -> rand(Float32, (9, 9, sz, batchsize)), () -> Batchnorm(sz, a))
  , "chain$sz" => (() -> rand(Float32, (9, 9, sz, batchsize)), () -> chain)
  , "residual$sz" => (() -> rand(Float32, (9, 9, sz, batchsize)), () -> Residual(chain))
  ]
end

function layers(sz = 256; batchsize = sz, trials = 100, backend = :default)
  T = Model.arraytype(backend)
  layers = layerdata(sz, batchsize)

  @printf "%15s  %7s  %11s\n" "operation" "runtime" "inputs / ms"
  println("  ", join(repeat("-", 38)))
  for (key, (get_data, get_layer)) in layers
    @printf "%15s  " key
    flush(stdin)
    data = get_data()
    data = convert(T, data)
    layer = get_layer()
    layer = Model.adapt(backend, layer)

    # precompile
    layer(data)
    # measure
    dt = @elapsed for _ in 1:trials
      # conversion should trigger synchronization for e.g. CUDA
      convert(Array{Float32}, layer(data))
    end
    dt = dt / batchsize / trials * 1000
    # report
    @printf "%7.2g  %11.2f\n" dt 1/dt
  end
end

function model( model :: NeuralModel{G}
              ; trials = 1000
              , batchsizes = [16, 32, 64, 128, 256, 512]
              , backend = :default ) where {G <: AbstractGame}

  T = Model.arraytype(backend)
  model = Model.adapt(backend, model)

  @printf "     mode (backend)  bsize     games/s  moves/s (@power 250)\n"
  println(" ", repeat("-", 60))

  for batchsize in batchsizes
    games = [Game.randominstance(G) for _ in 1:batchsize]
    data = convert(T, Game.array(games))

    model(data) # warmup
    dt = @elapsed for _ in 1:trials
      # conversion should trigger synchronization for e.g. CUDA
      convert.(Array{Float32}, model(data))
    end
    dt = dt / batchsize / trials
    sps = 1/dt
    mps = sps / 250
    @printf "%12s (%4s)    %3d  %10.2f  %7.2f\n" "direct" backend batchsize sps mps

    model(games) # warmup
    dt = @elapsed for _ in 1:trials
      # conversion should trigger synchronization for e.g. CUDA
      convert.(Array{Float32}, model(games))
    end
    dt = dt / batchsize / trials
    sps = 1/dt
    mps = sps / 250
    @printf "%12s (%4s)    %3d  %10.2f  %7.2f\n" "+conversion" backend batchsize sps mps

    # async model throughput
    amodel = AsyncModel(model, batchsize = batchsize)
    res = Vector(undef, batchsize)

    @sync for (i, game) in enumerate(games) # warmup
      @async res[i] = Model.apply(amodel, game)
    end

    dt = @elapsed for _ in 1:trials
      @elapsed @sync for (i, game) in enumerate(games)
        @async res[i] = Model.apply(amodel, game)
      end
    end

    dt = dt / batchsize / trials
    sps = 1/dt
    mps = sps / 250
    bs = sum(amodel.profile.batchsize) / length(amodel.profile.batchsize)
    @printf "%12s (%4s)    %3d  %10.2f  %7.2f (avg. batchsize %.2f)\n" "async" backend batchsize sps mps bs
    println()
  end
end


function record(player, n; augment = false, kwargs...)
  start_time = time()
  moves = 0
  matches = 0
  peak = 0

  # dataset variable
  dss = nothing

  move_ch = Channel{Bool}(1000)
  match_ch = Channel{Bool}(1000)
  
  move_cb = _ -> (put!(move_ch, true); yield())
  match_cb = () -> (put!(match_ch, true); yield())

  update = t -> begin
    dt = t - start_time
    mps = moves / dt
    if mps > peak
      peak = mps
    end
    @printf "\e[2K\e[1G%.2f m/s (%d / %.2f)  |  %d matches finished" mps moves dt matches
  end

  @sync begin
    @async while take!(move_ch)
      moves += 1
    end
    @async while take!(match_ch)
      matches += 1
    end
    @async while matches < n
      sleep(0.25)
      update(time())
    end
    @async begin
      dss = Training.record(
        player,
        n;
        callback_match = match_cb,
        callback_move = move_cb,
        merge = false,
        progress = false,
        augment = augment,
        kwargs...
      )
      put!(move_ch, false)
      put!(match_ch, false)
    end
  end

  states_per_match = length.(dss)
  avg = mean(states_per_match)
  std = var(states_per_match) |> sqrt
  min = minimum(states_per_match)
  max = maximum(states_per_match)

  @printf "\n%d states created in %.2f seconds\n" sum(states_per_match) (time() - start_time)
  @printf "peak: %.2f m/s, avg: %.2f m/s\n" peak (sum(states_per_match) / (time() - start_time))
  @printf "%.2f ± %.2f states per match (min: %d, max: %d)\n" avg std min max
  dss
end


function learn(player, ds = nothing; batchsize = 512, epochs = 20, kwargs...)

  if isnothing(ds)
    print("generating dummy dataset...")
    G = Model.gametype(player)
    p = MCTSPlayer(Model.RandomModel(G), power = 10)
    ds = Training.record(p, 200, progress = false)
    println(" done")
  end
  println("dataset size ", length(ds))

  model = copy(trainingmodel(player))

  # Precompile
  learn!(
    model,
    ds[1:5],
    batchsize = 5,
    progress = false,
    verbose = false
  )

  epoch = 1
  time_start = time()

  callback_epoch = n -> (epoch += 1)
  callback_batch = n -> begin
    dt = time() - time_start
    games = n * batchsize
    gps = games / dt
    @printf "\e[2K\e[1G%.2f g/s (%d / %.2f)  |  %d epoch(s) finished" gps games dt epoch
  end

  learn!(
    model,
    ds;
    batchsize,
    callback_epoch,
    callback_batch,
    kwargs...,
    partial = false,
    progress = false,
    verbose = false,
  )

  batches = div(length(ds), batchsize)
  dt = time() - time_start
  println()
  @printf "trained on %d batches with batchsize %d in %.2f seconds\n" batches batchsize dt
  @printf "%.2f seconds per batch on average\n" dt / batches
end


function bench(model; threads = false, async = 64, power = 250, matches = 100, backend = :default)
  backend = getbackend(backend)
  model = Model.configure(model; backend, async)
  player = Player.MCTSPlayer(model, power = power)
  println("-- recording --")
  ds = record(player, matches, threads = threads)
  if Model.istrainable(backend)
    println()
    println("-- learning --")
    learn(player, merge(ds))
  end
end

function benchtiny(G = MetaTac; kwargs...)
  model = Model.Zoo.ZeroConv(G; blocks = 1, filters = 64)
  println("$G (tiny model)")
  bench(model; kwargs...)
end

function benchsmall(G = MetaTac; kwargs...)
  model = Model.Zoo.ZeroConv(G; blocks = 2, filters = 64, backend)
  println("$G (small model)")
  bench(model; kwargs...)
end

function benchmedium(G = MetaTac; kwargs...)
  model = Model.Zoo.ZeroConv(G; blocks = 6, filters = 128, backend)
  println("$G (medium model)")
  bench(model; kwargs...)
end

function benchlarge(G = MetaTac; kwargs...)
  model = Model.Zoo.ZeroRes(G; blocks = 8, filters = 256, backend)
  println("$G (large model)")
  bench(model; kwargs...)
end

function benchhuge(G = MetaTac; kwargs...)
  model = Model.Zoo.ZeroRes(G; blocks = 16, filters = 256, backend)
  println("$G (huge model)")
  bench(model; kwargs...)
end

