
using Revise, Test
using Jtac
using Random

function packcycle(value, T = typeof(value), isequal = isequal)
  bytes = Pack.pack(value)
  uvalue = Pack.unpack(bytes, T)
  isequal(value, uvalue) &&
  all(bytes .== Pack.pack(uvalue))
end

@testset "Status" begin
  vals = [1, 0, -1]
  @test Game.isover(Game.undecided) == false
  @test Game.Status.(vals) .|> Game.isover |> all
  @test Game.Status.(vals) == [Game.win, Game.draw, Game.loss]
end

@testset "TicTacToe" begin
  game = Game.TicTacToe()
  @test size(Game.TicTacToe) == size(game) == size(Game.array(game)) == (3, 3, 1)
  @test size(Game.array([game, game])) == (3, 3, 1, 2)
  @test Game.policylength(Game.TicTacToe) == Game.policylength(game) == 9
  @test Game.activeplayer(game) == 1
  @test_throws BoundsError Game.move!(game, -1)

  Game.move!(game, 5)
  @test Game.legalactions(game) == [ i for i in 1:9 if i != 5]
  @test Game.activeplayer(game) == -1
  @test Game.randomaction(game) in Game.legalactions(game)

  Game.move!(game, 3)
  @test .!Game.isactionlegal.(game, [3, 5]) |> all
  @test Game.isactionlegal.(game, Game.legalactions(game)) |> all
  @test Game.status(game) == Game.undecided
  @test copy(game).board == game.board
  @test_throws AssertionError Game.move!(game, 5)

  game = Game.randommatch(game)
  @test Game.legalactions(game) == []
  @test Game.isover(game) == Game.isover(Game.status(game))

  games, policies = Game.augment(game, rand(Float32, 9))
  @test length(games) == length(policies) == 8
end

@testset "MetaTac" begin
  game = Game.MetaTac()
  @test size(Game.MetaTac) == size(game) == size(Game.array(game)) == (9, 9, 3)
  @test size(Game.array([game, game])) == (9, 9, 3, 2)
  @test Game.policylength(Game.MetaTac) == Game.policylength(game) == 81
  @test Game.activeplayer(game) == 1
  @test_throws BoundsError Game.move!(game, -1)

  Game.move!(game, 5)
  @test Game.activeplayer(game) == -1
  @test Game.randomaction(game) in Game.legalactions(game)

  Game.randommove!(game)
  @test .!Game.isactionlegal.(game, [3, 5]) |> all
  @test Game.isactionlegal.(game, Game.legalactions(game)) |> all
  @test Game.status(game) == Game.undecided
  @test copy(game).board == game.board
  @test_throws AssertionError Game.move!(game, 5)

  game = Game.randommatch(game)
  @test Game.legalactions(game) == []
  @test Game.isover(game) == Game.isover(Game.status(game))

  games, policies = Game.augment(game, rand(Float32, 81))
  @test length(games) == length(policies) == 8
end

@testset "Pack" begin
  @test packcycle(nothing)
  @test packcycle(true)
  @test packcycle(false)
  for v in [-100000000000, -1000, -100, -10, 0, 10, 100, 1000, 10000000000]
    @test packcycle(v)
  end
  for v in [rand(Float32), rand(Float64)]
    @test packcycle(v)
  end
  for v in [randstring(n) for n in (3, 16, 32, 100, 1000)]
    @test packcycle(v)
  end
  @test packcycle(rand(Float32, 1000))
  @test packcycle(rand(Float64, 1000))
  @test packcycle((:this, :is, "a tuple", (:with, true, :numbers), 5))
  @test packcycle((a = "named", b = "tuple", length = 3))

  # struct A
  #   a :: Vector{Float32}
  #   b :: String
  #   c :: Int
  # end

  # Pack.@pack A in MapFormat a in BinArrayFormat (a, b)
  # @test packcycle(A(rand(100), "test"), A, (x,y) -> x.a == y.a && x.b == y.b)

  # function A(a, b)
  #   A(a, b, 5)
  # end

  # Pack.@pack A (a, b)
  # @test packcycle(A(rand(100), "test"), A, (x,y) -> x.a == y.a && x.b == y.b)
  
end

@testset "Target" begin
  
  G = Game.TicTacToe

  vt = Target.DefaultValueTarget(G)
  pt = Target.DefaultPolicyTarget(G)

  policy = rand(Float32, 9)
  policy ./= sum(policy)
  ctx = Target.LabelContext(G(), policy, Game.loss, 0, G[], Vector{Float32}[])

  @test length(vt) == 1
  @test Target.label(vt, ctx) == [-1f0]
  @test Target.defaultlossfunction(vt) == :sumabs2

  @test length(pt) == Game.policylength(G)
  @test Target.label(pt, ctx) == policy
  @test Target.defaultlossfunction(pt) == :crossentropy

  @test packcycle(vt)
  @test packcycle(pt)
end


@testset "Model" begin

  @testset "Activation" begin
    data = rand(12, 9)

    for (name, a) in Util.lookup(Model.Activation)
      @test Pack.unpack(Pack.pack(a), Model.Activation) == a
      if a isa Model.Activation{true}
        @test all(a(data) .== a.f.(data))
      else
        @test all(a(data) .== a.f(data))
      end
    end
  end

  @testset "Backend" begin
    backend = Model.DefaultBackend{Array{Float32}}()
    @test backend == Model.getbackend(:default)
    @test Model.istrainable(backend) == false
    @test Model.arraytype(backend) == Array{Float32}

    backend64 = Model.adapt(Array{Float64}, backend)
    @test backend64 == Model.getbackend(:default64)
    @test Model.istrainable(backend64) == false
    @test Model.arraytype(backend64) == Array{Float64}

    backend = Model.getbackend(:default)
    @test backend == Pack.unpack(Pack.pack(backend), Model.Backend)
  end
  
  @testset "Layer" begin

    @testset "Dense" begin
      layer = Model.Dense(50, 20, :sigmoid)
      layerp = Pack.unpack(Pack.pack(layer), Model.Layer)
      layerc = copy(layer)
      @test Model.adapt(:default, layer) === layer

      input = rand(Float32, (50, 10))
      out = layer(input)
      @test out isa Matrix{Float32}
      @test size(out) == (20, 10)
      @test all(out .== layerp(input))
      @test all(out .== layerc(input))

      layer64 = Model.adapt(Array{Float64}, layer)
      layer64p = Pack.unpack(Pack.pack(layer64), Model.Layer)
      layer64c = copy(layer64)
      @test Model.adapt(:default64, layer64) === layer64

      input = rand(Float64, (50, 10))
      out = layer64(input)
      @test out isa Matrix{Float64}
      @test size(out) == (20, 10)
      @test all(out .== layer64p(input))
      @test all(out .== layer64c(input))
    end

    @testset "Conv" begin
      layer = Model.Conv(50, 20, :sigmoid)
      layerp = Pack.unpack(Pack.pack(layer), Model.Layer)
      layerc = copy(layer)
      @test Model.adapt(:default, layer) === layer

      input = rand(Float32, (9, 8, 50, 10))
      out = layer(input)
      @test out isa Array{Float32, 4}
      @test size(out) == (7, 6, 20, 10)
      @test all(out .== layerp(input))
      @test all(out .== layerc(input))

      layer64 = Model.adapt(Array{Float64}, layer)
      layer64p = Pack.unpack(Pack.pack(layer64), Model.Layer)
      layer64c = copy(layer64)
      @test Model.adapt(:default64, layer64) === layer64

      input = rand(Float64, (9, 8, 50, 10))
      out = layer64(input)
      @test out isa Array{Float64, 4}
      @test size(out) == (7, 6, 20, 10)
      @test all(out .== layer64p(input))
      @test all(out .== layer64c(input))
    end


    @testset "Batchnorm" begin
      layer = Model.Batchnorm(50, :relu)
      layerp = Pack.unpack(Pack.pack(layer), Model.Layer)
      layerc = copy(layer)
      @test Model.adapt(:default, layer) === layer

      input = rand(Float32, (9, 8, 50, 10))
      out = layer(input)
      @test out isa Array{Float32, 4}
      @test size(out) == (9, 8, 50, 10)
      @test all(out .== layerp(input))
      @test all(out .== layerc(input))

      layer64 = Model.adapt(Array{Float64}, layer)
      layer64p = Pack.unpack(Pack.pack(layer64), Model.Layer)
      layer64c = copy(layer64)
      @test Model.adapt(:default64, layer64) === layer64

      input = rand(Float64, (9, 8, 50, 10))
      out = layer64(input)
      @test out isa Array{Float64, 4}
      @test size(out) == (9, 8, 50, 10)
      @test all(out .== layer64p(input))
      @test all(out .== layer64c(input))
    end
  end

  @testset "Chain" begin
    sz = (50, 50, 3)
    layer = Model.@chain sz Dense(20, :sigmoid) Dense(50, :relu)
    player = Pack.unpack(Pack.pack(layer), Model.Layer)
    clayer = copy(layer)
  
    @test layer isa Model.Chain{Array{Float32}}
    input = rand(Float32, sz..., 10)
    out = layer(input)
    @test out isa Matrix{Float32}
    @test size(out) == (50, 10)
    @test all(out .== player(input))
    @test all(out .== clayer(input))

    layer64 = Model.adapt(Array{Float64}, layer)
    player64 = Pack.unpack(Pack.pack(layer64), Model.Layer)
    clayer64 = copy(layer64)
    @test layer64 isa Model.Chain{Array{Float64}}
    input = rand(Float64, sz..., 10)
    out = layer64(input)
    @test out isa Matrix{Float64}
    @test size(out) == (50, 10)
    @test all(out .== player64(input))
    @test all(out .== clayer64(input))
  end

  @testset "Residual" begin
    sz = (50, 50, 10)
    layer = Model.@residual sz Conv(10, pad = 1) Batchnorm() Conv(10, pad = 1)
    player = Pack.unpack(Pack.pack(layer), Model.Layer)
    clayer = copy(layer)
  
    @test layer isa Model.Residual{Array{Float32}}
    input = rand(Float32, sz..., 10)
    out = layer(input)
    @test out isa Array{Float32, 4}
    @test size(out) == (50, 50, 10, 10)
    @test all(out .== player(input))
    @test all(out .== clayer(input))

    layer64 = Model.adapt(Array{Float64}, layer)
    player64 = Pack.unpack(Pack.pack(layer64), Model.Layer)
    clayer64 = copy(layer64)
    @test layer64 isa Model.Residual{Array{Float64}}
    input = rand(Float64, sz..., 10)
    out = layer64(input)
    @test out isa Array{Float64, 4}
    @test size(out) == (50, 50, 10, 10)
    @test all(out .== player64(input))
    @test all(out .== clayer64(input))
  end

  @testset "Model" begin

    @testset "Base" begin
      G = Game.TicTacToe
      for model in [Model.DummyModel(G), Model.RandomModel(G), Model.RolloutModel(G)]
        model = Model.DummyModel(G)
        @test Model.gametype(model) == G
        @test Model.ntasks(model) == 1
        @test Model.isasync(model) == false
        @test Model.basemodel(model) == model
        @test Model.playingmodel(model) == model
        @test Model.trainingmodel(model) == nothing
        @test Target.targetnames(model) == [:value, :policy]
        @test Target.targets(model) == (
          value = Target.DefaultValueTarget(G),
          policy = Target.DefaultPolicyTarget(G),
        )
        res = Model.apply(model, G(), targets = [:value, :policy])
        @test haskey(res, :value)
        @test haskey(res, :policy)
        @test length(res.value) == 1
        @test length(res.policy) == Game.policylength(G)
        res = Model.apply(model, G(), targets = [:policy])
        @test haskey(res, :policy)
        @test length(res.policy) == Game.policylength(G)

        model_ = Pack.unpack(Pack.pack(model), Model.AbstractModel)
        @test model == model_
      end
    end

    @testset "NeuralModel" begin
      G = Game.TicTacToe
      for model in [
        Model.Zoo.MLP(G, [64, 32], :relu),
        Model.Zoo.ZeroConv(G, filters = 16),
        Model.Zoo.ZeroRes(G, filters = 16),
      ]
        @test Model.gametype(model) == G
        @test Model.ntasks(model) == 1
        @test Model.isasync(model) == false
        @test Model.basemodel(model) == model
        @test Model.playingmodel(model) == model
        @test Model.trainingmodel(model) == nothing
        @test Target.targetnames(model) == [:value, :policy]
        @test Target.targets(model) == (
          value = Target.DefaultValueTarget(G),
          policy = Target.DefaultPolicyTarget(G),
        )

        res = Model.apply(model, G(), targets = [:value, :policy])
        @test haskey(res, :value)
        @test haskey(res, :policy)
        @test length(res.value) == 1
        @test length(res.policy) == Game.policylength(G)
        res = Model.apply(model, G(), targets = [:policy])
        @test haskey(res, :policy)
        @test length(res.policy) == Game.policylength(G)

        games = [Game.randommove!(G(), 4) for _ in 1:100]
        model_ = Pack.unpack(Pack.pack(model), Model.AbstractModel)
        @test all(model(games) .== model_(games))

        Base.Filesystem.mktemp() do path, io
          Model.save(path, model)
          model_ = Model.load(path, format = :jtm)
          @test all(model(games) .== model_(games))
        end

        Model.addtarget!(model, :dummy, Target.DummyTarget(G))
        res = Model.apply(model, G(), targets = [:value, :policy, :dummy])
        @test haskey(res, :value)
        @test haskey(res, :policy)
        @test haskey(res, :value)
        @test length(res.value) == 1
        @test length(res.policy) == Game.policylength(G)
        @test length(res.dummy) == length(Target.DummyTarget(G))

        @test Target.compatibletargets(model, model_) == Target.targets(model_)

        model64 = Model.adapt(:default64, model)
        @test all(isa.(model64([G()]), Array{Float64}))
        @test Model.getbackend(model64) isa Model.DefaultBackend{Array{Float64}}
        @test Model.arraytype(model64) == Array{Float64}
      end
    end

    @testset "AsyncModel" begin
      G = Game.TicTacToe
      base =  Model.Zoo.MLP(G, [64, 32], :relu)
      model = Model.AsyncModel(base)
      @test Model.gametype(model) == G
      @test Model.ntasks(model) > 1
      @test Model.isasync(model) == true
      @test Model.basemodel(model) == base
      @test Model.playingmodel(model) == model
      @test Model.trainingmodel(model) == nothing
      @test Target.targetnames(model) == [:value, :policy]
      @test Target.targets(model) == (
        value = Target.DefaultValueTarget(G),
        policy = Target.DefaultPolicyTarget(G),
      )

      res = Model.apply(model, G(), targets = [:value, :policy])
      @test haskey(res, :value)
      @test haskey(res, :policy)
      @test length(res.value) == 1
      @test length(res.policy) == Game.policylength(G)
      res = Model.apply(model, G(), targets = [:policy])
      @test haskey(res, :policy)
      @test length(res.policy) == Game.policylength(G)

      game = Game.randommove!(G(), 4)
      model_ = Pack.unpack(Pack.pack(model), Model.AbstractModel)
      @test Model.apply(model, game) == Model.apply(model_, game)

      Base.Filesystem.mktemp() do path, io
        Model.save(path, model)
        model_ = Model.load(path, format = :jtm)
        @test Model.apply(model, game) == Model.apply(model_, game)
      end
    end

    @testset "CachingModel" begin
      G = Game.TicTacToe
      base =  Model.Zoo.MLP(G, [64, 32], :relu)
      model = Model.CachingModel(base)
      @test Model.gametype(model) == G
      @test Model.ntasks(model) == 1
      @test Model.isasync(model) == false
      @test Model.basemodel(model) == base
      @test Model.playingmodel(model) == model
      @test Model.trainingmodel(model) == nothing
      @test Target.targetnames(model) == [:value, :policy]
      @test Target.targets(model) == (
        value = Target.DefaultValueTarget(G),
        policy = Target.DefaultPolicyTarget(G),
      )

      res = Model.apply(model, G(), targets = [:value, :policy])
      @test haskey(res, :value)
      @test haskey(res, :policy)
      @test length(res.value) == 1
      @test length(res.policy) == Game.policylength(G)
      res = Model.apply(model, G(), targets = [:policy])
      @test haskey(res, :policy)
      @test length(res.policy) == Game.policylength(G)

      game = Game.randommove!(G(), 4)
      model_ = Pack.unpack(Pack.pack(model), Model.AbstractModel)
      @test Model.apply(model, game) == Model.apply(model_, game)

      Base.Filesystem.mktemp() do path, io
        Model.save(path, model)
        model_ = Model.load(path, format = :jtm)
        @test Model.apply(model, game) == Model.apply(model_, game)
      end
    end

    @testset "AssistedModel" begin
      G = Game.TicTacToe
      struct Dummy <: Model.AbstractModel{G} end
      Model.assist(d :: Dummy, game) = (value = 0.42, )
      base =  Model.Zoo.MLP(G, [64, 32], :relu)
      model = Model.AssistedModel(base, Dummy())
      @test Model.gametype(model) == G
      @test Model.ntasks(model) == 1
      @test Model.isasync(model) == false
      @test Model.basemodel(model) == base
      @test Model.playingmodel(model) == model
      @test Model.trainingmodel(model) == nothing
      @test Target.targetnames(model) == [:value, :policy]
      @test Target.targets(model) == (
        value = Target.DefaultValueTarget(G),
        policy = Target.DefaultPolicyTarget(G),
      )

      res = Model.apply(model, G(), targets = [:value, :policy])
      @test haskey(res, :value)
      @test haskey(res, :policy)
      @test length(res.value) == 1
      @test length(res.policy) == Game.policylength(G)
      res = Model.apply(model, G(), targets = [:policy])
      @test haskey(res, :policy)
      @test length(res.policy) == Game.policylength(G)

      game = Game.randommove!(G(), 4)
      model_ = Pack.unpack(Pack.pack(model), Model.AbstractModel)
      @test Model.apply(model, game) == Model.apply(model_, game)

      Base.Filesystem.mktemp() do path, io
        Model.save(path, model)
        model_ = Model.load(path, format = :jtm)
        @test Model.apply(model, game) == Model.apply(model_, game)
      end
    end
  end
end

@testset "Player" begin

  @testset "General" begin
    G = Game.TicTacToe
    model = Model.RolloutModel(G)
    for player in [
      Player.RandomPlayer(G),
      Player.IntuitionPlayer(model, temperature = 0.25),
      Player.MCTSPlayer(model, temperature = 0.25),
      Player.MCTSPlayerGumbel(model, temperature = 0.25),
    ]
      @test Player.name(player) isa String

      p = Player.think(player, G())
      @test p isa Vector{Float32}
      @test length(p) == Game.policylength(G)
      @test Player.decide(player, G()) isa Game.ActionIndex
      @test Player.decidechain(player, G()) isa Vector{Game.ActionIndex}
      @test Player.move!(G(), player) isa G
      @test Player.turn!(G(), player) isa G
      @test Player.ntasks(player) == Model.ntasks(player) == 1
      @test Player.gametype(player) == Model.gametype(player) == G

      bytes = Pack.pack(player)
      player2 = Pack.unpack(bytes, Player.AbstractPlayer)
      @test typeof(player) == typeof(player2)
    end
  end

  @testset "Ranking" begin
    G = Game.TicTacToe
    model = Model.RolloutModel(G)
    players = [Player.MCTSPlayer(model; power) for power in [10, 100, 500]]
    rk = Player.rank(players, 50)
    @test rk isa Player.Ranking
  end
end

@testset "Training" begin

  @testset "LabelData" begin
    ld = Training.LabelData()
    push!(ld, rand(Float32, 5), rand(Float32, 5))
    bytes = Pack.pack(ld)
    ld_ = Pack.unpack(bytes, Training.LabelData)
    @test all(v -> all(v[1] .== v[2]), zip(ld.data, ld_.data))
  end

  @testset "DataSet" begin
    G = Game.TicTacToe
    model = Model.RolloutModel(G)
    player = Player.MCTSPlayer(Model.RolloutModel(G), power = 100)
    ds0 = Training.DataSet(G, Target.targetnames(player), Target.targets(player))
    ds1 = Training.DataSet(model)
    @test Training.isconsistent(ds0)
    @test ds0 isa Training.DataSet{G}
    @test length(ds0) == length(ds1) == 0
    @test Training.targets(ds1) == Model.targets(model)
    @test Training.targetnames(ds1) == Model.targetnames(model)

    ds2 = Training.record(player, 5)
    @test ds2[1:3] isa Training.DataSet{G}
    @test length(ds2[1:3]) == 3
    dss = split(ds2, 4)
    @test all(length.(dss) .<= 4)

    ds3 = Training.record(player, 5, augment = false)
    ds4 = merge([ds1, ds2])
    @test length(ds1) + length(ds2) == length(ds4)
    deleteat!(ds4, 5)
    @test length(ds1) + length(ds2) == length(ds4) + 1
  end

  @testset "DataBatches" begin
    G = Game.TicTacToe
    player = Player.MCTSPlayer(Model.RolloutModel(G), power = 100)
    ds = Training.record(player, 10, targets = (dummy = Target.DummyTarget(G),))

    for T in [Array{Float32}, Array{Float64}]
      for cache in Training.DataBatches(T, ds, 50, partial = false)
        @test cache isa Training.DataCache
        @test length(cache) == 50
        @test cache.data isa T
        @test cache.target_labels isa Vector{T}
      end
    end
  end

  @testset "Loss" begin
    G = Game.TicTacToe
    reg = (l1 = Training.L1Reg(), l2 = Training.L2Reg())
    targets = (dummy = Target.DummyTarget(G), )
    model = Model.Zoo.ZeroConv(G; filters = 8, blocks = 2, targets)
    player = Player.MCTSPlayer(model, power = 50)
    ds = Training.record(player, 1)

    lc = Training.losscomponents(model, ds)
    lcvp = Training.losscomponents(model, ds, targets = (:value, :policy))
    lcreg = Training.losscomponents(model, ds; reg)

    lcreg2 = Training.losscomponents(model, ds; reg, weights = (; value = 2, l1 = 2))

    @test keys(lc) == (:value, :policy, :dummy)
    @test keys(lcvp) == (:value, :policy)
    @test keys(lcreg) == (:value, :policy, :dummy, :l1, :l2)
    @test length(lc) == 3
    @test length(lcvp) == 2
    @test lc.value == lcvp.value && lc.policy == lcvp.policy
    @test length(lcreg) == 5
    @test 2lcreg.value == lcreg2.value && 2lcreg.l1 == lcreg2.l1

    l = Training.loss(model, ds)
    lvp = Training.loss(model, ds, targets = (:value, :policy))
    lreg = Training.loss(model, ds; reg, batchsize = 1)

    lreg2 = Training.loss(model, ds; reg, weights = (; value = 2, l1 = 2))
    @test l ≈ sum(lc)
    @test lvp ≈ sum(lcvp)
    @test lreg ≈ sum(lcreg)
    @test lreg2 ≈ sum(lcreg2)
  end

  
  
end

