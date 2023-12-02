
@testset "Activation" begin
  data = rand(12, 9)

  for (name, a) in Jtac.Util.lookup(Model.Activation)
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
      Model.Zoo.MLP(G, :relu, widths = [64, 32]),
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
    base = Model.Zoo.MLP(G, :relu, widths = [64, 32])
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
    base = Model.Zoo.MLP(G, :relu, widths = [64, 32])
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
    base = Model.Zoo.MLP(G, :relu, widths = [64, 32])
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

