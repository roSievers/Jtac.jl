
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

