
using Jtac
using .Game, .Model, .Data, .Player, .Target

using Test

@testset "Target" begin
  G = TicTacToe
  opt_targets = [DummyTarget(G, [1, 2, 3])]
  model = NeuralModel(G, @chain G Conv(64, "relu", padding = 1) Dense(64, "relu"); opt_targets)

  @test all(Target.name.(targets(model)) .== [:value, :policy, :dummy])
  res = apply(model, G(), true)
  @test all(keys(res) .== [:value, :policy, :dummy])

  for target in [Target.targets(model)..., L1Reg(), L2Reg()]
    p = Pack.pack(target)
    t = Pack.unpack(p, AbstractTarget)
    @test typeof(t) == typeof(target)
  end

  ds = DataSet(model)
  @test Data.check_consistency(ds)
  @test all(map(Target.compatible, Target.targets(ds), Target.targets(model)))

  ds2 = Target.adapt(ds, [PolicyTarget(G), ValueTarget(G)])
  @test all(Target.name.(ds2.targets) .== [:policy, :value])

  player = MCTSPlayer(model, power = 1000)
  ds = Player.record(player, 100)
  @test all(map(Target.compatible, Target.targets(ds), Target.targets(player)))

  l = Training.loss(model, ds)
  @test all(collect(keys(l)) .== [:value, :policy, :dummy])
  l = Training.loss(model, ds, reg_targets = [L1Reg(), L2Reg()], weights = (;l2reg = 1e-4))
  @test all(collect(keys(l)) .== [:value, :policy, :dummy, :l1reg, :l2reg])

end

