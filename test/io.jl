

@testset "IO" begin
  for (l, r) in [ (Model.Dense(20, 30, f = relu), rand(Float32, 20, 2))
                , (Model.Pointwise(f = tanh), rand(Float32, 15, 24, 2, 5))
                , (Model.Conv(3, 5), rand(Float32, 7, 7, 3, 12))
                , (Model.Deconv(5, 3), rand(Float32, 8, 8, 5, 4))
                , (Model.Pool(), rand(Float32, 5, 5, 5, 5))
                , (Model.Dropout(0.7, f = elu), rand(Float32, 7,7,7,7))
                , (Model.Batchnorm(10), rand(Float32, 10, 5)) ]

    ld = l |> Model.decompose |> Model.compose
    @test typeof(l) == typeof(ld)
    @test all(l(r) .== ld(r))

  end

  for G in [Game.TicTacToe, Game.MetaTac]
    feats = [Model.ConstantFeature([0.7, 0.5, 1.]), Model.ConstantFeature([-5.])]
    chain = Model.@chain G Conv(10) Deconv(7) Batchnorm() Pool() Dense(25) Dropout(0.7)
    model = Model.NeuralModel(G, chain, feats)
    name  = tempname()

    Model.save(name, model)
    model2 = Model.load(name)
    @test typeof(model) == typeof(model2)

    game = Game.random_turns!(G(), 3)
    @test model(game) == model2(game)

    player = Player.MCTSPlayer(model, power = 10)
    dataset = Data.record(player, 1, augment = true)

    name = tempname()
    Data.save(name, dataset)
    ds = Data.load(name)
    @test all(Game.array(ds.games) .== Game.array(dataset.games))
    @test all(ds.label .== dataset.label)
    @test all(ds.flabel .== dataset.flabel)
  end
end
