

@testset "IO" begin
  for (l, r) in [ (Dense(20, 30, relu), rand(Float32, 20, 2))
                , (Pointwise(tanh), rand(Float32, 15, 24, 2, 5))
                , (Conv(3, 5), rand(Float32, 7, 7, 3, 12))
                , (Deconv(5, 3), rand(Float32, 8, 8, 5, 4))
                , (Pool(), rand(Float32, 5, 5, 5, 5))
                , (Dropout(0.7, elu), rand(Float32, 7,7,7,7))
                , (Batchnorm(10), rand(Float32, 10, 5)) ]

    ld = l |> Jtac.decompose |> Jtac.compose
    @test typeof(l) == typeof(ld)
    @test all(l(r) .== ld(r))

  end

  for G in [TicTacToe, MetaTac]

    feats = [ConstantFeature([0.7, 0.5, 1.]), ConstantFeature([-5.])]
    chain = @chain G Conv(10) Deconv(7) Batchnorm() Pool() Dense(25) Dropout(0.7)
    model = NeuralModel(G, chain, feats)
    name  = tempname()

    save_model(name, model)
    model2 = load_model(name)
    @test typeof(model) == typeof(model2)

    game = random_turn!(G())
    @test model(game) == model2(game)

    player = MCTSPlayer(model, power = 10)
    dataset = record_self(player, 1, augment = true)
    
    name = tempname()
    save_dataset(name, dataset)
    ds = load_dataset(name)
    @test all(representation(ds.games) .== representation(dataset.games))
    @test all(ds.label .== dataset.label)
    @test all(ds.flabel .== dataset.flabel)

  end

end
