

@testset "Layers" begin
  @test Model.atype(false) == Array{Float32}
  @test Model.atype(true)  == Knet.KnetArray{Float32}
  p = Knet.Param([1,2])
  p.opt = Knet.Adam(lr = 1e-3)
  p2 = Model.copy_param(p)
  @test Model.is_param(p) == Model.is_param(p2) == true
  @test !Model.is_param([1,2])
  @test p2.opt == p.opt
  chain = Model.@chain (11, 10, 5) Conv(10) Pool() Dropout(0.5) Dense(15) Batchnorm()
  stack1 = Model.@stack (11, 10, 5) Conv(10) Pool() Dropout(0.5) Dense(15) Batchnorm()
  stack2 = Model.@stack (11, 10, 5) stack_input=true Conv(10) Pool() Dropout(0.5) Dense(15) Batchnorm()
  resnet = Model.@residual (9, 9, 5) Conv(5, relu, padding = 1) Conv(5, relu, padding = 1)
  @test size(chain(ones(Float32, 11, 10, 5, 2))) == (15, 2)
  @test size(stack1(ones(Float32, 11, 10, 5, 2))) == (1070, 2)
  @test size(stack2(ones(Float32, 11, 10, 5, 2))) == (1620, 2)
  @test size(resnet(ones(Float32, 9, 9, 5, 3))) == (9, 9, 5, 3)
  @test_throws ErrorException Model.@residual (9, 9, 5) Dense(10, relu)
end

@testset "Models" begin
  for G in [Game.TicTacToe, Game.MetaTac]
    game = G()
    for model in [Model.Shallow(G), Model.ShallowConv(G, 20), Model.MLP(G, [200, 50])]
      value, policy = model(game)
      @test length(value) == 1
      @test length(policy) == Model.policy_length(G)
    end
  end
  for G in [Game.MetaTac]
    chain = Model.@chain G Conv(10) Pool() Dropout(0.5) Dense(15) Batchnorm()
    stack = Model.@stack G Conv(10) Pool() Dropout(0.5) Dense(15) Batchnorm()
    for model in [Model.NeuralModel(G, chain), Model.NeuralModel(G, stack)]
      value, policy = model(G())
      @test length(value) == 1
      @test length(policy) == Model.policy_length(G)
    end
  end
end
