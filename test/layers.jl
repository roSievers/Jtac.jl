

@testset "Layers" begin
  @test Jtac.atype(false) == Array{Float32}
  @test Jtac.atype(true)  == Knet.KnetArray{Float32}
  p = Knet.Param([1,2])
  p.opt = Knet.Adam(lr = 1e-3)
  p2 = Jtac.copy_param(p)
  @test Jtac.is_param(p) == Jtac.is_param(p2) == true
  @test !Jtac.is_param([1,2])
  @test p2.opt == p.opt
  Conv(50, 10)
  chain = @chain (11, 10, 5) Conv(10) Pool() Dropout(0.5) Dense(15) Batchnorm()
  stack1 = @stack (11, 10, 5) Conv(10) Pool() Dropout(0.5) Dense(15) Batchnorm()
  stack2 = @stack (11, 10, 5) stack_input=true Conv(10) Pool() Dropout(0.5) Dense(15) Batchnorm()
  @test size(chain(ones(Float32, 11, 10, 5, 2))) == (15, 2)
  @test size(stack1(ones(Float32, 11, 10, 5, 2))) == (1070, 2)
  @test size(stack2(ones(Float32, 11, 10, 5, 2))) == (1620, 2)
end

@testset "Models" begin
  for G in [TicTacToe, MetaTac]
    game = G()
    @test Shallow(G)(game)         |> length == policy_length(G) + 1
    @test ShallowConv(G, 20)(game) |> length == policy_length(G) + 1
    @test MLP(G, [200, 50])(game)  |> length == policy_length(G) + 1
  end
  chain = @chain MetaTac Conv(10) Pool() Dropout(0.5) Dense(15) Batchnorm()
  stack = @stack MetaTac Conv(10) Pool() Dropout(0.5) Dense(15) Batchnorm()
  @test NeuralModel(MetaTac, chain)(MetaTac()) |> length == 82
  @test NeuralModel(MetaTac, stack)(MetaTac()) |> length == 82
end
