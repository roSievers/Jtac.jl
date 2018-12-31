

@testset Layers begin
  @test atype(false) == Array{Float32}
  p = Param([1,2])
  p.opt = Adam(lr = 1e-3)
  p2 = copy_param(p)
  @test is_param(p) == is_param(p2) == true
  @test !is_param([1,2])
  @test p2.opt == p.opt
  Conv(50, 10)
  chain = @chain (11, 10, 5) Conv(10) Pool() Dropout(0.5) Dense(15) Batchnorm()
  @test size(chain(ones(Float32, 11, 10, 5, 2))) == (15, 2)
end
