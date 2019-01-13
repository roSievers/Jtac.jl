

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
  @test size(chain(ones(Float32, 11, 10, 5, 2))) == (15, 2)
end
