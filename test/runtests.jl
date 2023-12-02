using Jtac, Test, Random

Random.seed!(42)

function packcycle(value, T = typeof(value), isequal = isequal)
  bytes = Pack.pack(value)
  uvalue = Pack.unpack(bytes, T)
  isequal(value, uvalue) &&
  all(bytes .== Pack.pack(uvalue))
end

@testset "Pack" include("pack.jl")
@testset "Game"  include("game.jl")
@testset "Target" include("target.jl")
@testset "Model" include("model.jl")
@testset "Player" include("player.jl")
@testset "Training" include("training.jl")

try
  using Flux
  @testset "Flux" include("fluxext.jl")
catch
  println("Flux not installed, skipping tests")
end