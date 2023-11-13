
using Jtac
import Jtac.Model: Activation, Backend, DefaultBackend, Layer, PrimitiveLayer, CompositeLayer

using Test
import Flux

struct FluxBackend{T} <: Backend{T} end

function Base.show(io :: IO, :: FluxBackend{T}) where {T}
  print(io, "FluxBackend{$T}()")
end

adapt(T, :: FluxBackend) = FluxBackend{T}()


const FluxDense = Flux.Chain{Tuple{typeof(Flux.flatten), D}} where {D <: Flux.Dense}
const FluxResidual = Flux.Chain{Tuple{S, A}} where {S <: Flux.SkipConnection, A<: Activation}

wrap(T, layer :: FluxDense) = Dense{T}(layer)
wrap(T, layer :: Flux.Conv) = Conv{T}(layer)
wrap(T, layer :: Flux.BatchNorm) = Batchnorm{T}(layer)
wrap(T, layer :: Flux.Chain) = Chain{T}(layer)
wrap(T, layer :: FluxResidual) = Residual{T}(layer)

unwrap(l :: Layer{FluxBackend{T}}) where {T} = l.layer

(l :: Layer{F})(x) where {F <: FluxBackend} = unwrap(l)(x)

function Model.isvalidinputsize(l :: Layer{<: FluxBackend}, insize)
  @show insize
  try Flux.outputsize(unwrap(l), insize, padbatch = true)
  catch _err
    return false
  end
  true
end

function Model.outputsize(l :: Layer{<: FluxBackend}, insize)
  @assert Model.isvalidinputsize(l, insize)
  Flux.outputsize(unwrap(l), insize, padbatch = true)[1:end-1]
end

##
## FluxBackend layers
##

struct Dense{T} <: PrimitiveLayer{FluxBackend{T}}
  layer :: FluxDense
end

function Model.adapt(:: FluxBackend{T}, d :: Model.Dense) where {T}
  w = convert(T, d.w)
  b = d.bias ? convert(T, d.b) : false
  layer = Flux.Dense(w, b, d.f.f)
  layer = Flux.Chain(Flux.flatten, layer)
  Dense{T}(layer)
end

function Model.adapt(:: DefaultBackend{T}, d :: Dense) where {T}
  layer = d.layer.layers[2] # Flux.Dense layer
  w = convert(T, layer.weight)
  b = layer.bias == false ? zeros(size(w, 1)) : layer.bias
  b = convert(T, b)
  Model.Dense{T}(w, b, layer.σ, !(layer.bias == false))
end


struct Conv{T} <: PrimitiveLayer{FluxBackend{T}}
  layer :: Flux.Conv
end

function Model.adapt(:: FluxBackend{T}, c :: Model.Conv) where {T}
  w = convert(T, c.w)
  b = c.bias ? convert(T, c.b) : false
  layer = Flux.Conv(w, b, c.f.f; stride = c.s, pad = c.p)
  Conv{T}(layer)
end

function Model.adapt(:: DefaultBackend{T}, d :: Conv) where {T}
  w = convert(T, d.layer.weight)
  b = d.layer.bias == false ? zeros(size(w, 4)) : d.layer.bias
  b = convert(T, b)
  padding = d.layer.pad
  stride = d.layer.stride
  Model.Conv{T}(w, b, d.layer.σ, !(d.layer.bias == false), padding, stride)
end

struct Batchnorm{T} <: PrimitiveLayer{FluxBackend{T}}
  layer :: Flux.BatchNorm
end

function Model.adapt(:: FluxBackend{T}, b :: Model.Batchnorm) where {F, T <: AbstractArray{F}}
  eps = F(1e-5)
  momentum = F(0.1)
  affine = true
  track_stats = true
  active = nothing
  chs = length(b.bias)

  layer = Flux.BatchNorm(
    b.f.f,
    b.bias,
    b.scale,
    b.mean,
    b.var,
    eps,
    momentum,
    affine,
    track_stats,
    active,
    chs,
  )
  Flux.testmode!(layer)
  Batchnorm{T}(layer)
end

function Model.adapt(:: DefaultBackend{T}, b :: Batchnorm) where {T}
  Model.Batchnorm{T}(
    b.layer.μ,
    b.layer.σ²,
    b.layer.β,
    b.layer.γ,
    b.layer.λ,
  )
end


struct Chain{T} <: CompositeLayer{FluxBackend{T}}
  layer :: Flux.Chain
end

function Model.adapt(backend :: FluxBackend{T}, c :: Model.Chain) where {T}
  fluxlayers = map(c.layers) do layer
    unwrap(Model.adapt(backend, layer))
  end
  Chain{T}(Flux.Chain(fluxlayers...))
end

function Model.adapt(backend :: DefaultBackend{T}, c :: Chain) where {T}
  layers = map(c.layer.layers) do layer
    Model.adapt(backend, wrap(T, layer))
  end
  Model.Chain(collect(layers))
end


struct Residual{T} <: CompositeLayer{FluxBackend{T}}
  layer :: FluxResidual
end

function Model.adapt(backend :: FluxBackend{T}, r :: Model.Residual) where {T}
  chain = unwrap(Model.adapt(backend, r.chain))
  layer = Flux.Chain(Flux.SkipConnection(chain, +), r.f)
  Residual{T}(layer)
end

function Model.adapt(backend :: DefaultBackend{T}, r :: Residual) where {T}
  chain = Model.adapt(backend, wrap(T, r.layer.layers[1].layers))
  Model.Residual(chain, r.layer.layers[2])
end

##
## NeuralModel wrapper
##

##
## Tests
##

@testset "Dense" begin
  for F in [Float32, Float64]
    for l in [
      Model.Dense(5, 7, :relu),
      Model.Dense(5, 7, sin, bias = false),
    ]
      l = Model.adapt(DefaultBackend{Array{F}}(), l)
      lflux = Model.adapt(FluxBackend{Array{F}}(), l)
      l2 = Model.adapt(DefaultBackend{Array{F}}(), lflux)

      @test Model.getbackend(l) isa Model.DefaultBackend{Array{F}}
      @test Model.getbackend(l2) isa Model.DefaultBackend{Array{F}}
      @test Model.getbackend(lflux) isa FluxBackend{Array{F}}

      @test l isa Model.Dense{Array{F}}
      @test l2 isa Model.Dense{Array{F}}
      @test lflux isa Dense{Array{F}}

      data = rand(F, 5, 10)
      r1 = l(data)
      r2 = l2(data)
      r3 = lflux(data)

      @test all(r1 .== r2 .== r3)
    end
  end
end

@testset "Conv" begin
  for F in [Float32, Float64]
    for l in [
      Model.Conv(5, 7, :relu, window = 3, padding = 1),
      Model.Conv(5, 7, sin, window = 3, padding = 2, bias = false),
    ]
      l = Model.adapt(DefaultBackend{Array{F}}(), l)
      lflux = Model.adapt(FluxBackend{Array{F}}(), l)
      l2 = Model.adapt(DefaultBackend{Array{F}}(), lflux)

      @test Model.getbackend(l) isa Model.DefaultBackend{Array{F}}
      @test Model.getbackend(l2) isa Model.DefaultBackend{Array{F}}
      @test Model.getbackend(lflux) isa FluxBackend{Array{F}}

      @test l isa Model.Conv{Array{F}}
      @test l2 isa Model.Conv{Array{F}}
      @test lflux isa Conv{Array{F}}

      data = rand(F, 3, 6, 5, 10)
      r1 = l(data)
      r2 = l2(data)
      r3 = lflux(data)

      @test all(r1 .== r2 .== r3)
    end
  end
end

@testset "Batchnorm" begin
  for F in [Float32, Float64]
    for l in [
      Model.Batchnorm(3, :relu),
      Model.Batchnorm(3, sin),
    ]
      l = Model.adapt(DefaultBackend{Array{F}}(), l)
      lflux = Model.adapt(FluxBackend{Array{F}}(), l)
      l2 = Model.adapt(DefaultBackend{Array{F}}(), lflux)

      @test Model.getbackend(l) isa Model.DefaultBackend{Array{F}}
      @test Model.getbackend(l2) isa Model.DefaultBackend{Array{F}}
      @test Model.getbackend(lflux) isa FluxBackend{Array{F}}

      @test l isa Model.Batchnorm{Array{F}}
      @test l2 isa Model.Batchnorm{Array{F}}
      @test lflux isa Batchnorm{Array{F}}

      data = rand(F, 2, 2, 3, 1)
      r1 = l(data)
      r2 = l2(data)
      r3 = lflux(data)

      @test all(isapprox(r1, r2) && isapprox(r2, r3))
    end
  end
end

@testset "Chain" begin
  for F in [Float32, Float64]
    chain = Model.@chain (5, 5, 3) Conv(10) Batchnorm(:relu) Dense(42)
    for l in [ chain ]
      l = Model.adapt(DefaultBackend{Array{F}}(), l)
      lflux = Model.adapt(FluxBackend{Array{F}}(), l)
      l2 = Model.adapt(DefaultBackend{Array{F}}(), lflux)

      @test Model.getbackend(l) isa Model.DefaultBackend{Array{F}}
      @test Model.getbackend(l2) isa Model.DefaultBackend{Array{F}}
      @test Model.getbackend(lflux) isa FluxBackend{Array{F}}

      @test l isa Model.Chain{Array{F}}
      @test l2 isa Model.Chain{Array{F}}
      @test lflux isa Chain{Array{F}}

      data = rand(F, 5, 5, 3, 1)
      r1 = l(data)
      r2 = l2(data)
      r3 = lflux(data)

      @test all(isapprox(r1, r2) && isapprox(r2, r3))
    end
  end
end

@testset "Residual" begin
  for F in [Float32, Float64]
    res = Model.@residual (5, 5, 3) Conv(3, padding = 1) Batchnorm(:relu) Conv(3, padding = 1)
    for l in [ res ]
      l = Model.adapt(DefaultBackend{Array{F}}(), l)
      lflux = Model.adapt(FluxBackend{Array{F}}(), l)
      l2 = Model.adapt(DefaultBackend{Array{F}}(), lflux)

      @test Model.getbackend(l) isa Model.DefaultBackend{Array{F}}
      @test Model.getbackend(l2) isa Model.DefaultBackend{Array{F}}
      @test Model.getbackend(lflux) isa FluxBackend{Array{F}}

      @test l isa Model.Residual{Array{F}}
      @test l2 isa Model.Residual{Array{F}}
      @test lflux isa Residual{Array{F}}

      data = rand(F, 5, 5, 3, 1)
      r1 = l(data)
      r2 = l2(data)
      r3 = lflux(data)

      @test all(isapprox(r1, r2) && isapprox(r2, r3))
    end
  end
end

@testset "Model" begin
  for F in [Float32, Float64]
    G = Game.TicTacToe
    model = Model.Zoo.ZeroRes(G, filters = 8, blocks = 3)
    for l in [model.trunk]
      l = Model.adapt(DefaultBackend{Array{F}}(), l)
      lflux = Model.adapt(FluxBackend{Array{F}}(), l)
      l2 = Model.adapt(DefaultBackend{Array{F}}(), lflux)

      @test Model.getbackend(l) isa Model.DefaultBackend{Array{F}}
      @test Model.getbackend(l2) isa Model.DefaultBackend{Array{F}}
      @test Model.getbackend(lflux) isa FluxBackend{Array{F}}

      games = [Game.randominstance(G) for _ in 1:10]
      data = convert(Array{F}, Game.array(games))

      r1 = l(data)
      r2 = l2(data)
      r3 = lflux(data)

      @test all(isapprox(r1, r2) && isapprox(r2, r3))
    end
  end
end

G = Game.TicTacToe
model = Model.Zoo.ZeroRes(G, filters = 8, blocks = 3)
fluxmodel = Model.adapt(FluxBackend{Array{Float32}}(), model)
