
module FluxExt

import Flux

using Jtac
import Jtac.Game: AbstractGame
import Jtac.Model: Activation,
                   Backend,
                   DefaultBackend,
                   Layer,
                   PrimitiveLayer,
                   CompositeLayer,
                   NeuralModel

"""
Neural layer backend supported by the julia package Flux.jl.
"""
struct FluxBackend{T} <: Backend{T} end

adapt(T, :: FluxBackend) = FluxBackend{T}()

function Base.show(io :: IO, :: FluxBackend{T}) where {T}
  print(io, "FluxBackend{$T}()")
end


#
# TODO: There is a flux bug in Flux.outputsize(trunk, size(data))
# on the gpu - help fix this!
#
# function Model.isvalidinputsize(l :: Layer{<: FluxBackend}, insize)
#   try Flux.outputsize(unwrap(l), insize, padbatch = true)
#   catch _err
#     return false
#   end
#   true
# end
Model.isvalidinputsize(l :: Layer{<: FluxBackend}, insize) = true

function Model.outputsize(l :: Layer{<: FluxBackend}, insize)
  @assert Model.isvalidinputsize(l, insize)
  Flux.outputsize(unwrap(l), insize, padbatch = true)[1:end-1]
end

##
## FluxBackend layer implementations
##

const FluxDense = Flux.Chain{Tuple{typeof(Flux.flatten), D}} where {D <: Flux.Dense}
const FluxResidual = Flux.Chain{Tuple{S, A}} where {S <: Flux.SkipConnection, A<: Activation}

wrap(T, layer :: FluxDense) = Dense{T}(layer)
wrap(T, layer :: Flux.Conv) = Conv{T}(layer)
wrap(T, layer :: Flux.BatchNorm) = Batchnorm{T}(layer)
wrap(T, layer :: Flux.Chain) = Chain{T}(layer)
wrap(T, layer :: FluxResidual) = Residual{T}(layer)

unwrap(l :: Layer{FluxBackend{T}}) where {T} = l.layer

(l :: Layer{F})(x) where {F <: FluxBackend} = unwrap(l)(x)


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
  pad = d.layer.pad
  stride = d.layer.stride
  Model.Conv{T}(w, b, d.layer.σ, !(d.layer.bias == false), pad, stride)
end


struct Batchnorm{T} <: PrimitiveLayer{FluxBackend{T}}
  layer :: Flux.BatchNorm
end

function Model.adapt( :: FluxBackend{T}
                    , b :: Model.Batchnorm
                    ) where {F, T <: AbstractArray{F}}
  eps = F(1e-5)
  momentum = F(0.1)
  affine = true
  track_stats = true
  active = nothing
  chs = length(b.bias)

  layer = Flux.BatchNorm(
    b.f.f,
    convert(T, b.bias),
    convert(T, b.scale),
    convert(T, b.mean),
    convert(T, b.var),
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
    convert(T, b.layer.μ),
    convert(T, b.layer.σ²),
    convert(T, b.layer.β),
    convert(T, b.layer.γ),
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
## Training
##

function Model.trainingmodel( model :: NeuralModel{G, B}
                            ) where {G <: AbstractGame, B <: FluxBackend}
  model
end

"""
Flux optimizer setup that can be used to continue training with seamless
optimizer settings.
"""
struct FluxOptimizerSetup
  setup
end

function Base.show(io :: IO, ::MIME"text/plain", s :: FluxOptimizerSetup)
  print(io, "FluxOptimizerSetup()")
end


"""
Helper model struct that collects the trunk, heads, and activations of a
[`NeuralModel`](@ref) after targets have been fixed in a way that is more
compatible with Flux and Zygote (e.g., by using tuples).
"""
struct FluxProxyModel
  trunk
  heads
  activations
end

Flux.@functor FluxProxyModel

function FluxProxyModel(model :: Model.NeuralModel, ctx)
  heads = []
  activations = []

  for name in ctx.target_names
    index = findfirst(isequal(name), model.target_names)
    push!(heads, model.target_heads[index].layer)
    push!(activations, model.target_activations[index])
  end

  FluxProxyModel(model.trunk.layer, Tuple(heads), Tuple(activations))
end

function Training.setup( model :: NeuralModel{G, <: FluxBackend}
                       , ctx :: Training.LossContext
                       , opt ) where {G <: AbstractGame}
  if opt isa FluxOptimizerSetup
    opt
  else
    proxy = FluxProxyModel(model, ctx)
    FluxOptimizerSetup(Flux.setup(opt, proxy))
  end
end

Training.defaultoptimizer(:: FluxBackend) = Flux.Momentum(0.01, 0.9)

function Training.step!( model :: Model.NeuralModel{G, <: FluxBackend}
                       , cache
                       , ctx
                       , setup ) where {G <: AbstractGame}

  proxy = FluxProxyModel(model, ctx)
  labels = Tuple(cache.target_labels)
  weights = Tuple(ctx.target_weights)
  losses = Tuple(ctx.target_lossfunctions)

  loss, grads = Flux.withgradient(proxy) do m
    tloss = Training._loss(
      m.trunk,
      m.heads,
      m.activations,
      cache.data,
      labels,
      weights,
      losses,
    )
    # rloss = Training._regloss(Flux.params(m), ctx)
    rloss = 0f0
    tloss + rloss
  end

  Flux.update!(setup.setup, proxy, grads[1])
  loss
end


##
## Register Array-based Flux backends
##
  
function __init__()
  Util.register!(Backend, FluxBackend{Array{Float32}}(), :flux, :flux32)
  Util.register!(Backend, FluxBackend{Array{Float64}}(), :flux64)
end

end
