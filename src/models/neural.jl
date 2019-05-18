
# Neural Model
# Wrapper around a model that generates 1 + policy_length "logit" values

struct NeuralModel{G, GPU} <: Model{G, GPU}
  logits :: Layer{GPU}  # Takes input and returns "logits"
  vconv                   # Converts value-logit to value
  pconv                   # Converts policy-logits to policy
end

function NeuralModel(:: Type{G}, features :: Layer{GPU}; 
                   vconv = Knet.tanh, pconv = Knet.softmax) where {G, GPU}

  @assert valid_insize(features, size(G)) "Input layer does not fit the game"

  logits = @chain G features Dense(policy_length(G)+1, gpu = GPU)

  NeuralModel{G, GPU}(logits, vconv, pconv)
end

function (m :: NeuralModel{G, GPU})(games :: Vector{G}) where {G, GPU}
  at = atype(GPU)
  data = convert(at, representation(games))
  result = m.logits(data)
  vcat(m.vconv.(result[1:1,:]), m.pconv(result[2:end,:], dims = 1))
end

function (m :: NeuralModel{G, GPU})(game :: G) where {G, GPU}
  reshape(m([game]), (policy_length(game) + 1,))
end

swap(m :: NeuralModel{G, GPU}) where {G, GPU} = 
  NeuralModel{G, !GPU}(swap(m.logits), m.vconv, m.pconv)

Base.copy(m :: NeuralModel{G, GPU}) where {G, GPU} =
  NeuralModel{G, GPU}(copy(m.logits), m.vconv, m.pconv)


# Some simple NeuralModels

# Linear Model

Shallow(:: Type{G}; kwargs...) where {G} = NeuralModel(G, Pointwise(); kwargs...)

# Multilayer perception

function MLP(:: Type{G}, hidden, f = Knet.relu; kwargs...) where {G}
  widths = [ prod(size(G)), hidden..., policy_length(G) + 1 ]
  layers = [ Dense(widths[j], widths[j+1], f) for j in 1:length(widths) - 2 ]
  push!(layers, Dense(widths[end-1], widths[end], identity))
  NeuralModel(G, Chain(layers...); kwargs...)
end

# Shallow convolutional network

function ShallowConv(:: Type{G}, filters, f = Knet.relu; kwargs...) where {G}
  NeuralModel(G, Conv(size(G, 3), filters, f); kwargs...)
end
