
# Base Model
# Wrapper around a model that generates 1 + policy_length "logit" values

struct BaseModel{G, GPU} <: Model{G, GPU}
  logits :: Layer{GPU}  # Takes input and returns "logits"
  vconv                 # Converts value-logit to value
  pconv                 # Converts policy-logits to policy
end

function BaseModel(:: Type{G}, logits :: Layer{GPU}; 
                   vconv = tanh, pconv = softmax) where {G, GPU}
  BaseModel{G, GPU}(logits, vconv, pconv)
end

function (m :: BaseModel{G, GPU})(games :: Vector{G}) where {G, GPU}
  at = atype(GPU)
  data = convert(at, representation(games))
  result = m.logits(data)
  vcat(m.vconv.(result[1:1,:]), m.pconv(result[2:end,:], dims = 1))
end

function (m :: BaseModel{G, GPU})(game :: G) where {G, GPU}
  reshape(m([game]), (policy_length(game) + 1,))
end

swap(m :: BaseModel{G, GPU}) where {G, GPU} = 
  BaseModel{G, !GPU}(swap(m.logits), m.vconv, m.pconv)


Base.copy(m :: BaseModel{G, GPU}) where {G, GPU} =
  BaseModel{G, GPU}(copy(m.logits), m.vconv, m.pconv)


# Some simple BaseModels

# Linear Model

function Shallow(:: Type{G}; kwargs...) where {G}
  logits = Dense(prod(size(G)), policy_length(G) + 1, identity)
  BaseModel(G, logits; kwargs...)
end

# Multilayer perception

function MLP(:: Type{G}, hidden, f = relu; kwargs...) where {G}
  widths = [ prod(size(G)), hidden..., policy_length(G) + 1 ]
  layers = [ Dense(widths[j], widths[j+1], f) for j in 1:length(widths) - 1 ]
  BaseModel(G, Chain(layers...); kwargs...)
end

# Shallow convolutional network

function ShallowConv(:: Type{G}, channels, f = relu; kwargs...) where {G}
  logits = Chain(
    Conv(size(G, 3), channels, f),
    Dense(channels, policy_length(G) + 1, identity)
  )
  BaseModel(G, logits; kwargs...)
end
