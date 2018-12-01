
# Base Model
# Wrapper around a model that generates 1 + policy_length "logit" values

struct BaseModel{GPU, G} <: Model{GPU, G}
  logits :: Layer{GPU}  # Takes input and returns "logits"
  vconv                 # Converts value-logit to value
  pconv                 # Converts policy-logits to policy
end

function BaseModel(:: Type{G}, logits :: Layer{GPU}; 
                   vconv = tanh, pconv = softmax) where {GPU, G}
  BaseModel{GPU, G}(logits, vconv, pconv)
end

function (m :: BaseModel{GPU, G})(games :: Vector{G}) where {GPU, G}
  at = atype(GPU)
  data = convert(at, representation(games))
  result = m.logits(data)
  vcat(m.vconv.(result[1:1,:]), m.pconv(result[2:end,:], dims = 1))
end

function (m :: BaseModel{GPU, G})(game :: G) where {GPU, G}
  reshape(m([game]), (policy_length(game) + 1,))
end

swap(m :: BaseModel{GPU, G}) where {GPU, G} = 
  BaseModel{!GPU, G}(swap(m.logits), m.vconv, m.pconv)


Base.copy(m :: BaseModel{GPU, G}) where {GPU, G} =
  BaseModel{GPU, G}(copy(m.logits), m.vconv, m.pconv)


# Some simple BaseModels

# Linear Model

function LinearModel(:: Type{G}; kwargs...) where {G}
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

function SimpleConv(:: Type{G}, channels, f = relu; kwargs...) where {G}
  logits = Chain(
    Conv(size(G, 3), channels, f),
    Dense(channels, policy_length(G) + 1, identity)
  )
  BaseModel(G, logits; kwargs...)
end
