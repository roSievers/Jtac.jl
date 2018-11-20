
# Base Model
# Wrapper around a model that generates 1 + policy_length "logit" values

struct BaseModel{GPU} <: Model{GPU}
  logits :: Layer{GPU}  # Takes input and returns "logits"
  vconv                 # Converts value-logit to value
  pconv                 # Converts policy-logits to policy
end

function BaseModel(logits :: Layer{GPU}; vconv = tanh, pconv = softmax) where {GPU}
  BaseModel{GPU}(logits, vconv, pconv)
end

function (m :: BaseModel{GPU})(games :: Vector{G}) where {GPU, G <: Game}
  at = atype(GPU)
  data = convert(at, representation(games))
  result = m.logits(data)
  vcat(m.vconv.(result[1:1,:]), m.pconv(result[2:end,:], dims = 1))
end

(m :: BaseModel)(game :: Game) = reshape(m([game]), (policy_length(game) + 1,))

swap(m :: BaseModel{GPU}) where {GPU} = BaseModel{!GPU}(swap(m.logits), m.vconv, m.pconv)
Base.copy(m :: BaseModel{GPU}) where {GPU} = BaseModel{GPU}(copy(m.logits), m.vconv, m.pconv)


# Some simple BaseModels

# Linear Model

function LinearModel(game :: Game; kwargs...)
  logits = Dense(prod(size(game)), policy_length(game) + 1, identity)
  BaseModel(logits; kwargs...)
end

# Multilayer perception

function MLP(game :: Game, hidden, f = relu; kwargs...)
  widths = [ prod(size(game)), hidden..., policy_length(game) + 1 ]
  layers = [ Dense(widths[j], widths[j+1], f) for j in 1:length(widths) - 1 ]
  BaseModel(Chain(layers...); kwargs...)
end

# Shallow convolutional network

function SimpleConv(game :: Game, channels, f = relu; kwargs...)
  logits = Chain(
    Conv(size(game, 3), channels, f),
    Dense(channels, policy_length(game) + 1, identity)
  )
  BaseModel(logits; kwargs...)
end
