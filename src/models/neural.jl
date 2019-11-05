
# -------- Neural Model ------------------------------------------------------ #

"""
Trainable model that uses a neural network to generate the value and policy
for a game state. Optionally, the network can also predict features by
applying a dedicated dense layer on the representation used for value and
policy prediction.
"""
struct NeuralModel{G, GPU} <: Model{G, GPU}

  trunk :: Layer{GPU}           # Takes input and returns layer before logits
  features :: Vector{Feature}   # Features that the network must predict

  vhead :: Layer{GPU}                  # value head
  phead :: Layer{GPU}                  # policy head
  fhead :: Union{Nothing, Layer{GPU}}  # feature head

  vconv                         # Converts value-logit to value
  pconv                         # Converts policy-logits to policy

end

"""
    NeuralModel(G, trunk [, features; vhead, phead, fhead, vconv, pconv])

Construct a model for gametype `G` based on the neural network `trunk`,
optionally with `features` enabled. The heads `vhead`, `phead`, and `fhead` are
optional neural network layers that produce "logits" for the value, policy and
feature prediction. The functions `vconv` and `pconv` are used to map the
"logits" to values and policies. The respective feature converters are contained
in `features`.
"""
function NeuralModel( :: Type{G}
                    , trunk :: Layer{GPU}
                    , features = Feature[]
                    ; vhead :: Union{Nothing, Layer{GPU}} = nothing
                    , phead :: Union{Nothing, Layer{GPU}} = nothing
                    , fhead :: Union{Nothing, Layer{GPU}} = nothing
                    , vconv = Knet.tanh
                    , pconv = Knet.softmax
                    ) where {G, GPU}

  @assert valid_insize(trunk, size(G)) "Trunk incompatible with $G"

  features = check_features(Feature[features...])

  pl = policy_length(G)
  fl = feature_length(features, G)
  os = outsize(trunk, size(G))

  # Check the given heads and create linear default heads if not specified
  vhead = prepare_head(vhead, os, 1, GPU)
  phead = prepare_head(phead, os, pl, GPU)
  fhead = fl > 0 ? prepare_head(fhead, os, fl, GPU) : nothing

  NeuralModel{G, GPU}(trunk, features, vhead, phead, fhead, vconv, pconv) 

end

# Low level access to neural models
function (m :: NeuralModel{G})(data, use_features = false) where {G <: Game}

  # Get the trunk output to calculate policy, value, features
  out = m.trunk(data)

  # Apply the converters for value and policy
  v = reshape(m.vconv.(m.vhead(out)), :)
  p = m.pconv(m.phead(out), dims=1)

  # Apply the feature head if features are to be calculated
  if use_features && !isnothing(m.fhead)

    f = m.fhead(out)

    fs = map(features(l), feature_indices(features(l), G)) do feat, sel
      feature_conv(feat, f[sel,:])
    end

    # Collect the converted features
    f = isempty(fs) ? similar(p, 0, length(v)) : vcat(fs...)

  else

    f = similar(p, 0, length(v))

  end

  (v, p, f)

end

# Higher level access
function (m :: NeuralModel{G, GPU})( games :: Vector{G}
                                   , use_features = false
                                   ) where {G <: Game, GPU}

  at = atype(GPU)
  data = convert(at, representation(games))

  m(data, use_features)

end

function (m :: NeuralModel{G, GPU})( game :: G
                                   , use_features = false
                                   ) where {G <: Game, GPU}

  v, p, f = m([game], use_features)

  v[1], reshape(p, :), reshape(f, :)

end

function swap(m :: NeuralModel{G, GPU}) where {G, GPU}
  
  NeuralModel{G, !GPU}( swap(m.trunk)
                      , m.features
                      , swap(m.vhead)
                      , swap(m.phead)
                      , isnothing(m.fhead) ? nothing : swap(m.fhead)
                      , m.vconv
                      , m.pconv )

end

function Base.copy(m :: NeuralModel{G, GPU}) where {G, GPU}

  NeuralModel{G, GPU}( copy(m.trunk)
                     , copy(m.features)
                     , copy(m.vhead)
                     , copy(m.phead)
                     , copy(m.fhead)
                     , m.vconv
                     , m.pconv )

end

features(m :: NeuralModel) = m.features

training_model(m :: NeuralModel) = m

# -------- Linear Neural Model ----------------------------------------------- #

function Shallow(:: Type{G}; kwargs...) where {G <: Game}
  NeuralModel(G, Pointwise(); kwargs...)
end


# -------- Multilayer Perceptron --------------------------------------------- #

function MLP(:: Type{G}, hidden, f = Knet.relu; kwargs...) where {G <: Game}

  widths = [ prod(size(G)), hidden..., policy_length(G) + 1 ]
  layers = [ Dense(widths[j], widths[j+1], f) for j in 1:length(widths) - 2 ]
  push!(layers, Dense(widths[end-1], widths[end], identity))

  NeuralModel(G, Chain(layers...); kwargs...)

end


# -------- Shallow Convolutional Network ------------------------------------ #

function ShallowConv( :: Type{G}
                    , filters
                    , f = Knet.relu
                    ; kwargs...
                    ) where {G <: Game}

  NeuralModel(G, Conv(size(G)[3], filters, f); kwargs...)

end
