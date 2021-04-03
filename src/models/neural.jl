
# -------- Neural Network Head Creation -------------------------------------- #

function prepare_head(head, s, l, gpu)

  if isnothing(head)

    head = Dense(prod(s), l, gpu = gpu)

  else

    @assert valid_insize(head, s) "Head incompatible with trunk."
    @assert prod(outsize(head, s)) == l "Head incompatible with game."
    head = (on_gpu(head) == gpu) ? head : swap(head)

  end

  head

end

# -------- Neural Model ------------------------------------------------------ #

"""
Trainable model that uses a neural network to generate the value and policy
for a game state. Optionally, the network can also predict features by
applying a dedicated dense layer on the representation used for value and
policy prediction.
"""
struct NeuralModel{G, GPU} <: AbstractModel{G, GPU}

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
  @assert length(unique(features)) == length(features) "Provided features are not unique"

  pl = policy_length(G)
  fl = feature_length(features, G)
  os = outsize(trunk, size(G))

  # Check the provided heads and create linear heads if not specified
  vhead = prepare_head(vhead, os, 1, GPU)
  phead = prepare_head(phead, os, pl, GPU)
  fhead = fl > 0 ? prepare_head(fhead, os, fl, GPU) : nothing

  NeuralModel{G, GPU}(trunk, features, vhead, phead, fhead, vconv, pconv) 
end


# Low level access to neural model predictions
function (m :: NeuralModel{G})(data, use_features = false) where {G <: AbstractGame}

  # Get the trunk output to calculate policy, value, features
  out = m.trunk(data)

  bs = size(out)[end]                 # batchsize
  pl = policy_length(G)               # policy length
  fl = feature_length(m.features, G)  # feature length

  # Apply the converters for value and policy on suitable reshapes
  v = m.vconv.(reshape(m.vhead(out), bs))
  p = m.pconv(reshape(m.phead(out), pl, bs), dims=1)

  # Apply the feature head if features are to be calculated
  if use_features && !isnothing(m.fhead)

    f = reshape(m.fhead(out), fl, bs)

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

# Higher level access to neural model predictions
function (m :: NeuralModel{G, GPU})( games :: Vector{G}
                                   , use_features = false
                                   ) where {G <: AbstractGame, GPU}

  at = atype(GPU)
  data = convert(at, representation(games))

  m(data, use_features)

end

function (m :: NeuralModel{G, GPU})( game :: G
                                   , use_features = false
                                   ) where {G <: AbstractGame, GPU}

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
                     , m.features
                     , copy(m.vhead)
                     , copy(m.phead)
                     , isnothing(m.fhead) ? nothing : copy(m.fhead)
                     , m.vconv
                     , m.pconv )

end

features(m :: NeuralModel) = m.features

training_model(m :: NeuralModel) = m

# -------- Linear Neural Model ----------------------------------------------- #

function Shallow(:: Type{G}; kwargs...) where {G <: AbstractGame}
  NeuralModel(G, Pointwise(); kwargs...)
end


# -------- Multilayer Perceptron --------------------------------------------- #

function MLP(:: Type{G}, hidden, f = Knet.relu; kwargs...) where {G <: AbstractGame}

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
                    ) where {G <: AbstractGame}

  NeuralModel(G, Conv(size(G)[3], filters, f); kwargs...)

end
