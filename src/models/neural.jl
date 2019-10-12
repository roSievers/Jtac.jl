
# -------- Neural Model ------------------------------------------------------ #

"""
Trainable model that uses a neural network to generate predictions for
the value and policy for a game state.
"""
struct NeuralModel{G, GPU} <: Model{G, GPU}

  layer :: Layer{GPU}           # Takes input and returns layer before logits
  features :: Vector{Feature}   # Features that the network must predict

  vphead                        # value/policy head
  fhead                         # feature head

  vconv                         # Converts value-logit to value
  pconv                         # Converts policy-logits to policy

end

"""
    NeuralModel(G, layer, features...; convert_value, convert_policy)

Constructs a neural model for gametype `G` from the neural network `layer`
with `features` enabled. The functions `convert_value` and `convert_policy`
are used to convert the logits to values, respectively policies.
"""
function NeuralModel( :: Type{G}
                    , layer :: Layer{GPU}
                    , features = Feature[]
                    ; value_conv = Knet.tanh
                    , policy_conv = Knet.softmax
                    ) where {G, GPU}

  @assert valid_insize(layer, size(G)) "Input layer does not fit the game"

  features = check_features(Feature[features...])

  pl = policy_length(G)
  fl = feature_length(features, G)

  # Create dense layers that operate on `layer` as heads to create value/policy
  # and feature outputs
  os = outsize(layer, size(G))
  vphead = Dense(prod(os), pl + 1, gpu = GPU)
  fhead = Dense(prod(os), fl, gpu = GPU)

  NeuralModel{G, GPU}( layer
                     , features
                     , vphead
                     , fhead
                     , value_conv
                     , policy_conv )

end

# Low level access to neural models
function (m :: NeuralModel{G})(data, use_features = false) where {G <: Game}

  # Get the general network output used to calculate policy, value, features
  output = m.layer(data)

  # Apply the value / policy head
  vp = m.vphead(output)

  # Apply the converters for value and policy
  v  = m.vconv.(vp[1,:])
  p  = m.pconv(vp[2:end,:], dims=1)

  # Apply the feature head, if features are to be calculated
  if use_features

    fout = m.fhead(output)

    j = 0
    flabels = []
    
    # Apply the feature converters
    for f in features(m)
      l = feature_length(f, G)
      push!(flabels, feature_conv(f, fout[j+1:j+l,:]))
      j += l
    end

    # Collect the converted features again
    fout = vcat(flabels...)

  else

    fout = similar(p, 0, length(v))

  end

  (v, p, fout)

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
  
  NeuralModel{G, !GPU}( swap(m.layer)
                      , m.features
                      , swap(m.vphead)
                      , swap(m.fhead)
                      , m.vconv
                      , m.pconv )

end

function Base.copy(m :: NeuralModel{G, GPU}) where {G, GPU}

  NeuralModel{G, GPU}( copy(m.layer)
                     , copy(m.features)
                     , copy(m.vphead)
                     , copy(m.fhead)
                     , m.vconv
                     , m.pconv )

end

features(m :: NeuralModel) = m.features


# -------- Linear Neural Model ----------------------------------------------- #

function Shallow(:: Type{G}; kwargs...) where {G <: Game}
  NeuralModel(G, Pointwise(); kwargs...)
end


# -------- Multilayer perception --------------------------------------------- #

function MLP(:: Type{G}, hidden, f = Knet.relu; kwargs...) where {G <: Game}

  widths = [ prod(size(G)), hidden..., policy_length(G) + 1 ]
  layers = [ Dense(widths[j], widths[j+1], f) for j in 1:length(widths) - 2 ]
  push!(layers, Dense(widths[end-1], widths[end], identity))

  NeuralModel(G, Chain(layers...); kwargs...)

end


# -------- Shallow Convolutional Networrk ------------------------------------ #

function ShallowConv( :: Type{G}
                    , filters
                    , f = Knet.relu
                    ; kwargs...
                    ) where {G <: Game}

  NeuralModel(G, Conv(size(G)[3], filters, f); kwargs...)

end
