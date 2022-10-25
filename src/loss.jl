
# -------- Loss -------------------------------------------------------------- #

"""
Loss type that determines how the output of a NeuralNetwork model is compared 
to policy, value, and feature labels.
"""
struct Loss

  value  :: NamedTuple{(:loss, :weight), Tuple{Function, Float32}}
  policy :: NamedTuple{(:loss, :weight), Tuple{Function, Float32}}
  reg    :: NamedTuple{(:loss, :weight), Tuple{Function, Float32}}

  features :: Vector{Feature}
  fweights :: Vector{Float32}

end

# Default value loss
default_loss(x, y :: Union{Vector, Knet.KnetVector}) = sum(abs2, x .- y)

# Default policy loss
default_loss(x, y :: Union{Matrix, Knet.KnetMatrix}) = -sum(y .* log.(x.+1f-10))

# Default regularization loss for parameters
default_loss(param) = sum(abs2, param)

# Convert Constructor arguments
conv_loss_arg(w :: Real) = (loss = default_loss, weight = Float32(w))
conv_loss_arg(f :: Function) = (loss = f, weight = 1f0)
conv_loss_arg(t :: Tuple{Function, Real}) = (loss = t[1], weight = Float32(t[2]))
conv_loss_arg(nt) = nt

"""
    Loss([; value, policy, reg, features, feature_weights])

Construct a loss.

The arguments `value`, `policy`, and `reg` take either a float (the
respective weight value) or tuples `(f, w)` of a loss function and
a regularization value. The argument `features` is a list of supported features,
and `feature_weights` are the corresponding weights.
"""
function Loss(
             ; value = 1f0
             , policy = 1f0
             , reg = 0f0
             , features = Feature[]
             , feature_weights = ones(Float32, length(features)) )

  Loss( conv_loss_arg(value)
      , conv_loss_arg(policy)
      , conv_loss_arg(reg)
      , Feature[f for f in features]
      , feature_weights )

end

function loss_names(l :: Loss)
  base = ["value", "policy", "reg"]
  features = feature_name.(l.features)
  vcat(base, features)
end

Model.features(l :: Loss) = l.features

function Base.show(io :: IO, l :: Loss)
  v = l.value.weight
  p = l.policy.weight
  r = l.reg.weight
  print(io, "Loss($v value, $p policy, $r reg")
  for (i, (f, w)) in enumerate(zip(l.features, l.fweights))
    print(io, ", $w $(Model.feature_name(f))")
  end
  print(io, ")")
end


# -------- Loss Calculation -------------------------------------------------- #

function loss( l :: Loss
             , model :: NeuralModel{G, GPU}
             , cache :: DataCache{G, GPU}
             ) where {G, GPU}

  n = length(cache)
  use_features = !isnothing(cache.flabel)

  # Get the model output for value/policy labels and feature labels
  v, p, f = model(cache.data, use_features)

  # Calculate the value and policy loss
  vloss = l.value.weight * l.value.loss(v, cache.vlabel) / n
  ploss = l.policy.weight * l.policy.loss(p, cache.plabel) / n

  # Calculate the regularization loss
  rloss = l.reg.weight * sum(Knet.params(model)) do param
    s = size(param)
    maximum(s) < prod(s) ? l.reg.loss(param) : 0f0
  end

  # Calculate the feature losses
  if use_features

    feats = features(l)
    indices = feature_indices(feats, G)

    flosses = map(feats, indices) do feat, sel
      feature_loss(feat, f[sel,:], cache.flabel[sel,:]) / n
    end

  else

    flosses = zeros(Float32, length(features(l)))

  end

  # Combine all losses in one array
  vcat(vloss, ploss, rloss, flosses...)

end

"""
    loss(l, model, dataset [; maxbatch = 1024])

Calculate the loss determined by `l` for `model` on `dataset` while evaluating
model with at most `maxbatch` game states at once.
"""
function loss( l :: Loss
             , model :: NeuralModel{G, GPU}
             , dataset :: DataSet{G}
             ; maxbatch = 1024
             ) where {G, GPU}

  # Check if features can be used
  use_features = feature_compatibility(l, model, dataset)

  # Cut the dataset in batches if it is too large
  batches = Batches(dataset, maxbatch, gpu = GPU, use_features = use_features)

  # Get the total loss
  sum(batches) do cache
    loss(l, model, cache) .* length(cache)
  end ./ length(dataset)

end

"""
    loss(l, model, game, label)

Calculate the loss determined by `l` between `label` and `model(game)` 
"""
function loss(l :: Loss, model :: AbstractModel, game, label)

  loss(l, model, DataSet([game], [label], [zeros(Float32, 0)]))

end

