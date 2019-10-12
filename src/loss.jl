
# -------- Loss -------------------------------------------------------------- #

"""
Loss type that determines how the output of a NeuralNetwork model is compared 
to policy, value, and feature labels.
"""
struct Loss

  value          :: NamedTuple{(:loss, :weight), Tuple{Function, Float32}}
  policy         :: NamedTuple{(:loss, :weight), Tuple{Function, Float32}}
  regularization :: NamedTuple{(:loss, :weight), Tuple{Function, Float32}}

  features :: Vector{Feature}
  fweights :: Vector{Float32}

end

# Default value loss
default_loss(x, y :: Union{Vector, Knet.KnetVector}) = sum(abs2, x .- y)

# Default policy loss
default_loss(x, y :: Union{Matrix, Knet.KnetMatrix}) = -sum(y .* log.(x))

# Default regularization loss for parameters
default_loss(param) = sum(abs2, param)

# Convert Constructor arguments
conv_loss_arg(w :: Real) = (loss = default_loss, weight = Float32(w))
conv_loss_arg(f :: Function) = (loss = f, weight = 1f0)
conv_loss_arg(t :: Tuple{Function, Real}) = (loss = t[1], weight = Float32(t[2]))
conv_loss_arg(nt) = nt

"""
    Loss([; value, policy, regularization, features, feature_weights])

Construct a loss.

The arguments `value`, `policy`, and `regularization` take either a float (the
respective weight value) or tuples `(f, w)` of a loss function and
a regularization value. The argument `features` is a list of supported features,
and `feature_weights` are the corresponding weights.
"""
function Loss(
             ; value = 1f0
             , policy = 1f0
             , regularization = 0f0
             , features = Feature[]
             , feature_weights = ones(Float32, length(features)) )

  Loss( conv_loss_arg(value)
      , conv_loss_arg(policy)
      , conv_loss_arg(regularization)
      , Feature[f for f in features]
      , feature_weights )

end

function caption(l :: Loss)
  base = [:value, :policy, :reg]
  features = feature_name.(l.features)
  vcat(base, features)
end

features(l :: Loss) = l.features


# -------- Loss Calculation -------------------------------------------------- #

"""
    loss(l, model, dataset)

Calculate the loss determined by `l` for `model` on `dataset`.
"""
function loss( l :: Loss
             , model :: NeuralModel{G, GPU}
             , dataset :: DataSet{G}
             ) where {G, GPU}

  n = length(dataset)

  # Check if the features in the loss, model, and dataset are compatible and
  # prepare the dataset for evaluation
  use_features = check_features(l, model, dataset)
  cache = prepare_data(dataset, gpu = GPU, use_features = use_features)

  # Get the model output for value/policy labels and feature labels
  v, p, f = model(cache.data, use_features)

  # Calculate the value and policy loss
  vloss = l.value.weight * l.value.loss(v, cache.vlabel) / n
  ploss = l.policy.weight * l.policy.loss(p, cache.plabel) / n

  # Calculate the regularization loss
  rloss = sum(Knet.params(model)) do param
    s = size(param)
    maximum(s) < prod(s) ? l.regularization.loss(param) : 0f0
  end

  rloss *= l.regularization.weight

  # Calculate the feature losses
  flosses = zeros(Float32, length(features(l)))

  if !isnothing(cache.flabel)

    # Iterate over features to get the losses
    j = 0
    for (i, f) in enumerate(features(ls))

      l = feature_length(f, G)
      sel = (j+1:j+l)

      floss = feature_loss(f, f[sel,:], cache.flabel[sel,:])
      flosses[i] = ls.feature_weights[i] * floss / n

      j += l

    end

  end

  # Combine all losses in one array
  vcat(vloss, ploss, rloss, flosses...)

end

"""
    loss(l, model, game, label)

Calculate the loss determined by `l` between `label` and `model(game)` 
"""
function loss(l :: Loss, model :: Model, data, label)

  loss(l, model, DataSet([game], [label]))

end

