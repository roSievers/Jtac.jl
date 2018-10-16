
# A (neural network) model that is trained by playing against itself.
# Each concrete subtype of Model should provide a constructor that
# takes a game to adapt the input and output dimensions
abstract type Model end

# The model is usually characterized by
#   - the model type
#   - the model weights (updated by gradient descent during training)
#   - the model state (not updated by gradient descent)
# Both the weights and the state are assumed to be of type Array{Any}
modelweights(::Model) = error("unimplemented")
modelstate(::Model)   = error("unimplemented")
# TODO: check if we can also leave the type of modelstate unspecified

#
# Applying a model
#

# Low level
# Takes raw weights and state and returns a result vector [value; policy]
apply(weights, state, :: Type{Model}, :: Game) :: Vector{Float32} = error("unimplemented")

# High level
# Takes a complete model and returns tuple (value, policy)
function apply(model :: M, game :: Game) :: Tuple{Float32, Vector{Float32}} where M <: Model
  result = apply(modelweights(model), modelstate(model), M, game)
  (result[1], result[2:end])
end

