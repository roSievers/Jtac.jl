
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

# Applying a model
# Low level
apply(weights, state, :: Type{Model}, :: Game) = error("unimplemented")

# High level
function apply(model :: M, game :: Game) :: Array{Float32} where M <: Model
  apply(modelweights(model), modelstate(model), M, game)
end


# Some toy models: DummyModel, RolloutModel, LinearModel
include("models/toymodels.jl")
