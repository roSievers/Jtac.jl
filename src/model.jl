
# A (neural network) model that is trained by playing against itself.
# Each concrete subtype of Model should provide a constructor that
# takes a game to adapt the input and output dimensions.
#
# Also, each subtype must be made callable with arguments of type
# (:: Game) and (:: # Vector{Game})
#
# The output of applying a model is a Vector where the first entry
# is the model prediction of the state value, and the 
# policy_length(game) entries afterwards are the policy (expected to be
# normalized).

abstract type Model end

function apply(model :: Model, game :: Game)
  result = model(game)
  result[1], result[2:end]
end
