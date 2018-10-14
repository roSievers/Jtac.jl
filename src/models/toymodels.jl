# Collection of toy models that are mainly used for development and debugging

# Some of the models will return uninformative policies
uniform_policy(length) = ones(Float32, length) / length

# Dummy Model
# This model always returns the neutral value 0 and a uniform policy vector.
# Totally uninformative, and for debugging purposes only
struct DummyModel <: Model 
  policy_length :: Int
end

DummyModel(game :: Game) = DummyModel(policy_length(game))

modelweights(:: DummyModel)     = Any[]
modelstate(model :: DummyModel) = Any[model.policy_length]

function apply(weigths, state, :: Type{DummyModel}, game :: Game) :: Array{Float32}
  Float32[ 0; uniform_policy(state[1]) ]
end


# Rollout Model
# This model executes random moves untile the game is over and reports the
# result as chance/value. It always proposes a uniform policy vector.
# This integrates the usual rollout step of a MCTS in the MC-Model interface.
# Like DummyModel, this model will not learn anything.
struct RolloutModel <: Model 
  policy_length :: Int
end

RolloutModel(game :: Game) = RolloutModel(policy_length(game))

modelweights(:: RolloutModel)     = Any[]
modelstate(model :: RolloutModel) = Any[model.policy_length]

function apply(weights, state, :: Type{RolloutModel}, game :: Game) :: Array{Float32}
  result = random_playout(game)
  # We can be sure that is_over(result) holds, thus typeof(status(result)) == Int
  Float32[ status(result) * game.current_player; uniform_policy(state[1]) ]
end



# Linear Model
# Model given by a linear transformation of the data representation.
# This model can actually be trained, but will presumably perform abysmal.
struct LinearModel <: Model
  w :: Matrix{Float32}
  b :: Vector{Float32}
end


function LinearModel(game :: Game)
  len  = policy_length(game)
  gamesize = size(game)
  w = randn(Float32, (len + 1, prod(gamesize)))
  b = randn(Float32, len + 1)
  LinearModel(w, b)
end

modelweights(model :: LinearModel) = Any[ model.w, model.b ]
modelstate(:: LinearModel) = Any[]

# TODO: put this function in some utility-functions file!
function softmax(values)
  res = exp.(values .- maximum(values))
  res / sum(res)
end

function apply(weights, state, :: Type{LinearModel}, game :: Game) :: Array{Float32}
  data = representation(game)
  flat_data = reshape(data, prod(size(data)))
  result = weights[1] * (game.current_player * flat_data) + weights[2]
  
  # The value/chance must be between -1 and 1, and the policy should 
  # be a probability vector
  result[1] = tanh(result[1])
  result[2:end] = softmax(result[2:end])

  result
end

