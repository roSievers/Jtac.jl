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

(m :: DummyModel)(game :: Game) = Float32[ 0; uniform_policy(m.policy_length) ]
(m :: DummyModel)(games :: Vector{G}) where G <: Game = hcat(m.(games)...)


# Rollout Model
# This model executes random moves untile the game is over and reports the
# result as chance/value. It always proposes a uniform policy vector.
# This integrates the usual rollout step of a MCTS in the MC-Model interface.
# Like DummyModel, this model will not learn anything.
struct RolloutModel <: Model 
  policy_length :: Int
end

RolloutModel(game :: Game) = RolloutModel(policy_length(game))

function (m :: RolloutModel)(game :: Game)
  result = random_playout(game)
  # We can be sure that is_over(result) holds, thus typeof(status(result)) == Int
  Float32[ status(result) * current_player(game); uniform_policy(m.policy_length) ]
end

(m :: RolloutModel)(games :: Vector{G}) where G <: Game = hcat(m.(games)...)


# Linear Model
# A linear model that will perform terrible

struct LinearModel <: Model
  layer :: Dense
end

function LinearModel(game :: Game)
  layer = Dense(prod(size(game)), policy_length(game) + 1, x -> x)
  LinearModel(layer)
end

function (m :: LinearModel)(games :: Vector{G}) where G <: Game
  data = representation(games)
  result = m.layer(data)

  result[1,:] = tanh.(result[1,:])
  result[2:end,:] = softmax(result[2:end,:], dims = 1)

  result
end

(m :: LinearModel)(game :: Game) = reshape(m([game]), (policy_length(game) + 1,))

