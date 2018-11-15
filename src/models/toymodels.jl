# Collection of toy models that are mainly used for development and debugging

# Some of the models will return uninformative policies
uniform_policy(length) = ones(Float32, length) / length

function random_policy(length)
  policy = zeros(Float32, length)
  policy[rand(1:length)] = 1.
  policy
end

# Dummy Model
# This model always returns the neutral value 0 and a uniform policy vector.
# Totally uninformative, and for debugging purposes only

struct DummyModel <: Model
  policy_length :: Int
end

DummyModel(game :: Game) = DummyModel(policy_length(game))

(m :: DummyModel)(game :: Game) = Float32[ 0; uniform_policy(m.policy_length) ]
(m :: DummyModel)(games :: Vector{G}) where G <: Game = hcat(m.(games)...)

# Random Model
# This model proposes a random policy

struct RandomModel <: Model
  policy_length :: Int
end

RandomModel(game :: Game) = RandomModel(policy_length(game))

(m :: RandomModel)(game :: Game) = Float32[ 0; random_policy(m.policy_length) ]
(m :: RandomModel)(games :: Vector{G}) where G <: Game = hcat(m.(games)...)

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

function LinearModel(game :: Game)
  logitmodel = Dense(prod(size(game)), policy_length(game) + 1, id)
  GenericModel(logitmodel)
end


function MLP(game :: Game, hidden, f = relu)
  widths = [ prod(size(game)), hidden..., policy_length(game) + 1 ]
  layers = [ Dense(widths[j], widths[j+1], f) for j in 1:length(widths) - 1 ]
  GenericModel(Chain(layers...))
end


function SimpleConv(game :: Game, channels, f = relu)
  logitmodel = Chain(
    Conv(1, channels, f),
    Dense(channels, policy_length(game) + 1, id)
  )
  GenericModel(logitmodel)
end
