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
struct DummyModel <: Model{Game, false} end

(m :: DummyModel)(game :: Game) = Float32[ 0; uniform_policy(policy_length(game)) ]
(m :: DummyModel)(games :: Vector{G}) where {G <: Game} = hcat(m.(games)...)

Base.copy(m :: DummyModel) = m

# Random Model
# This model proposes a random policy
struct RandomModel <: Model{Game, false} end

(m :: RandomModel)(game :: Game) = Float32[ 0; random_policy(policy_length(game)) ]
(m :: RandomModel)(games :: Vector{G}) where {G <: Game} = hcat(m.(games)...)

Base.copy(m :: RandomModel) = m

# Rollout Model
# This model executes random moves until the game is over and reports the
# result as chance/value. It always proposes a uniform policy vector.
# This integrates the usual rollout step of a MCTS in the MC-Model interface.
# Like DummyModel, this model will not learn anything.
struct RolloutModel <: Model{Game, false} end

function (m :: RolloutModel)(game :: Game)
  result = random_playout(game)
  # We can be sure that is_over(result) holds, thus typeof(status(result)) == Int
  Float32[ status(result) * current_player(game); uniform_policy(policy_length(game)) ]
end

(m :: RolloutModel)(games :: Vector{G}) where {G <: Game} = hcat(m.(games)...)

Base.copy(m :: RolloutModel) = m

