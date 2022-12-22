
# -------- Policies ---------------------------------------------------------- # 

uniform_policy(length) = ones(Float32, length) / length

function random_policy(length)
  policy = zeros(Float32, length)
  policy[rand(1:length)] = 1.
  policy
end

# -------- Glueing together single outputs ----------------------------------- #

function cat_outputs(outputs)

  n = length(outputs)
  v, p, f = first(outputs)

  vs = zeros(Float32, n)
  ps = zeros(Float32, length(p), n)
  fs = zeros(Float32, length(f), n)

  for (i, o) in enumerate(outputs)
    vs[i]   = o[1]
    ps[:,i] = o[2]
    fs[:,i] = o[3]
  end

  vs, ps, fs

end

# -------- Dummy Model ------------------------------------------------------- #

"""
This model returns value 0 and a uniform policy vector for each game state.
"""
struct DummyModel <: AbstractModel{AbstractGame, false} end

apply(m :: DummyModel, g :: AbstractGame) =
  (value = 0f0, policy = uniform_policy(policy_length(g)))

Base.copy(m :: DummyModel) = m

Base.show(io :: IO, m :: DummyModel) = print(io, "DummyModel()")

# -------- Random Model ------------------------------------------------------ #

"""
This model returns value 0 and a randomly drawn policy vector distribution for
each game state.
"""
struct RandomModel <: AbstractModel{AbstractGame, false} end

apply(m :: RandomModel, g :: AbstractGame) =
  (value = 0f0, policy = random_policy(policy_length(g)))

Base.copy(m :: RandomModel) = m

Base.show(io :: IO, m :: RandomModel) = print(io, "RandomModel()")

# -------- Rollout Model ----------------------------------------------------- #

"""
This model executes random moves until the game is over. The result is returned
as the value of the game state. It always proposes a uniform policy vector.
Therefore, the classical rollout step of MCTS is implemented when this model
is used for the tree search.
"""
struct RolloutModel <: AbstractModel{AbstractGame, false} end

function apply(m :: RolloutModel, g :: AbstractGame)
  result = random_playout(g)
  value = status(result) * current_player(g)

  (value = Float32(value), policy = uniform_policy(policy_length(g)))
end

Base.copy(m :: RolloutModel) = m

Base.show(io :: IO, m :: RolloutModel) = print(io, "RolloutModel()")

