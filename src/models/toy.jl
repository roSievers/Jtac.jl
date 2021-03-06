
# -------- Policies ---------------------------------------------------------- # 

uniform_policy(length) = ones(Float32, length) / length

function random_policy(length)
  policy = zeros(Float32, length)
  policy[rand(1:length)] = 1.
  policy
end

# -------- Glueing Single Outputs -------------------------------------------- #

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
struct DummyModel <: Model{Game, false} end

(m :: DummyModel)(g :: Game, args...) = 0f0, uniform_policy(policy_length(g)), Float32[]

(m :: DummyModel)(g :: Vector{<:Game}, args...) = cat_outputs(m.(g, args...))

Base.copy(m :: DummyModel) = m


# -------- Random Model ------------------------------------------------------ #

"""
This model returns value 0 and a randomly drawn policy vector distribution for
each game state.
"""
struct RandomModel <: Model{Game, false} end

(m :: RandomModel)(g :: Game, args...) = 0f0, random_policy(policy_length(g)), Float32[]

(m :: RandomModel)(g :: Vector{<: Game}, args...) = cat_outputs(m.(g, args...))

Base.copy(m :: RandomModel) = m


# -------- Rollout Model ----------------------------------------------------- #

"""
This model executes random moves until the game is over. The result is returned
as the value of the game state. It always proposes a uniform policy vector.
By that, the classical rollout step of MCTS is implemented when this model
is used for the tree search.
"""
struct RolloutModel <: Model{Game, false} end

function (m :: RolloutModel)(g :: Game, args...)

  result = random_playout(g)
  value = status(result) * current_player(g)

  Float32(value), uniform_policy(policy_length(g)), Float32[]

end

(m :: RolloutModel)(g :: Vector{<: Game}, args...) = cat_outputs(m.(g, args...))

Base.copy(m :: RolloutModel) = m

