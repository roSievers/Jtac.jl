
struct RolloutModel{G} <: AbstractModel{G} end

"""
This model implements a classical MCTS rollout step. Given a game state, it
executes random moves until the game ends. The result is returned as value,
while the policy proposal is always uniform.

---

    RolloutModel(G)

Create a `RolloutModel` compatible with game type `G`.
"""
RolloutModel(G :: Type{<: AbstractGame}) = RolloutModel{G}()

function apply( m :: RolloutModel
              , game :: AbstractGame
              ; targets = [:value, :policy] )

  @assert issubset(targets, targetnames(m))
  n = policylength(game)
  result = Game.rollout(game)
  value = Int(status(result)) * mover(game)

  (; value = Float32(value), policy = ones(Float32, n) / n)
end

Base.copy(m :: RolloutModel) = m
Base.show(io :: IO, m :: RolloutModel) = print(io, "RolloutModel()")

