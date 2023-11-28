
"""
This model implements the classical MCTS rollout step. Given a game state, it
executes random moves until the game ends. The result is returned as value,
while the policy proposal is always uniform.
"""
struct RolloutModel{G} <: AbstractModel{G} end

RolloutModel(G :: Type{<: AbstractGame}) = RolloutModel{G}()

function apply( m :: RolloutModel
              , game :: AbstractGame
              ; targets = [:value, :policy] )

  @assert issubset(targets, targetnames(m))
  n = policylength(game)
  result = randommatch(game)
  value = Int(status(result)) * activeplayer(game)

  (; value = Float32(value), policy = ones(Float32, n) / n)
end

Base.copy(m :: RolloutModel) = m
Base.show(io :: IO, m :: RolloutModel) = print(io, "RolloutModel()")

