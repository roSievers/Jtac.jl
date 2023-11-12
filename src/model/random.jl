
function random_onehot(length)
  policy = zeros(Float32, length)
  policy[rand(1:length)] = 1.
  policy
end

"""
This model returns value 0 and a random one-hot policy for each game state.
"""
struct RandomModel{G} <: AbstractModel{G} end

RandomModel(G :: Type{<: AbstractGame}) = RandomModel{G}()

function apply( m :: RandomModel
              , game :: AbstractGame
              ; targets = [:value, :policy] )

  @assert issubset(targets, targetnames(m))
  n = policylength(game)
  (value = 0f0, policy = random_onehot(n))
end

Base.copy(m :: RandomModel) = m
Base.show(io :: IO, m :: RandomModel) = print(io, "RandomModel()")

