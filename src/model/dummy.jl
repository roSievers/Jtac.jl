
"""
Dummy model that always predicts value 0 and a uniform policy vector.
"""
struct DummyModel{G} <: AbstractModel{G} end

DummyModel(G :: Type{<: AbstractGame}) = DummyModel{G}()

function apply( m :: DummyModel
              , game :: AbstractGame
              ; targets = [:value, :policy] )

  @assert issubset(targets, targetnames(m))
  n = policylength(game)
  (value = 0f0, policy = ones(Float32, n) / n)
end

Base.copy(m :: DummyModel) = m
Base.show(io :: IO, m :: DummyModel) = print(io, "DummyModel()")
