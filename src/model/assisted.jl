
struct AssistedModel{G <: AbstractGame} <: AbstractModel{G}
  model :: AbstractModel
  assistant :: AbstractModel
end

function AssistedModel( model :: AbstractModel{H}
                      , assistant :: AbstractModel{F}
                      ) where {H , F}
  @assert F <: H || H <: F "Assistant cannot be applied to the correct game type"
  G = typeintersect(F, H)
  AssistedModel{G}(model, assistant)
end

function apply( m :: AssistedModel{G}
              , game :: G
              ; targets = [:value, :policy]
              ) where {G <: AbstractGame}

  @assert issubset(targets, targetnames(m))
  hint = assist(m.assistant, game)
  if !haskey(hint, :value) || !haskey(hint, :policy)
    r = apply(m.model, game)
    (value = get(hint, :value, r.value), policy = get(hint, :policy, r.policy))
  else
    (value = hint.value, policy = hint.policy)
  end
end

function switchmodel( m :: AssistedModel{G}
                    , model :: AbstractModel{G}
                    ) where {G <: AbstractGame}
  AssistedModel(model, m.assistant)
end

assist(m :: AssistedModel{G}, game :: G) where {G} = assist(m.assistant, game)

function adapt(backend :: Backend, m :: AssistedModel)
  switchmodel(m, adapt(backend, m.model))
end

isasync(m :: AssistedModel) = isasync(m.model)
ntasks(m :: AssistedModel) = ntasks(m.model)
basemodel(m :: AssistedModel) = basemodel(m.model)
childmodel(m :: AssistedModel) = m.model
trainingmodel(m :: AssistedModel) = trainingmodel(m.model)

Base.copy(m :: AssistedModel) = AssistedModel(copy(m.model), copy(m.assistant))

function Base.show(io :: IO, m :: AssistedModel)
  print(io, "Assisted("); show(io, m.model); print(io, ", ")
  show(io, m.assistant); print(io, ")")
end

function Base.show(io :: IO, mime :: MIME"text/plain", m :: AssistedModel)
  print(io, "Assisted("); show(io, m.assistant); print(io, ")")
  show(io, mime, m.model)
end
