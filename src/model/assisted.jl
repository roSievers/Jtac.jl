
mutable struct Assisted{G} <: AbstractModel{G, false}
  model :: AbstractModel
  assistent :: AbstractModel
end

Pack.register(Assisted)
Pack.@mappack Assisted

function Assisted(model :: AbstractModel{H}, assistent :: AbstractModel{F}) where {H, F}
  @assert F <: H || H <: F "Assistent and model cannot be applied to the same game types"
  G = typeintersect(F, H)
  Assisted{G}(model, assistent)
end

function apply(m :: Assisted{G}, game :: G) where {G}
  hint = assist(m.assistent, game)
  if !haskey(hint, :value) || !haskey(hint, :policy)
    r = apply(m.model, game)
    (value = get(hint, :value, r.value), policy = get(hint, :policy, r.policy))
  else
    (value = hint.value, policy = hint.policy)
  end
end

assist(m :: Assisted{G}, game :: G) where {G} = assist(m.assistent, game)

swap(m :: Assisted) = @warn "Assisted cannot be swapped"
Base.copy(m :: Assisted) = Assisted(copy(m.model), copy(m.assistent))

ntasks(m :: Assisted) = ntasks(m.model)
base_model(m :: Assisted) = base_model(m.model)
training_model(m :: Assisted) = training_model(m.model)

is_async(m :: Assisted) = is_async(m.model)

function Base.show(io :: IO, m :: Assisted)
  print(io, "Assisted("); show(io, m.model); print(io, ", ")
  show(io, m.assistent); print(io, ")")
end

function Base.show(io :: IO, mime :: MIME"text/plain", m :: Assisted)
  print(io, "Assisted("); show(io, m.assistent); print(io, ")")
  show(io, mime, m.model)
end
