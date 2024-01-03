
"""
Quantity that models try to predict, given the current game state.

Mandatory targets, which must be predictable by all models, are the value of
the current state and the policy recommendation for the next action.
The types [`DefaultValueTarget`](@ref) and [`DefaultPolicyTarget`](@ref) provide
default implementations for these targets.

Other targets are used to enhance trainable models, like the
[`Model.NeuralModel`](@ref). To this end, labels for specified targets can be
generated during self-play (see [`Target.label`](@ref) and
[`Training.record`](@ref)) and can be stored in datasets accessible to training
(see [`Training.DataSet`](@ref)).

# Implementation

The following methods have to be implemented for concrete subtypes `T{G} <:
AbstractTarget{G}`, where `G` denotes the game type supported by the target.

- [`Base.length`](@ref): The number of values to be predicted.
- [`Target.label`](@ref): Generate a prediction label for the target. Used \
  during dataset creation.

Optionally, the methods [`Target.defaultlossfunction`](@ref) and
[`Target.defaultactivation`](@ref) may be extended.
"""
abstract type AbstractTarget{G <: AbstractGame} end

@pack {<: AbstractTarget} in TypedFormat{MapFormat}

"""
    length(target :: Target.AbstractTarget)

Length of the prediction target `target`. For example,
`length(:: DefaultPolicyTarget{G}) where {G} = Game.policylength(G)`.
"""
Base.length(t :: AbstractTarget) = error("Not implemented")

"""
Context object that is used to generate target labels.

A `LabelContext` provides information about a game state within its match.
During recording of prediction labels, one context per game state is passed to
[`Target.label`](@ref).
"""
struct LabelContext{G <: AbstractGame}
  game :: G
  player_policy :: Vector{Float32}
  outcome :: Status
  
  index :: Int
  games :: Vector{G}
  player_policies :: Vector{Vector{Float32}}
end

"""
    label(target, context)

Produce a prediction label for `target`, given a `context` object of type
[`LabelContext`](@ref). The latter stores information about the current game
state, the `history` of the full match, and the relevant player predictions.
"""
function label(t :: AbstractTarget{G}, ctx :: LabelContext{G}) where {G}
  error("Not implemented")
end

"""
    defaultlossfunction(target)

Returns the name of the default loss function for `target`. This loss function
is used if no loss function is explicitly specified in calls to
[`Training.loss`](@ref).

If not implemented for subtypes of `AbstractTarget`, it defaults to `:sumabs2`.
"""
defaultlossfunction(:: AbstractTarget) = :sumabs2

"""
    defaultactivation(target)

Returns the name of the default output activation function for `target`. This
activation function is used if no alternative is explicitly specified when
constructing a [`Model.NeuralModel`](@ref).

If not implemented for subtypes of `AbstractTarget`, it defaults to `:identity`.
"""
defaultactivation(:: AbstractTarget) = :identity

"""
    targets(object)

Returns a named tuple of the targets associated to `object`.
For example, this applies to the prediction targets of an `AbstractModel` or
a `DataSet`.
"""
targets(object) = error("not implemented")

"""
    targetnames(object)

Returns a vector of target names associated to `object`.
"""
targetnames(object) = collect(keys(targets(object)))

"""
    compatibletargets(a, b)

Given two objects `a` and `a` that support [`targets`], return a named tuple of
compatible targets (i.e., with same name and equal target value).
"""
function compatibletargets(a, b)
  atargets = Target.targets(a)
  btargets = Target.targets(b)

  names = intersect(keys(atargets), keys(btargets))
  filter!(names) do name
    atargets[name] == btargets[name]
  end

  atargets[names]
end


"""
Default value target. Uses the outcome of the match as value label for each game
state of the match.
"""
struct DefaultValueTarget{G} <: AbstractTarget{G} end

function DefaultValueTarget(G :: Type{<: AbstractGame})
  DefaultValueTarget{G}()
end

Base.length(:: DefaultValueTarget) = 1

function label(:: DefaultValueTarget{G}, ctx :: LabelContext{G}) where {G}
  value = activeplayer(ctx.game) * Int(ctx.outcome)
  Float32[value]
end

defaultactivation(:: DefaultValueTarget) = :tanh

"""
Default policy target. Uses the player policy as policy label.
"""
struct DefaultPolicyTarget{G} <: AbstractTarget{G} end

function DefaultPolicyTarget(G :: Type{<: AbstractGame})
  DefaultPolicyTarget{G}()
end

function Base.length(:: DefaultPolicyTarget{G}) where {G <: AbstractGame}
  policylength(G)
end

function label(:: DefaultPolicyTarget{G}, ctx :: LabelContext{G}) where {G}
  ctx.player_policy
end

defaultactivation(:: DefaultPolicyTarget) = :softmax
defaultlossfunction(:: DefaultPolicyTarget) = :crossentropy


"""
    defaulttargets(G)

Returns a named tuple with the default value target for `G` under name `value`
and the default policy target for `G` under name `policy`.
"""
defaulttargets(G :: Type{<: AbstractGame}) = (
  value = DefaultValueTarget(G),
  policy = DefaultPolicyTarget(G),
)


"""
Dummy target that always returns a constant label. Useful for testing purposes.
"""
struct DummyTarget{G} <: AbstractTarget{G}
  data :: NTuple{N, Float32} where {N}
end

function DummyTarget(G :: Type{<: AbstractGame}, data = [0.0f0, 42.0f0])
  DummyTarget{G}(Tuple(data))
end

Base.length(t :: DummyTarget) = length(t.data)

label(t :: DummyTarget{G}, :: LabelContext{G}) where {G} = collect(t.data)

