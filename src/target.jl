
"""
A `Target` is a component of the loss function. It is either
a `PredictionTarget` (like the value, policy, or the next move of the enemy) or
a `RegularizationTarget` (like the L2 norm of all network weights).

The labels for a `PredictionTarget` are constructed during mcts selfplays
via the function

    evaluate(target, game, game_idx, history)

which has access to the current game state, the game index, and the the full
history of game states of the match.
For each prediction target, the neural model requires an additional target head
with output size

    Base.length(target)

The output of this target head is activated by

    activate(target, output)

which could involve normalizations like `tanh` or `softmax`. During training,
the network prediction is compared to the actual target label via

    loss(target, model, label, prediction)

In contrast, a `RegularizationTarget` only has to implement

    loss(target, model)

since no labels or length specifications are needed.

Two prediction targets receive special treatment: `ValueTarget` and
`PolicyTarget`, for which `evaluate` does not have to be defined (evaluation
automatically happens during the selfplay logic). Each network automatically
includes a policy and a value head, since this is required for the mcts.

Each target has to be named via

    name(target)

The name is used to pretty-print loss information and to assign weights to the
different components of the loss function.
"""
abstract type AbstractTarget end

Pack.@mappack AbstractTarget
Pack.register(AbstractTarget)

abstract type PredictionTarget{G <: AbstractGame} <: AbstractTarget end
abstract type RegularizationTarget <: AbstractTarget end

Pack.register(PredictionTarget)
Pack.register(RegularizationTarget)

Base.length(t :: PredictionTarget) = error("Not implemented")

"""
    evaluate(target, game, game_idx, history)

Produce a `target` label given a `game` state, the `history` of the
full match, and the index `game_idx` of `game` within the history.
"""
evaluate(t :: PredictionTarget, game, game_idx, history) = error("Not implemented")

"""
    activate(target, x)

Post-process the output of the target layer in the neural network.
"""
activate(t :: PredictionTarget, x) = x

"""
    loss(pred_target, model, label, prediction)
    loss(reg_target, model)

Loss function of a target.
"""
loss(t :: PredictionTarget, l, v) = error("Not implemented")
loss(t :: RegularizationTarget, model) = error("Not implemented")

"""
    name(target)

Name of `target` of type `Symbol`. Used for pretty printing and assigning
weights.
"""
name(t :: AbstractTarget) = error("Not implemented")


"""
    compatible(targets...)

Whether all `targets` have compatible names and lengths.
"""
function compatible(targets :: PredictionTarget...)
  lengths = length.(targets)
  names = name.(targets)
  all(lengths .== lengths[1]) && all(names .== names[1])
end

"""
    defaults(G)

Returns the default `ValueTarget(G)` and `PolicyTarget(G)` for gametype `G`.
"""
defaults(G :: Type{<: AbstractGame}) =
  PredictionTarget[ValueTarget(G), PolicyTarget(G)]

"""
    targets(object)

Retrieve the targets associated to `object`. For example, this applies to the
prediction targets of an `AbstractModel` or a `DataSet`.
"""
targets(object) = error("not implemented")
targets(ts :: Vector{<: AbstractTarget}) = ts

"""
    adapt(object, targets)

Return a version of `object` whose targets are compatible with `targets`.

If `object` is a vector of targets, return the target indices that make it
compatible to `targets`.
"""
function adapt(ts :: Vector{<: AbstractTarget}, tss)
  idx = []
  names = name.(ts)
  for target in targets(tss)
    name = Target.name(target)
    i = findfirst(isequal(name), names)
    @assert !isnothing(i) "Cannot adapt targets: $name not found"
    push!(idx, i)
  end
  idx
end


#
# Value target
#

"""
Value target. Receives special treatment and does not have to implement `evaluate`.
"""
struct ValueTarget{G} <: PredictionTarget{G} end

Pack.register(ValueTarget)

ValueTarget(G :: Type{<: AbstractGame}) = ValueTarget{G}()

Base.length(:: ValueTarget) = 1
name(:: ValueTarget) = :value
activate(:: ValueTarget, x) = Knet.tanh.(x)
loss(:: ValueTarget, l, v) = sum(abs2, l .- v)

#
# Policy target
#

"""
Policy target. Receives special treatment and does not have to implement
`evaluate`.
"""
struct PolicyTarget{G} <: PredictionTarget{G} end

Pack.register(PolicyTarget)

PolicyTarget(G :: Type{<: AbstractGame}) = PolicyTarget{G}()

Base.length(:: PolicyTarget{G}) where {G <: AbstractGame} = policy_length(G)
name(:: PolicyTarget) = :policy
activate(:: PolicyTarget, x) = Knet.softmax(x, dims = 1)
loss(:: PolicyTarget, l, v) = - sum(l .* log.(v .+ 1f-10))


#
# Dummy target for testing
#

struct DummyTarget{G} <: PredictionTarget{G}
  data :: Vector{Float32}
end

Pack.register(DummyTarget)

DummyTarget(G :: Type{<: AbstractGame}, data = [0.0f0]) = DummyTarget{G}(data)

Base.length(t :: DummyTarget) = length(t.data)
name(:: DummyTarget) = :dummy

loss(:: DummyTarget, l, v) = sum(abs2, l .- v)
evaluate(t :: DummyTarget, _game, _idx, _history) = t.data


#
# Regularization targets
#

struct L2Reg <: RegularizationTarget end

Pack.register(L2Reg)

name(:: L2Reg) = :l2reg

function loss(:: L2Reg, model)
  sum(Model.params(model, false)) do param
    sum(abs2, param)
  end
end

struct L1Reg <: RegularizationTarget end

Pack.register(L1Reg)

name(:: L1Reg) = :l1reg

function loss(:: L1Reg, model)
  sum(Model.params(model, false)) do param
    sum(abs, param)
  end
end

