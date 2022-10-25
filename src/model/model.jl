
# -------- Model ------------------------------------------------------------- #

"""
A model that evaluates the value of game states and provides policies for the
next action to take. They can be applied to game states via `apply(model,
game)`, which returns a `(value, policy)` tuple.

Jtac models act as the decision makers for Monte-Carlo tree search (MCTS)
players: When expanding the search tree, the proposed `policy` assesses the
different paths according to the models preferences. When reaching an unexplored
game state, the proposed `value` is used for backpropagation to inform and
improve the policy.

Static model implementations, like the `RolloutModel`, yield non-trainable
players, like the classical MCTS algorithm. Parameterized models with adjustable
weights on the other hand, like the `NeuralNetworkModel`, can systematically be
optimized by playing and training. Roughly speaking, the training goal is to
make the proposed policy (before any MCTS steps) and the improved policy (after
these steps) converge.

In order to use a model for playing, it has to be attached to a `Player`, like
the `IntuitionPlayer` (which only uses the policy proposed by the network) or
the `MCTSPlayer` (which conducts several MCTS steps and finally uses the
improved policy for its decision).
"""
abstract type AbstractModel{G <: AbstractGame, GPU} <: Element{GPU} end

Pack.@mappack AbstractModel

to_cpu(a) = convert(atype(false), a)
to_cpu(a :: Float32) = a

to_gpu(a) = convert(atype(true), a)

"""
    apply(model, game)

Apply `model` to `game`, yielding a named `(value, policy)` tuple.
"""
function apply(model :: AbstractModel{G}, game :: G) where {G <: AbstractGame}
  v, p, _ = model(game)
  (value = v, policy = p |> to_cpu)
end

"""
    features(model)

Obtain the list of features which are enabled for `model`.
"""
features(model :: AbstractModel) = Feature[]

"""
    apply_features(model, game)

Apply `model` to `game`, yielding a `(value, policy, feature_values)` tuple.
"""
function apply_features(model :: AbstractModel{G}, game :: G) where {G <: AbstractGame}
  v, p, f = model(game, true)
  (value = v, policy = p |> to_cpu, features = f |> to_cpu)
end


"""
    ntasks(model)

The number of tasks that should be applied if the model is called with `asyncmap`.
"""
ntasks(:: AbstractModel) = 1

"""
    base_model(model)

Extract the model on which `model` or `player` is based. Except for explicit
wrapper models like `Async`, this usually returns the model itself.
"""
base_model(m :: AbstractModel) = m

"""
    training_model(model)
    training_model(player)

Extract the component of `model` or `player` that can be trained. Returns
nothing if `model` is not trainable.
"""
training_model(m :: AbstractModel) = nothing

"""
    playing_model(model)
    playing_model(player)

Extract the model in `model` or `player` that is most suitable for playing.
On models, this always returns the model itself. Can return `nothing` if
`player` does not use a model.

"""
playing_model(m :: AbstractModel) = m

"""
    gametype(model)

Get the julia type `G <: AbstractGame` that `model` can be applied to.
"""
gametype(model :: AbstractModel{G}) where {G <: AbstractGame} = G

"""
    count_params(model)

Count the number of free parameters in `model`.
"""
count_params(model :: AbstractModel) = sum(length, Knet.params(model))

"""
    tune(; gpu, async, cache)
    tune(model; gpu, async, cache)

Auxiliary function to flexibly adapt the GPU status or the Async / Caching
wrappers of a `model` with base model `NeuralModel`. The first function call
returns a function mapping a model to the given configuration, while the second
function call returns the configured model

The argument `gpu` is boolean, while `async` and `cache` can be both boolean or
positive integers (setting the parameters `max_batchsize` for `Async` and
`max_cachesize` for `Caching`, respectively).
"""
function tune(model; gpu, async, cache)
  @warn "trying to tune model of type $(typeof(model)) failed" maxlog = 1
  model
end

tune(; kwargs...) = model -> tune(model; kwargs...)

"""
    save(fname, model)

Save `model` to the file `fname`. The typical file extension
is `.jtm`.
"""
save(fname, model) = open(io -> Pack.pack(io, model), fname, "w")

"""
    load(fname)

Load a Jtac model from `fname`. The typical file extension is `.jtm`.
"""
load(fname) = open(io -> Pack.unpack(io, AbstractModel), fname)

