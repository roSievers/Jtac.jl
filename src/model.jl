
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
optimized by playing and training. Roughly speaking, the training goal is to let
the proposed policy (before any MCTS steps) and the improved policy (after these
steps) coincide.

In order to use a model for playing, it has to be attached to a `Player`, like
the `IntuitionPlayer` (which only uses the policy proposed by the network) or
the `MCTSPlayer` (which conducts several MCTS steps and finally uses the
improved policy for its decision).
"""
abstract type Model{G <: Game, GPU} <: Element{GPU} end

to_cpu(a :: Knet.KnetArray{Float32}) = convert(Array{Float32}, a)
to_cpu(a :: Array{Float32}) = a

to_gpu(a :: Knet.KnetArray{Float32}) = a
to_gpu(a :: Array{Float32}) = convert(Knet.KnetArray{Float32}, a)

"""
    apply(model, game)

Apply `model` to `game`, yielding a `(value, policy)` tuple.
"""
function apply(model :: Model{G}, game :: G) where {G <: Game}
  v, p, _ = model(game)
  (value = v, policy = p |> to_cpu)
end

"""
    features(model)

Obtain the list of features which are enabled for `model`.
"""
features(model :: Model) = Feature[]

"""
    apply_features(model, game)

Apply `model` to `game`, yielding a `(value, policy, feature_values)` tuple.
"""
function apply_features(model :: Model{G}, game :: G) where {G <: Game}
  v, p, f = model(game, true)
  (value = v, policy = p |> to_cpu, features = f |> to_cpu)
end


"""
    ntasks(model)

The number of tasks that should be applied if the model is called with `asyncmap`.
"""
ntasks(:: Model) = 1

"""
    base_model(model)

Extract the model on which `model` or `player` is based. Except for explicit
wrapper models like `Async`, this usually returns the model itself.
"""
base_model(m :: Model) = m

"""
    training_model(model)
    training_model(player)

Extract the component of `model` or `player` that can be trained. Returns
nothing if `model` is not trainable.
"""
training_model(m :: Model) = nothing

"""
    playing_model(model)
    playing_model(player)

Extract the model in `model` or `player` that is most suitable for playing.
On models, this always returns the model itself. Can return `nothing` if
`player` does not use a model.

"""
playing_model(m :: Model) = m

"""
    gametype(model)

Get the julia type `G <: Game` that `model` can be applied to.
"""
gametype(model :: Model{G}) where {G <: Game} = G
