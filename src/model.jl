

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
    save_model(name, model)

Save `model` under the filename `name` with automatically appended extension
".jtm". Note that the model is first converted to a saveable format, i.e., it is
moved to the CPU and `training_model(model)` is extracted.
"""
function save_model(fname :: String, model :: Model{G}) where {G}

  # Create a cpu-based copy of the training model
  model = model |> training_model |> to_cpu |> copy

  # Reset the optimizers, as they will not be saved
  for p in Knet.params(model)
    p.opt = nothing
  end

  BSON.bson(fname * ".jtm", model = model)

end

"""
    load_model(name)

Load a model from file `name`, where the extension ".jtm" is automatically
appended.
"""
function load_model(fname :: String)
  BSON.load(fname * ".jtm")[:model]
end

"""
    ntasks(model)

The number of tasks that should be applied if the model is called with `asyncmap`.
"""
ntasks(:: Model) = 1

"""
    training_model(model)

Extract the part of `model` that can explicitly be trained by stochastic
gradient descent. This is, for instance, important if `model` is wrapped by
`Async`, which makes playouts faster but stifles backpropagation.
"""
training_model(m :: Model) = m
#training_model(m) = nothing

