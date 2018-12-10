
# Elements
# --------------------------------------------------------------------------- #

# Neural network architectures are usually composed of different layers. We
# want separate types for Models on the one hand, which we can apply to games,
# and Layers on the other hand, which are applied to raw data and are
# composable. Both of these types, however, come in two flavors: living on the
# GPU or the CPU. In order to capture the common interface for swapping from/to
# GPU memory, we introduce the Element type.

abstract type Element{GPU} end

swap(l :: Element) = error("Not implemented")
on_gpu(:: Element{GPU}) where {GPU} = GPU

to_cpu(el :: Element{false}) :: Element{false} = el
to_gpu(el :: Element{true})  :: Element{true}  = el

to_cpu(el :: Element{true})  :: Element{false} = swap(el)
to_gpu(el :: Element{false}) :: Element{true}  = swap(el)

Base.copy(l :: Element) :: Element = error("Not implemented")


# Models
# --------------------------------------------------------------------------- #

# A (neural network) model that is trained by playing against itself. Each
# concrete subtype of Model should provide a constructor that takes a game to
# adapt the input and output dimensions.
#
# Also, each subtype must be made callable with arguments of type (:: Game) and
# (:: # Vector{Game}).
#
# The output of applying a model is a Vector where the first entry is the model
# prediction of the state value, and the policy_length(game) entries afterwards
# are the policy (expected to be normalized).

abstract type Model{G <: Game, GPU} <: Element{GPU} end

function apply(model :: Model{G, GPU}, game :: G) where {G, GPU}
  result = model(game)
  result[1], result[2:end]
end

# Saving and loading models
# Maybe we should think about something more version-stable here,
# because this can break if AutoGrad, or Knet, or BSON, or Jtac changes

function save_model(fname :: String, model :: Model{G, false}) where {G}
  bson(fname * ".jtm", model = model)
end

function load_model(fname :: String)
  BSON.load(fname * ".jtm")[:model]
end

