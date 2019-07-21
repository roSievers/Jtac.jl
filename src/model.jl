
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

to_cpu(el :: Element{false}) = el
to_gpu(el :: Element{true})  = el

to_cpu(el :: Element{true}) = swap(el)

function to_gpu(el :: Element{false}) 
  if Knet.gpu() == -1
    @warn "No GPU was found by Knet. Element stays on CPU"
    el
  else
   swap(el)
  end
end

Base.copy(l :: Element) :: Element = error("Not implemented")


# Models
# --------------------------------------------------------------------------- #

# A (neural network) model that is trained by playing against itself. Each
# concrete subtype of Model should provide a constructor that takes a game to
# adapt the input and output dimensions.
#
# Also, each subtype must be made callable with arguments of type (:: Game) and
# (:: Vector{Game}).
#
# The output of applying a model is a Vector where the first entry is the model
# prediction of the state value, and the policy_length(game) entries afterwards
# are the policy (expected to be normalized).

abstract type Model{G <: Game, GPU} <: Element{GPU} end

function apply(model :: Model{G, GPU}, game :: G) where {G, GPU}
  result = model(game) |> to_cpu
  result[1], result[2:end]
end

# Saving and loading models
# Maybe we should think about something more version-stable here,
# because this can break if AutoGrad, or Knet, or BSON, or Jtac changes

function save_model(fname :: String, model :: Model{G, false}) where {G}
  BSON.bson(fname * ".jtm", model = model)
end

function load_model(fname :: String)
  BSON.load(fname * ".jtm")[:model]
end

to_cpu(a :: Knet.KnetArray{Float32}) = convert(Array{Float32}, a)
to_cpu(a :: Array{Float32}) = a

to_gpu(a :: Knet.KnetArray{Float32}) = a
to_gpu(a :: Array{Float32}) = convert(Knet.KnetArray{Float32}, a)

# Some models will be able to harness asyncmap with more (or fewer) tasks
ntasks(:: Model) = 100

# For the training step, the model may be modified compared to the dataset
# creation
training_model(m) = nothing
training_model(m :: Model) = m

