
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

abstract type Model{GPU} <: Element{GPU} end

function apply(model :: Model, game :: Game)
  result = model(game)
  result[1], result[2:end]
end

is_compatible(:: Model, :: Game) = error("Not implemented")


# A layer is a functional element that provides a parameterized mapping from
# data to features. Each subtype of Layer should be callable with 4-d arrays,
# where the last dimension indicates the batch.

abstract type Layer{GPU} <: Element{GPU} end


# Auxiliary functions

# Convert (gpu :: Bool) to the underlying representing array type
atype(gpu :: Bool) = gpu ? KnetArray{Float32} : Array{Float32}

# Check if something is a AutoGrad param or not
is_param(array) = typeof(array) <: Param

# Fix for failure to copy param, issue #102 in AutoGrad.jl
Base.copy(p :: Param) = Param(copy(value(p)))


