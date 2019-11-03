
# -------- Layer ------------------------------------------------------------- # 

"""
Layers are the functional units for `NeuralNetworkModels`. They are composable
and, by the macros `@chain` and `@stack`, allow for the easy creation of neural
networks that are compatible to a given game.
"""
abstract type Layer{GPU} <: Element{GPU} end

valid_insize(:: Layer, _) = error("Not implemented")
outsize(:: Layer, _) = error("Not implemented")


# -------- Primitive / Composite --------------------------------------------- # 

"""
An atomic neural network operation.
"""
abstract type PrimitiveLayer{GPU} <: Layer{GPU} end

"""
A composed neural network operation
"""
abstract type CompositeLayer{GPU} <: Layer{GPU} end

"""
    layers(clayer)

Get all neural network layers of an the composite layer `clayer`.
"""
layers( :: CompositeLayer) = error("Not implemented")


# -------- Auxiliary Functions ----------------------------------------------- # 

# Convert (gpu :: Bool) to the underlying representing array type
 
atype(gpu :: Bool) = gpu ? Knet.KnetArray{Float32} : Array{Float32}

# Check if something is a AutoGrad param or not
 
is_param(p) = isa(p, Knet.Param)

# Function to copy (and convert) objects of type Param
# For convenience, we allow this function to be applied to other objects as well

function copy_param(p :: Knet.Param; at = nothing)

  if at == nothing || isa(Knet.value(p), at)
    val = copy(Knet.value(p))
  else
    val = convert(at, Knet.value(p))
  end

  param = Knet.Param(val)
  param.opt = p.opt
  param

end

function copy_param(p; at = nothing)
  if at == nothing || isa(p, at)
    copy(p)
  else
    convert(at, p)
  end
end

# Auxiliary function used for the constructor of some layers

expand_to_pair(t :: NTuple{2, Int}) = t
expand_to_pair(t :: Int) = (t, t)


# -------- Pointwise --------------------------------------------------------- # 

"""
Neural network layer that applies a function pointwisely.
"""
struct Pointwise{GPU} <: PrimitiveLayer{GPU}
  f # Activation function
end

Pointwise( f = Knet.identity; gpu = false ) = Pointwise{gpu}(f)

(p :: Pointwise)(x) = p.f.(x)

swap(p :: Pointwise{GPU}) where {GPU} = Pointwise(p.f, gpu = !GPU)
Base.copy(p :: Pointwise) = p

valid_insize(:: Pointwise, _) = true
outsize(p :: Pointwise, s) = s


# -------- Dense ------------------------------------------------------------- # 

"""
Dense neural network layer.
"""
struct Dense{GPU} <: PrimitiveLayer{GPU}
  w  # Weight matrix 
  b  # Bias vector
  f  # Activation function
end

function Dense( i :: Int, o :: Int, f = Knet.identity; 
                bias = true, gpu = false )
  at = atype(gpu)
  w = Knet.param(o, i, atype = at)
  b = bias ? Knet.param0(o, atype = at) : convert(at, zeros(Float32, o))
  Dense{gpu}(w, b, f)
end

(d :: Dense)(x) = d.f.(d.w * Knet.mat(x) .+ d.b)

function swap(d :: Dense{GPU}) where {GPU}
  at = atype(!GPU)
  w = copy_param(d.w, at = at)
  b = copy_param(d.b, at = at)
  Dense{!GPU}(w, b, d.f)
end

function Base.copy(d :: Dense{GPU}) where {GPU}
  Dense{GPU}(copy_param(d.w), copy_param(d.b), d.f)
end

# Check if some input of a given size s can be processed by the layer
# Note that the batchsize is not part of s, so s will usually have
# Dimension 1 (after dense layers) or 3 (before/after convolution layers)
valid_insize(d :: Dense, s) = (prod(s) == size(d.w, 2))

# Get the correct output size for a respective input size
function outsize(d :: Dense, s)
  @assert valid_insize(d, s) "Layer cannot be applied to input of size $s"
  size(d.w, 1)
end


# ------- Convolution -------------------------------------------------------- # 

"""
Convolutional neural network layer.
"""
struct Conv{GPU} <: PrimitiveLayer{GPU}
  w  # Convolutional kernel
  b  # Bias vector
  f  # Activation function

  p  # Padding for the convolution
  s  # Stride for the convolution
end

function Conv( ci :: Int, co :: Int, f = Knet.identity; 
               window = 3, padding = 0, 
               stride = 1, bias = true, gpu = false )

  k = expand_to_pair(window)
  p = expand_to_pair(padding)
  s = expand_to_pair(stride)

  at = atype(gpu)
  w = Knet.param(k[1], k[2], ci, co, atype = at)

  if bias
    b = Knet.param0(1, 1, co, 1, atype = at) 
  else
    b = convert(at, zeros(Float32, 1, 1, co, 1))
  end

  Conv{gpu}(w, b, f, p, s)

end

(c :: Conv)(x) = c.f.(Knet.conv4(c.w, x, padding = c.p, stride = c.s) .+ c.b)

function swap(c :: Conv{GPU}) where {GPU}
  at = atype(!GPU)
  w = copy_param(c.w, at = at)
  b = copy_param(c.b, at = at)
  Conv{!GPU}(w, b, c.f, c.p, c.s)
end

function Base.copy(c :: Conv{GPU}) where {GPU}
  Conv{GPU}(copy_param(c.w), copy_param(c.b), c.f, c.p, c.s)
end

function valid_insize(c :: Conv, s)
  all(( length(s) == 3,
        s[1] >= size(c.w, 1) - c.p[1],
        s[2] >= size(c.w, 2) - c.p[2],
        s[3] == size(c.w, 3) ))
end

function outsize(c :: Conv, s)
  @assert valid_insize(c, s) "Layer cannot be applied to input of size $s"
  ( 1 + (div(s[1] + 2c.p[1] - size(c.w, 1), c.s[1]) |> floor),
    1 + (div(s[2] + 2c.p[2] - size(c.w, 2), c.s[2]) |> floor),
    size(c.w, 4)
  )
end


# -------- Deconvolution ----------------------------------------------------- # 

"""
De-convolutional neural network layer.
"""
struct Deconv{GPU} <: PrimitiveLayer{GPU}
  w  # Deconvolutional kernel
  b  # Bias vector
  f  # Activation function

  p  # Padding for the convolution
  s  # Stride for the convolution
end

function Deconv( ci :: Int, co :: Int, f = Knet.identity; 
                 window :: Int = 3, padding = 0, 
                 stride = 1, bias = true, gpu = false )

  k = expand_to_pair(window)
  p = expand_to_pair(padding)
  s = expand_to_pair(stride)

  at = atype(gpu)
  w = Knet.param(k[1], k[2], co, ci, atype = at)

  if bias
    b = Knet.param0(1, 1, co, 1, atype = at) 
  else
    b = convert(at, zeros(Float32, 1, 1, co, 1))
  end

  Deconv{gpu}(w, b, f, p, s)

end

(d :: Deconv)(x) = d.f.(Knet.deconv4(d.w, x, padding = d.p, stride = d.s) .+ d.b)

function swap(d :: Deconv{GPU}) where {GPU}
  at = atype(!GPU)
  w = copy_param(d.w, at = at)
  b = copy_param(d.b, at = at)
  Deconv{!GPU}(w, b, d.f, d.p, d.s)
end

function Base.copy(d :: Deconv{GPU}) where {GPU}
  Deconv{GPU}(copy_param(d.w), copy_param(d.b), d.f, d.p, d.s)
end

valid_insize(d :: Deconv, s) = (length(s) == 3)

function outsize(d :: Deconv, s)
  @assert valid_insize(d, s) "Layer cannot be applied to input of size $s"
  ( size(d.w, 1) + d.s[1] * (s[1] - 1) - 2d.p[1],
    size(d.w, 2) + d.s[2] * (s[1] - 1) - 2d.p[2],
    size(d.w, 3) )
end


# -------- Pooling ----------------------------------------------------------- # 

"""
Pooling layer.
"""
struct Pool{GPU} <: PrimitiveLayer{GPU}
  w  # Window size for the pooling operation
  p  # Padding
  s  # Stride
  f  # Activation function
end

function Pool(f = Knet.identity; 
              window = 2, padding = 0, stride = window, gpu = false)
  w, p, s = expand_to_pair.((window, padding, stride))
  Pool{gpu}(w, p, s, f)
end

function (p :: Pool)(x)
  p.f.(Knet.pool(x, window = p.w, padding = p.p, stride = p.s))
end

swap(p :: Pool{GPU}) where {GPU} = Pool{!GPU}(p.w, p.p, p.s, p.f)
Base.copy(p :: Pool) = p

function valid_insize(p :: Pool, s)
  all(( length(s) == 3,
        s[1] >= p.w[1] - p.p[1],
        s[2] >= p.w[2] - p.p[2] ))
end

function outsize(p :: Pool, s)
  @assert valid_insize(p, s) "Layer cannot be applied to input of size $s"
  ( 1 + div(s[1] + 2p.p[1] - p.w[1], p.s[1]),
    1 + div(s[2] + 2p.p[2] - p.w[2], p.s[2]),
    s[3]
  )
end


# -------- Dropout ----------------------------------------------------------- #

# Note that we try to recognize if we are training (dropout is active)
# or not. This can only be done if the Dropout layer is not the first layer
# that manipulates the weights

"""
Dropout layer. This must not be the first layer of the network.
"""
struct Dropout{GPU} <: PrimitiveLayer{GPU}
  prob
  f
end

Dropout(prob, f = Knet.identity; gpu = false) = Dropout{gpu}(prob, f)

function (d :: Dropout)(x)
  if isa(x, AutoGrad.Value)
    d.f.(Knet.dropout(x, d.prob))
  else
    d.f.(x)
  end
end

swap(d :: Dropout{GPU}) where {GPU} = Dropout{!GPU}(d.prob, d.f)
Base.copy(d :: Dropout) = d

valid_insize(:: Dropout, s) = true
outsize(:: Dropout, s) = s


# -------- Batch Normalization ----------------------------------------------- #

"""
Batch normalization layer.
"""
struct Batchnorm{GPU} <: PrimitiveLayer{GPU}
  moments
  params
  f
end

function Batchnorm(channels, f = Knet.identity; gpu = false)
  b = Batchnorm{false}(Knet.bnmoments(), Knet.bnparams(Float32, channels), f)
  gpu ? swap(b) : b
end

(b :: Batchnorm)(x) = b.f.(Knet.batchnorm(x, b.moments, b.params))

function swap(b :: Batchnorm{GPU}) where {GPU}
  at = atype(!GPU)
  mean = (b.moments.mean != nothing) ? convert(at, b.moments.mean) : nothing
  var  = (b.moments.var != nothing) ? convert(at, b.moments.var) : nothing
  moments = Knet.bnmoments(momentum = b.moments.momentum, mean = mean, var = var)
  Batchnorm{!GPU}(moments, copy_param(b.params, at = at), b.f)
end

function Base.copy(b :: Batchnorm{GPU}) where {GPU}
  mean = (b.moments.mean != nothing) ? copy(b.moments.mean) : nothing
  var  = (b.moments.var  != nothing) ? copy(b.moments.var)  : nothing
  moments = Knet.bnmoments(momentum = b.moments.momentum, mean = mean, var = var)
  Batchnorm{GPU}(moments, copy_param(b.params), b.f)
end

valid_insize(:: Batchnorm, s) = true
outsize(:: Batchnorm, s) = s


# -------- Chain ------------------------------------------------------------- # 

"""
Composition of neural network layers.
"""
struct Chain{GPU} <: CompositeLayer{GPU}
  layers
end

function Chain(layers :: Layer{GPU}...; gpu = nothing) where {GPU} 
  c = Chain{GPU}(Layer[l for l in layers])
  if !isnothing(gpu)
    gpu != GPU ? swap(c) : c
  else
    c
  end
end

function (c :: Chain)(x)
  for l in c.layers
    x = l(x)
  end
  x
end

swap(c :: Chain{GPU}) where {GPU} = Chain(swap.(c.layers)...)
Base.copy(c :: Chain{GPU}) where {GPU} = Chain(copy.(c.layers)...)

valid_insize(c :: Chain, s) = valid_insize(c.layers[1], s)

function outsize(c :: Chain, s)
  for l in c.layers
    s = outsize(l, s)
  end
  s
end

layers(c :: Chain) = c.layers


# -------- Stack ------------------------------------------------------------- # 

"""
A stack of neural network layers. It is similar to a chain, but all
intermediate layers are concatenated in the output of the stack.
"""
struct Stack{GPU} <: CompositeLayer{GPU}
  layers
  stack_input :: Bool
end

function Stack(layers :: Layer{GPU}...; 
               gpu = nothing, stack_input :: Bool = false) where {GPU}

  s = Stack{GPU}(Layer[l for l in layers], stack_input)

  if !isnothing(gpu)
    gpu != GPU ? swap(s) : s
  else
    s
  end

end

function (s :: Stack{GPU})(x) where {GPU}

  batchsize = size(x)[end]

  features = s.stack_input ? Any[x] : Any[]

  for layer in s.layers
    x = layer(x)
    push!(features, x)
  end

  features = map(features) do data
    reshape(data, (prod(size(data)[1:end-1]), batchsize))
  end

  vcat(features...)

end


function swap(s :: Stack{GPU}) where {GPU}
  Stack(swap.(s.layers)..., stack_input = s.stack_input)
end

function Base.copy(s :: Stack{GPU}) where {GPU}
  Stack(copy.(s.layers)..., stack_input = s.stack_input)
end

valid_insize(stack :: Stack, s) = valid_insize(stack.layers[1], s)

function outsize(stack :: Stack, s)

  shapes  = []
  lengths = Int[]

  if stack.stack_input
    push!(shapes, s)
    push!(lengths, prod(s))
  end

  for layer in stack.layers
    @assert valid_insize(layer, s)
    s = outsize(layer, s)
    push!(shapes, s)
    push!(lengths, prod(s))
  end

  sum(lengths)
end

layers(s :: Stack) = s.layers


# -------- Residual ---------------------------------------------------------- # 

"""
Residual block that wraps a chain. The input to the residual block is added
to the output of the chain (on the same input). Note that the chain must be
shape-conserving.
"""
struct Residual{GPU} <: CompositeLayer{GPU}
  chain :: Chain{GPU}
  function Residual(c :: Chain{GPU}; gpu = GPU) where {GPU}
    new{gpu}(gpu == GPU ? c : swap(c))
  end
end

function Residual(layers :: Layer{GPU}...; gpu = GPU) where {GPU}
  ls = (gpu == GPU) ? Layer[layers...] : Layer[swap.(layers)...]
  Residual{gpu}(Chain(ls...))
end

(r :: Residual)(x) = r.chain(x) .+ x

swap(r :: Residual{GPU}) where {GPU} = Residual(swap(r.layer))
Base.copy(r :: Residual{GPU}) where {GPU} = Residual(copy(r.layer))

function valid_insize(r :: Residual, s)
  outsize(r.chain, s) != s && error("Residual layer is ill-constructed")
  valid_insize(r.chain, s)
end

function outsize(r :: Residual, s)
  os = outsize(r.chain, s)
  os != s && error("Residual layer is ill-constructed")
  os
end

layers(r :: Residual) = layers(r.chain)

# -------- Chain/Stack Macro ------------------------------------------------- #

getsize(t :: Int) = t
getsize(t :: Tuple) = t
getsize(g :: Game) = size(g)
getsize(:: Type{G}) where {G <: Game} = size(G)

function composite_arguments(symbol :: Symbol, size)
  if symbol in (:Dense,)
    (prod(size),)
  elseif symbol in (:Conv, :Deconv)
    (size[3],)
  elseif symbol in (:Batchnorm,)
    (size[end],)
  elseif symbol in (:Chain, :Stack, :Pool, :Dropout, :Pointwise)
    ()
  else
    error("$symbol is no valid layer constructor")
  end
end

# Take the size of a game / a game type itself as first argument,
# and take a tuple of "partial" layer constructors as second argument.
# The macro will calculate and prepend the constructor arguments that relate to
# the input size for each layer consecutively.
#
# Example:
#
#   @chain (9,9,1) (Conv(10), Dense(50))
#
# will call
#
#   Chain(Conv(1, 10), Dense(490, 50))
#
# such that the two layers are compatible.

function composite_macro_body(c, ex, layers...)

  s = :(Jtac.getsize($ex))

  ssym = gensym("s")
  names = []
  body = [:(local $ssym = $s)]

  # Expand begin-end blocks
  layers = mapreduce(vcat, layers) do layer
    if isa(layer, Expr) && layer.head == :block
      filter(x -> !isa(x, LineNumberNode), layer.args)
    else
      [layer]
    end
  end

  # Catch keyword arguments
  kwidx = findall(layers) do layer
    isa(layer, Expr) && layer.head == :(=)
  end

  # Convert them to real keyword arguments
  kwargs = map(layers[kwidx]) do kw
    Expr(:call, :(=>), QuoteNode(kw.args[1]), kw.args[2])
  end

  # Extract network layers
  layers = layers[setdiff(1:length(layers), kwidx)]


  if length(layers) == 1 && isa(layers[1], Expr) && layers[1].head == :block
    layers = filter(x -> !isa(x, LineNumberNode), layers[1].args)
  end

  for layer in layers

    # The "layer" is assumed to be a
    #   1. :call Expression
    #   2. :macrocall Expression
    #   3. Symbol
    #   4. Keyword argument for the chain call
    #
    # If it is a call, we may insert additional input size information
    # via Jtac.composite_arguments. If it is a macrocall, we give the macro a
    # new first argument, the total input size. If it is a symbol, we
    # leave it as it is.

    if isa(layer, Expr) && layer.head == :call
      
      # Maybe add missing input size argument
      ltype = :($(layer.args[1]))
      args = Expr(:call, :(Jtac.composite_arguments), QuoteNode(ltype), ssym)
      insert!(layer.args, 2, :($args...))
    
    elseif isa(layer, Expr) && layer.head == :macrocall

      # Add input size argument
      insert!(layer.args, 3, ssym)

    end

    # Give the layer a temporary name
    name = gensym("l") 
    push!(names, name)

    # Add the evaluation of the completed layer constructor to the body
    push!(body, :(local $name = $(layer)))

    # Obtain the new input size
    push!(body, :(local $ssym = outsize($name, $ssym)))

  end

  # Join all defined layers to a chain
  push!(body, :($c($(Expr(:vect, names...))...; $(Expr(:vect, kwargs...))...)))

  # Return the created block of code
  esc(Expr(:block, body...))

end

"""
    @chain(gametype, [kwoptions...,] partial_layer_constructors...)

Macro to comfortably create chains that are consistent with `gametype`.

The layer constructors in the arguments following `gametype` are supposed to
lack the input dimensions, as they are auto-derived from the gametype and the
previous layers. Keyword options to the call to `Chain` in the macro can also be
given.

# Examples
```julia
# The following two calls will create the same networks that are compatible
# with TicTacToe

Chain([ Conv(1, 32, relu), Dense(32, 50) ])
@chain TicTacToe Conv(32, relu) Dense(50)
```
"""
macro chain(ex, layers...)
  composite_macro_body(:(Jtac.Chain), ex, layers...)
end

"""
    @stack(gametype, [kwoptions...,] partial_layer_constructors...)

Stack macro that works analogously to `@chain`.
"""
macro stack(ex, layers...)
  composite_macro_body(:(Jtac.Stack), ex, layers...)
end

"""
    @residual(gametype, [kwoptions...,] partial_layer_constructors...)

Macro to comfortably create residual blocks.
"""
macro residual(ex, layers...)
  # TODO: Check consistency!!!
  composite_macro_body(:(Jtac.Residual), ex, layers...)
end
