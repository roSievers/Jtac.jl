
# Layers
# --------------------------------------------------------------------------- # 

# A layer is a functional element that provides a parameterized mapping from
# data to features. Each subtype of Layer should be callable with 4-d arrays,
# where the last dimension stands for the batch.

abstract type Layer{GPU} <: Element{GPU} end


# Auxiliary functions
# --------------------------------------------------------------------------- # 

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


# Dense layers
# --------------------------------------------------------------------------- # 

struct Dense{GPU} <: Layer{GPU}
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


# Convolutional layers
# --------------------------------------------------------------------------- # 

struct Conv{GPU} <: Layer{GPU}
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

# De-convolutional layers
# --------------------------------------------------------------------------- # 

struct Deconv{GPU} <: Layer{GPU}
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


# Chaining of layers
# --------------------------------------------------------------------------- # 

struct Chain{GPU} <: Layer{GPU}
  layers
end

Chain(layers :: Layer{GPU}...) where {GPU} = Chain{GPU}(layers)

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


# Pooling
# --------------------------------------------------------------------------- # 

struct Pool{GPU} <: Layer{GPU}
  w  # Window size for the pooling operation
  p  # Padding
  s  # Stride
end

function Pool(window = 2; padding = 0, stride = window, gpu = false)
  w, p, s = expand_to_pair.((window, padding, stride))
  Pool{gpu}(w, p, s)
end

function (p :: Pool)(x)
  Knet.pool(x, window = p.w, padding = p.p, stride = p.s)
end

swap(p :: Pool{GPU}) where {GPU} = Pool{!GPU}(p.w, p.p, p.s)
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


# Dropout layer
# --------------------------------------------------------------------------- #

# Note that we try to recognize if we are training (dropout is active)
# or not. This can only be done if the Dropout layer is not the first layer
# that manipulates the weights

struct Dropout{GPU} <: Layer{GPU}
  prob
end

Dropout(prob = 0.5; gpu = false) = Dropout{gpu}(prob)

function (d :: Dropout)(x)
  if isa(x, AutoGrad.Value)
    Knet.dropout(x, d.prob)
  else
    x
  end
end

swap(d :: Dropout{GPU}) where {GPU} = Dropout{!GPU}(d.prob)
Base.copy(d :: Dropout) = d

valid_insize(:: Dropout, s) = true
outsize(:: Dropout, s) = s


# Batch normalization layer
# --------------------------------------------------------------------------- #

struct Batchnorm{GPU} <: Layer{GPU}
  moments
  params
end

function Batchnorm(channels; gpu = false)
  b = Batchnorm{false}(Knet.bnmoments(), Knet.bnparams(Float32, channels))
  gpu ? swap(b) : b
end

(b :: Batchnorm)(x) = Knet.batchnorm(x, b.moments, b.params)

function swap(b :: Batchnorm{GPU}) where {GPU}
  at = atype(!GPU)
  mean = (b.moments.mean != nothing) ? convert(at, b.moments.mean) : nothing
  var  = (b.moments.var != nothing) ? convert(at, b.moments.var) : nothing
  moments = Knet.bnmoments(momentum = b.moments.momentum, mean = mean, var = var)
  Batchnorm{!GPU}(b.moments, copy_param(b.params, at = at))
end

function Base.copy(b :: Batchnorm{GPU}) where {GPU}
  mean = (b.moments.mean != nothing) ? copy(b.moments.mean) : nothing
  var  = (b.moments.var  != nothing) ? copy(b.moments.var)  : nothing
  moments = Knet.bnmoments(momentum = b.moments.momentum, mean = mean, var = var)
  Batchnorm{GPU}(b.moments, copy_param(b.params))
end

valid_insize(:: Batchnorm, s) = true
outsize(:: Batchnorm, s) = s



# Chaining macro that automatically adapts input sizes
# --------------------------------------------------------------------------- #


function chainargs(symbol :: Symbol, size)
  if symbol in (:Dense,)
    (prod(size),)
  elseif symbol in (:Conv, :Deconv)
    (size[3],)
  elseif symbol in (:Batchnorm,)
    (size[end],)
  elseif symbol in (:Chain, :Pool, :Dropout)
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

macro chain(ex, layers...)

  # TODO: realize when the input is an expression that evaluates to a size
  # instead of a gametype
  if isa(ex, Symbol)
    s = :(size($ex))
  elseif  isa(ex, Expr)
    @assert ex.head == :tuple "Only integer tuples can specify input size"
    s = ex
  end

  names = []
  body = [:(local s = $s)]
  for (i, layer) in enumerate(layers)

    # Check if the layer is indeed a :call
    @assert isa(layer, Expr) && layer.head == :call

    # Add the missing input size argument
    ltype = :($(layer.args[1]))
    args = Expr(:call, :(Jtac.chainargs), QuoteNode(ltype), :s)
    insert!(layer.args, 2, :($args...))

    # Give the layer a temporary name
    name = Symbol("l$i")
    push!(names, name)

    # Add the evaluation of the completed layer constructor to the body
    push!(body, :(local $name = $(layer)))

    # Obtain the new input size
    push!(body, :(local s = outsize($name, s)))

  end

  # Join all defined layers to a chain
  push!(body, :(Chain($(Expr(:vect, names...))...)))

  # Return the created block of code
  esc(Expr(:block, body...))
end


