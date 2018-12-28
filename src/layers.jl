
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
  w = Knet.param(o,i, atype = at)
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
               kernelsize :: Int = 3, padding = 0, 
               stride = 1, bias = true, gpu = false )

  at = Knet.atype(gpu)
  w = Knet.param(kernelsize, kernelsize, ci, co, atype = at)

  if bias
    b = Knet.param0(1, 1, co, 1, atype = at) 
  else
    b = convert(at, zeros(Float32, 1, 1, co, 1))
  end

  Conv{gpu}(w, b, f, padding, stride)

end

(c :: Conv)(x) = c.f.(Knet.conv4(c.w, x, padding = c.p, stride = c.s) .+ c.b)

function swap(c :: Conv{GPU}) where {GPU}
  at = atype(!GPU)
  w = copy_param(c.w, at = at)
  b = copy_param(c.b, at = at)
  Conv{!GPU}(w, b, c.f)
end

function Base.copy(c :: Conv{GPU}) where {GPU}
  Conv{GPU}(copy_param(c.w), copy_param(c.b), c.f)
end

expand_to_pair(t :: NTuple{2, Int}) = t
expand_to_pair(t :: Int) = (t, t)

function valid_insize(d :: Dense, s)
  p = expand_to_pair(d.p)
  all(( length(s) == 3,
        s[1] >= size(d.w, 1) - p[1],
        s[2] >= size(d.w, 2) - p[2],
        s[3] == size(d.w, 3) ))
end

function outsize(d :: Dense, insize)
  @assert valid_insize(d, s) "Layer cannot be applied to input of size $s"
  p, s = expand_to_pair.((d.p, d.s))
  ( 1 + (div(insize[1] + 2p[1] - size(d.w, 1), s[1]) |> floor),
    1 + (div(insize[2] + 2p[2] - size(d.w, 2), s[2]) |> floor),
    size(d.w, 4)
  )
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


# Batch normalization layer
# --------------------------------------------------------------------------- #

struct Batchnorm{GPU} <: Layer{GPU}
  moments
  params
end

Batchnorm(channels; gpu = false) = Batchnorm{gpu}(Knet.bnmoments(), Knet.bnparams(channels))

(b :: Batchnorm)(x) = Knet.batchnorm(x, b.moments, b.params)

function swap(b :: Batchnorm{GPU}) where {GPU}
  at = atype(!GPU)
  mean = (b.moments.mean != nothing) ? convert(at, b.moments.mean) : nothing
  var  = (b.moments.var != nothing) ? convert(at, b.moments.var) : nothing
  moments = bnmoments(momentum = b.moments.momentum, mean = mean, var = var)
  Batchnorm{!GPU}(b.moments, copy_params(b.params, at = at))
end

function Base.copy(b :: Batchnorm{GPU}) where {GPU}
  mean = (b.moments.mean != nothing) ? copy(b.moments.mean) : nothing
  var  = (b.moments.var  != nothing) ? copy(b.moments.var)  : nothing
  moments = bnmoments(momentum = b.moments.momentum, mean = mean, var = var)
  Batchnorm{GPU}(b.moments, copy_params(b.params))
end


