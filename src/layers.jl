
# Layers
# --------------------------------------------------------------------------- # 

# A layer is a functional element that provides a parameterized mapping from
# data to features. Each subtype of Layer should be callable with 4-d arrays,
# where the last dimension stands for the batch.

abstract type Layer{GPU} <: Element{GPU} end


# Auxiliary functions
# --------------------------------------------------------------------------- # 

# Convert (gpu :: Bool) to the underlying representing array type
 
atype(gpu :: Bool) = gpu ? KnetArray{Float32} : Array{Float32}

# Check if something is a AutoGrad param or not
 
is_param(p) = isa(p, Param)

# Function to copy (and convert) objects of type Param
# For convenience, we allow this function to be applied to other objects as well

function copy_param(p :: Param; at = nothing)

  if at == nothing || isa(value(p), at)
    val = copy(value(p))
  else
    val = convert(at, value(p))
  end

  param = Param(val)
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

function Dense(i :: Int, o :: Int, f = relu; bias = true, gpu = false)
  at = atype(gpu)
  w = param(o,i, atype = at)
  b = bias ? param0(o, atype = at) : convert(at, zeros(Float32, o))
  Dense{gpu}(w, b, f)
end

(d :: Dense)(x) = d.f.(d.w * mat(x) .+ d.b)

function swap(d :: Dense{GPU}) where {GPU}
  at = atype(!GPU)
  w = copy_param(d.w, at = at)
  b = copy_param(d.b, at = at)
  Dense{!GPU}(w, b, d.f)
end

function Base.copy(d :: Dense{GPU}) where {GPU}
  Dense{GPU}(copy_param(d.w), copy_param(d.b), d.f)
end


# Convolutional layers
# --------------------------------------------------------------------------- # 

struct Conv{GPU} <: Layer{GPU}
  w  # Convolutional kernel
  b  # Bias vector
  f  # Activation function
end

function Conv(ci :: Int, co :: Int, f = relu; 
              kernelsize :: Int = 3, bias = true, gpu = false)

  at = atype(gpu)
  w = param(kernelsize, kernelsize, ci, co, atype = at)

  if bias
    b = param0(1, 1, co, 1, atype = at) 
  else
    b = convert(at, zeros(Float32, 1, 1, co, 1))
  end

  Conv{gpu}(w, b, f)

end

(c :: Conv)(x) = c.f.(conv4(c.w, x) .+ c.b)

function swap(c :: Conv{GPU}) where {GPU}
  at = atype(!GPU)
  w = copy_param(c.w, at = at)
  b = copy_param(c.b, at = at)
  Conv{!GPU}(w, b, c.f)
end

function Base.copy(c :: Conv{GPU}) where {GPU}
  Conv{GPU}(copy_param(c.w), copy_param(c.b), c.f)
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
    dropout(x, d.prob)
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

Batchnorm(channels; gpu = false) = Batchnorm{gpu}(bnmoments(), bnparams(channels))

(b :: Batchnorm)(x) = batchnorm(x, b.moments, b.params)

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


