# Neural network layers that can be used to create more complex models

# Dense layers

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
  w = convert(at, value(d.w))
  b = convert(at, value(d.b))
  Dense{!GPU}(Param(w), is_param(d.b) ? Param(b) : b, d.f)
end

Base.copy(d :: Dense{GPU}) where {GPU} = Dense{GPU}(copy(d.w), copy(d.b), d.f)


# Convolutional layer

struct Conv{GPU} <: Layer{GPU}
  w  # Convolutional kernel
  b  # Bias vector
  f  # Activation function
end

function Conv(ci :: Int, co :: Int, f = relu; 
              kernelsize :: Int = 3, bias = true, gpu = false)
  at = atype(gpu)
  w = param(kernelsize, kernelsize, ci, co, atype = at)
  b = bias ? param0(1, 1, co, 1, atype = at) : convert(at, zeros(Float32, 1, 1, co, 1))
  Conv{gpu}(w, b, f)
end

(c :: Conv)(x) = c.f.(conv4(c.w, x) .+ c.b)

function swap(c :: Conv{GPU}) where {GPU}
  at = atype(!GPU)
  w = convert(at, value(c.w))
  b = convert(at, value(c.b))
  Conv{!GPU}(Param(w), is_param(c.b) ? Param(b) : b, c.f)
end

Base.copy(c :: Conv{GPU}) where {GPU} = Conv{GPU}(copy(c.w), copy(c.b), c.f)


# Chaining of layers

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

struct Batchnorm{GPU} <: Layer{GPU}
  moments
  params
end

Batchnorm(channels; gpu = false) = Batchnorm{gpu}(bnmoments(), bnparams(channels))

(b :: Batchnorm)(x) = batchnorm(x, b.moments, b.params)

swap(b :: Batchnorm{GPU}) where {GPU} = Batchnorm{!GPU}(b.moments, b.params)
Base.copy(b :: Batchnorm{GPU}) where {GPU} = Batchnorm{!GPU}(b.moments, b.params)


