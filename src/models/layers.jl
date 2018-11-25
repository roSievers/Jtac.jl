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

(d :: Dense)(x) = d.f.(d.w * mat(x) + d.b)

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

