# Neural network layers that can be used to create more complicated models

using AutoGrad
using Knet

# Auxiliary activation functions

id(x) = x
# need to give the primitive due to current AutoGrad limitations
@primitive id(x),dy (dy)


# Dense layer

struct Dense
  w  # Weight matrix 
  b  # Bias vector
  f  # Activation function
end

function Dense(i :: Int, o :: Int, f = relu, bias = true)
  if bias
    Dense(param(o,i), param0(o), f)
  else
    Dense(param(o,i), zeros(Float32, o), f)
  end
end

(d :: Dense)(x) = d.f.(d.w * mat(x) + d.b)


# Convolutional layer

struct Conv
  w  # Convolutional kernel
  b  # Bias vector
  f  # Activation function
end

function Conv(ci :: Int, co :: Int, f = relu; kernelsize :: Int = 3, bias = true)
  if bias
    Conv(param(kernelsize, kernelsize, ci, co), param0(1,1,co,1), f)
  else
    Conv(param(kernelsize, kernelsize, ci, co), zeros(Float32,1,1,co,1), f)
  end
end

(c :: Conv)(x) = c.f.(conv4(c.w, x) .+ c.b)

# Chaining of layers

struct Chain
  layers
end

Chain(layers...) = Chain(layers)

function (c :: Chain)(x)
  for l in c.layers
    x = l(x)
  end
  x
end
