# Neural network layers that can be used to create more complicated models

using Knet


# Dense layer

struct Dense
  w  # Weight matrix 
  b  # Bias vector
  f  # Activation function
end

Dense(i :: Int, o :: Int, f = relu) = Dense(param(o,i), param0(o), f)

(d :: Dense)(x) = d.f.(d.w * mat(x) + d.b)

