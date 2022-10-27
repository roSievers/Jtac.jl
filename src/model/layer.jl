
# -------- Layer ------------------------------------------------------------- # 

"""
Layers are the computational units of `NeuralModel`s. They come in primitive
(`PrimitiveLayer`, like `Dense` or `Conv`) and composite (`CompositeLayer`, like
`Chain`) forms. See the macros `@chain`, `@stack`, and `@residual` for the
convenient creation of composite layers.
"""
abstract type Layer{GPU} <: Element{GPU} end

Pack.@mappack Model.Layer
Pack.freeze(l :: Model.Layer) = to_cpu(l)

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

"""
    count_params(layer)

Count the number of free parameters in `layer`.
"""
count_params(layer :: Layer) = sum(length, Knet.params(layer))


# -------- Auxiliary Functions ----------------------------------------------- # 

# Convert (gpu :: Bool) to the underlying representing array type
 
atype(gpu :: Bool) = gpu ? Knet.KnetArray{Float32} : Array{Float32}

# Check if something is an AutoGrad param or not
 
is_param(p) = isa(p, Knet.Param)

# Function to copy (and convert) objects of type Knet.Param
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

# -------- Layer Weights ----------------------------------------------------- #

#const WeightData = Union{ Array{Float32}
#                        , Knet.Param{Array{Float32, N, D}} where {N, D}
#                        , CUDA.CuArray{Float32}
#                        , Knet.Param{CUDA.CuArray{Float32, N, D}} where {N, D} }

struct Weight
  data #:: WeightData
end

Pack.register(Weight)
Pack.@mappack Model.Weight

function Pack.destruct(w :: Weight)
  if w.data isa Knet.Param
    data = w.data.value
    bytes = reinterpret(UInt8, reshape(data, :))
    Dict("param" => true, "dims" => size(data), "data" => Pack.Bytes(bytes))
  else
    data = w.data
    bytes = reinterpret(UInt8, reshape(data, :))
    Dict("param" => false, "dims" => size(data), "data" => Pack.Bytes(bytes))
   end
end

function Pack.construct(:: Type{Weight}, d :: Dict)
  dims = Int.(d["dims"])
  data = reinterpret(Float32, d["data"])
  data = collect(reshape(data, dims...))
  if d["param"]
    Weight(Knet.Param(data))
  else
    Weight(data)
  end
end

Base.convert(:: Type{Weight}, data) = Weight(data)


# -------- Layer Activation -------------------------------------------------- #

const FUNCTIONS = Dict{String, Function}()

struct NamedFunction
  name :: String
  f :: Function
end

Pack.register(NamedFunction)
Pack.@mappack NamedFunction [:name]

NamedFunction(name :: String) = NamedFunction(name, FUNCTIONS[name])
NamedFunction(sym :: Symbol) = NamedFunction(String(sym), FUNCTIONS[String(sym)])

Base.convert(:: Type{NamedFunction}, name :: String) = NamedFunction(name)
Base.convert(:: Type{NamedFunction}, name :: Symbol) = NamedFunction(name)

name_function!(name :: String, f) = (FUNCTIONS[name] = f)

## 

name_function!("id", Knet.identity)
name_function!("elu", Knet.elu)
name_function!("relu", Knet.relu)
name_function!("selu", Knet.selu)
name_function!("sigm", Knet.sigm)
name_function!("tanh", Knet.tanh)
name_function!("softmax", Knet.softmax)

# -------- Pointwise --------------------------------------------------------- # 

"""
Neural network layer that applies a function pointwisely.
"""
struct Pointwise{GPU} <: PrimitiveLayer{GPU}
  a :: NamedFunction
end

Pack.register(Pointwise)

Pointwise(f = "id"; gpu = false) = Pointwise{gpu}(f)

(p :: Pointwise)(x) = p.a.f.(x)

swap(p :: Pointwise{GPU}) where {GPU} = Pointwise{!GPU}(p.a)
Base.copy(p :: Pointwise) = p

valid_insize(:: Pointwise, _) = true
outsize(p :: Pointwise, s) = s

function Base.show(io :: IO, l :: Pointwise)
  print(io, "Pointwise($(l.a.name))")
end

function Base.show(io :: IO, ::MIME"text/plain", l :: Pointwise{GPU}) where {GPU}
  at = GPU ? "GPU" : "CPU"
  print(io, "Pointwise{$at} $(l.a.name) layer")
end


# -------- Dense ------------------------------------------------------------- # 

"""
Dense neural network layer.
"""
struct Dense{GPU} <: PrimitiveLayer{GPU}
  w :: Weight     # Weight matrix 
  b :: Weight     # Bias vector
  a :: NamedFunction # Activation function
end

Pack.register(Dense)

function Dense( i :: Int, o :: Int, f = "id"; bias = true, gpu = false )
  at = atype(gpu)
  w = Knet.param(o, i, atype = at)
  b = bias ? Knet.param0(o, atype = at) : convert(at, zeros(Float32, o))
  Dense{gpu}(w, b, f)
end

(d :: Dense)(x) = d.a.f.(d.w.data * Knet.mat(x) .+ d.b.data)

function swap(d :: Dense{GPU}) where {GPU}
  at = atype(!GPU)
  w = copy_param(d.w.data; at)
  b = copy_param(d.b.data; at)
  Dense{!GPU}(w, b, d.a)
end

function Base.copy(d :: Dense{GPU}) where {GPU}
  Dense{GPU}(copy_param(d.w.data), copy_param(d.b.data), d.a)
end

# Check if some input of a given size s can be processed by the layer
# Note that the batchsize is not part of s, so s will usually have
# Dimension 1 (after dense layers) or 3 (before/after convolution layers)
valid_insize(d :: Dense, s) = (prod(s) == size(d.w.data, 2))

# Get the correct output size for a respective input size
function outsize(d :: Dense, s)
  @assert valid_insize(d, s) "Dense layer cannot be applied to input of size $s"
  size(d.w.data, 1)
end

function Base.show(io :: IO, d :: Dense)
  print(io, "Dense($(size(d.w.data, 1)), $(d.a.name))")
end

function Base.show(io :: IO, ::MIME"text/plain", d :: Dense{GPU}) where {GPU}
  n = size(d.w.data, 1)
  at = GPU ? "GPU" : "CPU"
  ac = d.a.name
  print(io, "Dense{$at} layer with $n neurons and $ac activation")
end


# ------- Convolution -------------------------------------------------------- # 

"""
Convolutional neural network layer.
"""
struct Conv{GPU} <: PrimitiveLayer{GPU}
  w :: Weight     # Convolutional kernel
  b :: Weight     # Bias vector
  a :: NamedFunction # Activation function

  p :: Tuple{Int, Int} # Padding for the convolution
  s :: Tuple{Int, Int} # Stride for the convolution
end

Pack.register(Conv)

function Conv( ci :: Int, co :: Int, f = "id";
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

(c :: Conv{false})(x) =
  c.a.f.(Knet.conv4(c.w.data, x, padding = c.p, stride = c.s) .+ c.b.data)

function (c :: Conv{true})(x)
  tmp = Knet.conv4(c.w.data, x, padding = c.p, stride = c.s)
  res = c.a.f.(tmp .+ c.b.data)
  if !(tmp isa AutoGrad.Result)
    Knet.KnetArrays.freeKnetPtr(tmp.ptr)
  end
  res
end

function swap(c :: Conv{GPU}) where {GPU}
  at = atype(!GPU)
  w = copy_param(c.w.data, at = at)
  b = copy_param(c.b.data, at = at)
  Conv{!GPU}(w, b, c.a, c.p, c.s)
end

function Base.copy(c :: Conv{GPU}) where {GPU}
  Conv{GPU}(copy_param(c.w.data), copy_param(c.b.data), c.a, c.p, c.s)
end

function valid_insize(c :: Conv, s)
  all(( length(s) == 3,
        s[1] >= size(c.w.data, 1) - c.p[1],
        s[2] >= size(c.w.data, 2) - c.p[2],
        s[3] == size(c.w.data, 3) ))
end

function outsize(c :: Conv, s)
  @assert valid_insize(c, s) "Conv layer cannot be applied to input of size $s"
  ( 1 + (div(s[1] + 2c.p[1] - size(c.w.data, 1), c.s[1]) |> floor),
    1 + (div(s[2] + 2c.p[2] - size(c.w.data, 2), c.s[2]) |> floor),
    size(c.w.data, 4)
  )
end

function Base.show(io :: IO, c :: Conv)
  window = size(c.w.data)[1:2]
  channels = size(c.w.data, 4)
  print(io, "Conv($channels, $window, $(c.a.name))")
end

function Base.show(io :: IO, ::MIME"text/plain", c :: Conv{GPU}) where {GPU}
  at = GPU ? "GPU" : "CPU"
  ac = c.a.name
  println(io, "Conv{$at} layer with $(size(c.w.data)[4]) out-channels and $ac activation:")
  println(io, " window:  $(size(c.w.data)[1:2])")
  println(io, " padding: $(c.p)")
  print(io, " stride:  $(c.s)")
end


# -------- Deconvolution ----------------------------------------------------- # 

"""
De-convolutional neural network layer.
"""
struct Deconv{GPU} <: PrimitiveLayer{GPU}
  w :: Weight     # Deconvolutional kernel
  b :: Weight     # Bias vector
  a :: NamedFunction # Activation function

  p :: Tuple{Int, Int} # Padding for the convolution
  s :: Tuple{Int, Int} # Stride for the convolution
end

Pack.register(Deconv)

function Deconv( ci :: Int, co :: Int, f = "id";
                 window = 3, padding = 0, 
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

(d :: Deconv)(x) =
  d.a.f.(Knet.deconv4(d.w.data, x, padding = d.p, stride = d.s) .+ d.b.data)

function swap(d :: Deconv{GPU}) where {GPU}
  at = atype(!GPU)
  w = copy_param(d.w.data, at = at)
  b = copy_param(d.b.data, at = at)
  Deconv{!GPU}(w, b, d.a, d.p, d.s)
end

function Base.copy(d :: Deconv{GPU}) where {GPU}
  Deconv{GPU}(copy_param(d.w.data), copy_param(d.b.data), d.a, d.p, d.s)
end

valid_insize(d :: Deconv, s) = (length(s) == 3)

function outsize(d :: Deconv, s)
  @assert valid_insize(d, s) "Deconv layer cannot be applied to input of size $s"
  ( size(d.w.data, 1) + d.s[1] * (s[1] - 1) - 2d.p[1],
    size(d.w.data, 2) + d.s[2] * (s[1] - 1) - 2d.p[2],
    size(d.w.data, 3) )
end

function Base.show(io :: IO, d :: Deconv)
  window = size(d.w.data)[1:2]
  channels = size(d.w.data, 3)
  print(io, "Deconv($channels, $window, $(d.a.name))")
end

function Base.show(io :: IO, ::MIME"text/plain", d :: Deconv{GPU}) where {GPU}
  at = GPU ? "GPU" : "CPU"
  ac = d.a.name
  println(io, "Deconv{$at} layer with $(size(d.w.data)[3]) out-channels and $ac activation:")
  println(io, " window:  $(size(d.w.data)[1:2])")
  println(io, " padding: $(d.p)")
  print(io, " stride:  $(d.s)")
end


# -------- Pooling ----------------------------------------------------------- # 

"""
Pooling layer.
"""
struct Pool{GPU} <: PrimitiveLayer{GPU}
  w :: Tuple{Int, Int} # Window size for the pooling operation
  p :: Tuple{Int, Int} # Padding
  s :: Tuple{Int, Int} # Stride
  a :: NamedFunction      # Activation function
end

Pack.register(Pool)

function Pool(f = "id"; window = 2, padding = 0, stride = window, gpu = false)
  w, p, s = expand_to_pair.((window, padding, stride))
  Pool{gpu}(w, p, s, f)
end

function (p :: Pool)(x)
  p.a.f.(Knet.pool(x, window = p.w, padding = p.p, stride = p.s))
end

swap(p :: Pool{GPU}) where {GPU} = Pool{!GPU}(p.w, p.p, p.s, p.a)
Base.copy(p :: Pool) = p

function valid_insize(p :: Pool, s)
  all(( length(s) == 3,
        s[1] >= p.w[1] - p.p[1],
        s[2] >= p.w[2] - p.p[2] ))
end

function outsize(p :: Pool, s)
  @assert valid_insize(p, s) "Pool layer cannot be applied to input of size $s"
  ( 1 + div(s[1] + 2p.p[1] - p.w[1], p.s[1]),
    1 + div(s[2] + 2p.p[2] - p.w[2], p.s[2]),
    s[3]
  )
end

function Base.show(io :: IO, p :: Pool)
  print(io, "Pool($(p.w), $(p.a.name))")
end

function Base.show(io :: IO, ::MIME"text/plain", p :: Pool{GPU}) where {GPU}
  at = GPU ? "GPU" : "CPU"
  ac = p.a.name
  println(io, "Pool{$at} layer with $ac activation:")
  println(io, " window:  $(p.w[1:2])")
  println(io, " padding: $(p.p)")
  print(io, " stride:  $(p.s)")
end


# -------- Dropout ----------------------------------------------------------- #

# Note that we try to recognize if we are training (dropout is active)
# or not. This can only be done if the Dropout layer is not the first layer
# that manipulates the weights

"""
Dropout layer. This must not be the first layer of the network.
"""
struct Dropout{GPU} <: PrimitiveLayer{GPU}
  prob :: Float32
  a :: NamedFunction
end

Pack.register(Dropout)

Dropout(prob, f = "id"; gpu = false) = Dropout{gpu}(prob, f)

function (d :: Dropout)(x)
  if isa(x, AutoGrad.Value)
    d.a.f.(Knet.dropout(x, d.prob))
  else
    d.a.f.(x)
  end
end

swap(d :: Dropout{GPU}) where {GPU} = Dropout{!GPU}(d.prob, d.a)
Base.copy(d :: Dropout) = d

valid_insize(:: Dropout, s) = true
outsize(:: Dropout, s) = s

function Base.show(io :: IO, d :: Dropout)
  print(io, "Dropout($(d.prob), $(d.a.name))")
end

function Base.show(io :: IO, ::MIME"text/plain", d :: Dropout{GPU}) where {GPU}
  at = GPU ? "GPU" : "CPU"
  print(io, "Dropout{$at} layer with prob $(d.prob) and $(d.a.name) activation")
end


# -------- Batch Normalization ----------------------------------------------- #

Pack.register(Knet.Ops20.BNMoments)
Pack.@mappack Knet.Ops20.BNMoments

Pack.destruct(bn :: Knet.Ops20.BNMoments) = Dict{String, Any}(
    "momentum" => bn.momentum
  , "mean" => isnothing(bn.mean) ? nothing : Model.Weight(bn.mean)
  , "var" => isnothing(bn.var) ? nothing : Model.Weight(bn.var)
)

function Pack.construct(:: Type{Knet.Ops20.BNMoments}, d) 
  mean = isnothing(d["mean"]) ? nothing : Pack.construct(Model.Weight, d["mean"]).data
  var = isnothing(d["var"]) ? nothing : Pack.construct(Model.Weight, d["var"]).data
  momentum = Float32(d["momentum"])
  Knet.bnmoments(; mean, var, momentum)
end


"""
Batch normalization layer.
"""
struct Batchnorm{GPU} <: PrimitiveLayer{GPU}
  moments :: Knet.Ops20.BNMoments
  params :: Weight
  a :: NamedFunction
end

Pack.register(Batchnorm)

function Batchnorm(channels, f = "id"; gpu = false)
  b = Batchnorm{false}( Knet.bnmoments()
                      , Knet.bnparams(Float32, channels)
                      , f )
  gpu ? swap(b) : b
end

(b :: Batchnorm)(x) = b.a.f.(Knet.batchnorm(x, b.moments, b.params.data))

(b :: Batchnorm{false})(x) = b.a.f.(Knet.batchnorm(x, b.moments, b.params.data))

function (b :: Batchnorm{true})(x)
  tmp = Knet.batchnorm(x, b.moments, b.params.data)
  res = b.a.f.(tmp)
#  Knet.KnetArrays.freeKnetPtr(tmp.ptr)
  res
end

function swap(b :: Batchnorm{GPU}) where {GPU}
  at = atype(!GPU)
  mean = (b.moments.mean != nothing) ? convert(at, b.moments.mean) : nothing
  var  = (b.moments.var != nothing) ? convert(at, b.moments.var) : nothing
  moments = Knet.bnmoments(momentum = b.moments.momentum, mean = mean, var = var)
  Batchnorm{!GPU}(moments, copy_param(b.params.data, at = at), b.a)
end

function Base.copy(b :: Batchnorm{GPU}) where {GPU}
  mean = (b.moments.mean != nothing) ? copy(b.moments.mean) : nothing
  var  = (b.moments.var  != nothing) ? copy(b.moments.var)  : nothing
  moments = Knet.bnmoments(momentum = b.moments.momentum, mean = mean, var = var)
  Batchnorm{GPU}(moments, copy_param(b.params.data), b.a)
end

valid_insize(:: Batchnorm, s) = true
outsize(:: Batchnorm, s) = s

function Base.show(io :: IO, b :: Batchnorm)
  print(io, "Batchnorm($(b.a.name))")
end

function Base.show(io :: IO, ::MIME"text/plain", b :: Batchnorm{GPU}) where {GPU}
  at = GPU ? "GPU" : "CPU"
  print(io, "Batchnorm{$at} layer with $(b.a.name) activation")
end


# -------- Chain ------------------------------------------------------------- #

"""
Composition of neural network layers.
"""
struct Chain{GPU} <: CompositeLayer{GPU}
  layers :: Vector{Layer{GPU}}
end

Pack.register(Chain)

function Chain(layers :: Layer{GPU}...; gpu = nothing) where {GPU} 
  c = Chain{GPU}(Layer{GPU}[l for l in layers])
  if !isnothing(gpu)
    gpu != GPU ? swap(c) : c
  else
    c
  end
end

function (c :: Chain{false})(x)
  for l in c.layers
    x = l(x)
  end
  x
end

function (c :: Chain{true})(x)
  xold = c.layers[1](x)
  for l in c.layers[2:end]
    x = l(xold)
    if !(xold isa AutoGrad.Result)
      Knet.KnetArrays.freeKnetPtr(xold.ptr)
    end
    xold = x
  end
  xold
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

function show_composite(name, io :: IO, layers)
  n = length(layers)
  if n == 0
    print(io, "$name(0)")
  else
    pp(layer) = isnothing(layer) ? print(io, "...") : show(io, layer)
    if n <= 2
      layers = layers
    else
      layers = [layers[1]; nothing; layers[end]]
    end
    print(io, "$name($n: ")
    for l in layers[1:end-1]
      pp(l);
      print(io, " -> ")
    end
    print(io, layers[end])
    print(io, ")")
  end
end

function show_composite(name, io :: IO, mime :: MIME"text/plain", layers, gpu, indent)
  n = length(layers)
  at = gpu ? "GPU" : "CPU"
  ind = join(repeat(" ", indent))
  ind2 = join(repeat(" ", indent+1))
  print(io, ind)
  if n == 1
    println(io, "$name{$at} with 1 layer:")
  else
    println(io, "$name{$at} with $n layers:")
  end
  for (i, layer) in enumerate(layers)
    if layer isa PrimitiveLayer
      print(io, ind2)
      show(io, layer)
    else
      show(io, mime, layer, indent+2)
    end
    if i != n
      println(io)
    end
  end
end

Base.show(io :: IO, c :: Chain) = show_composite("Chain", io, c.layers)

function Base.show(io :: IO, mime :: MIME"text/plain", c :: Chain{GPU}, indent = 0) where {GPU}
  show_composite("Chain", io, mime, c.layers, GPU, indent)
end

# -------- Stack ------------------------------------------------------------- # 

"""
A stack of neural network layers. It is similar to a chain, but all
intermediate layers are concatenated in the output of the stack.
"""
struct Stack{GPU} <: CompositeLayer{GPU}
  layers :: Vector{Layer{GPU}}
  stack_input :: Bool
end

Pack.register(Stack)

function Stack(layers :: Layer{GPU}...; 
               gpu = nothing, stack_input :: Bool = false) where {GPU}

  s = Stack{GPU}(Layer{GPU}[l for l in layers], stack_input)

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

Base.show(io :: IO, s :: Stack) = show_composite("Stack", io, s.layers)

function Base.show(io :: IO, mime :: MIME"text/plain", s :: Stack{GPU}, indent = 0) where {GPU}
  show_composite("Stack", io, mime, s.layers, GPU, indent)
end


# -------- Residual ---------------------------------------------------------- # 

"""
Residual block that wraps a chain. The input to the residual block is added
to the output of the chain (on the same input). Note that the chain must be
shape-conserving.
"""
struct Residual{GPU} <: CompositeLayer{GPU}
  chain :: Chain{GPU}
  a :: NamedFunction
end

Pack.register(Residual)

function Residual(layers :: Layer{GPU}...; f = "id", gpu = GPU) where {GPU}
  ls = (gpu == GPU) ? Layer[layers...] : Layer[swap.(layers)...]
  Residual(Chain(ls...), NamedFunction(f))
end

(r :: Residual{false})(x) = r.a.f.(r.chain(x) .+ x)

function (r :: Residual{true})(x)
  tmp = r.chain(x)
  res = r.a.f.(tmp .+ x)
  if !(tmp isa AutoGrad.Result)
    Knet.KnetArrays.freeKnetPtr(tmp.ptr)
  end
  res
end


swap(r :: Residual{GPU}) where {GPU} = Residual{!GPU}(swap(r.chain), r.a)
Base.copy(r :: Residual{GPU}) where {GPU} = Residual{GPU}(copy(r.chain), r.a)

function valid_insize(r :: Residual, s)
  valid = valid_insize(r.chain, s)
  if valid
    os = outsize(r.chain, s)
    os != s && error("Residual chain is not shape-conserving: $os != $s.")
  end
  valid
end

function outsize(r :: Residual, s)
  os = outsize(r.chain, s)
  os != s && error("Residual layer is not shape-conserving: $os != $s.")
  os
end

layers(r :: Residual) = layers(r.chain)

Base.show(io :: IO, r :: Residual) = show_composite("Residual", io, layers(r))

function Base.show(io :: IO, mime :: MIME"text/plain", r :: Residual{GPU}, indent = 0) where {GPU}
  show_composite("Residual", io, mime, layers(r), GPU, indent)
end

# -------- Chain/Stack Macro ------------------------------------------------- #

getsize(t :: Int) = t
getsize(t :: Tuple) = t
getsize(g :: AbstractGame) = size(g)
getsize(:: Type{G}) where {G <: AbstractGame} = size(G)

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

  s = :(Jtac.Model.getsize($ex))

  ssym = gensym("s")
  names = []
  body = Any[:(local $ssym = $s)]

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
      args = Expr(:call, :(Jtac.Model.composite_arguments), QuoteNode(ltype), ssym)
      insert!(layer.args, 2, :($args...))
    
    elseif isa(layer, Expr) && layer.head == :macrocall

      # Add input size argument
      insert!(layer.args, 3, ssym)

    end

    # Give the layer a temporary name
    name = gensym("l") 
    push!(names, name)

    layer.args[1] = :(Jtac.Model.$(layer.args[1]))

    # Add the evaluation of the completed layer constructor to the body
    push!(body, :(local $name = $(layer)))

    # Obtain the new input size
    push!(body, :(local $ssym = Jtac.Model.outsize($name, $ssym)))

  end

  name = gensym("l")
  # Join all defined layers to a chain
  push!(body, :(local $name = $c($(Expr(:vect, names...))...; $(Expr(:vect, kwargs...))...)))
  push!(body, :(@assert Jtac.Model.valid_insize($name, $s) "Macro failed to respect input size. This should not happen."))
  push!(body, :($name))

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
# The following two calls will create the same networks
# Both of them are compatible with the game TicTacToe

Model.Chain([ Model.Conv(1, 32, "relu"), Model.Dense(32, 50) ])
Model.@chain Game.TicTacToe Conv(32, "relu") Dense(50)
```
"""
macro chain(ex, layers...)
  composite_macro_body(:(Jtac.Model.Chain), ex, layers...)
end

"""
    @stack(gametype, [kwoptions...,] partial_layer_constructors...)

Stack macro that works analogously to `@chain`.
"""
macro stack(ex, layers...)
  composite_macro_body(:(Jtac.Model.Stack), ex, layers...)
end

"""
    @residual(gametype, [kwoptions...,] partial_layer_constructors...)

Macro to comfortably create residual blocks. It works like `@chain`.
"""
macro residual(ex, layers...)
  composite_macro_body(:(Jtac.Model.Residual), ex, layers...)
end

