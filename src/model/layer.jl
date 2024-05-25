
"""
Activation function that is applied to the output of a neural network layer.
"""
struct Activation{BC}
  f :: Function
end

@pack {<: Activation} in NamedValueFormat{Activation}


"""
    Activation(f, broadcast = true)

Wrap the function `f` as an activation function that does or does not support
broadcasting.
"""
Activation(f :: Function; broadcast = true) = Activation{broadcast}(f)

Base.convert(:: Type{Activation}, f :: Function) = Activation(f)

(f :: Activation{true})(args...) = f.f.(args...)
(f :: Activation{false})(args...) = f.f(args...)

"""
    activationname(activation)

Return the name of `activation`. If it is registered (see [`register!`](@ref)),
the registered name is returned. Otherwise, `Base.nameof(activation.f)` is
returned.
"""
function activationname(f :: Activation)
  if isregistered(Activation, f)
    lookupname(Activation, f)
  else
    Base.nameof(f.f)
  end
end

"""
Backend for neural network layers and models.

A backend implements forward and (optionally) backward passes of basic deep
learning operations. The underlying array type that the backend operators on,
like `Array{Float32}` or `CuArray{Float32}`, is indicated as type parameter.
"""
abstract type Backend{T} end

@pack {<: Backend} in NamedValueFormat{Backend}

"""
    istrainable(backend)

Check whether `backend` can be used for training or for inference only.
"""
istrainable(:: Backend) = false

"""
    arraytype(array)

Returns the general array type of `array`, i.e., forgets the dimension.
"""
arraytype(:: AbstractArray{F}) where {F} = error("not implemented")
arraytype(:: Array{F}) where {F} = Array{F}

"""
    arraytype(backend)

Returns the array type that the backend operates on.
"""
arraytype(:: Backend{T}) where {T} = T
arraytype(name :: Union{Symbol, String}) = arraytype(getbackend(name))

"""
Default backend implementation. Relies on NNlib.jl and supports forward passes
only.

This backend is used for saving / loading models. Other backends have to
implement conversions from / to `DefaultBackend`.
"""
struct DefaultBackend{T} <: Backend{T} end

function Base.show(io :: IO, :: DefaultBackend{T}) where {T}
  print(io, "DefaultBackend{$T}()")
end

"""
    getbackend(name)

Returns the neural network backend that has been registered under `name`.
"""
getbackend(name) = resolve(Backend, name)

"""
    getbackends()

Return a dictionary of all registered backends.
"""
getbackends() = lookup(Backend)

"""
    adapt(T, backend)

Change the array type of the neural layer backend `backend` to `T`.
"""
adapt(T, :: Backend) = error("Not implemented")
adapt(T, :: DefaultBackend) = DefaultBackend{T}()


"""
    io_neurons(dims...)

Returns the effective input and output lengths of a layer with size `dims`.
Used for parameter initialization.
"""
io_neurons(n_out :: Integer, n_in :: Integer) = n_in, n_out
io_neurons(dims :: Integer...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end])

"""
    kaiming([rng], dims...)

Create a `Float32` parameter array of size `dims` with kaiming initialization.
"""
function kaiming(rng, dims :: Integer...)
  std = Float32(sqrt(2 / first(io_neurons(dims...))))
  randn(rng, Float32, dims...) .* std
end

kaiming(dims :: Integer...) = kaiming(Random.default_rng(), dims...)

"""
Neural network layer.
"""
abstract type Layer{B <: Backend} end

@pack {<: Layer} in TypedFormat{MapFormat}

"""
Elementary neural network layer that is not composed of other layers.
"""
abstract type PrimitiveLayer{B <: Backend} <: Layer{B} end

"""
Composite neural network layer that is composed of other layers.
"""
abstract type CompositeLayer{B <: Backend} <: Layer{B} end

"""
    getbackend(layer)

Returns the backend of the neural layer `layer`.
"""
getbackend(:: Layer{B}) where {B} = B()

"""
    layers(layer)

Returns the layers that a composite layer `layer` is built upon.
"""
layers(:: CompositeLayer) = error("To be implemented")

"""
    isvalidinputsize(layer, insize)

Checks whether the tuple `insize` is a valid input size for the neural layer
`layer`.
"""
isvalidinputsize(:: Layer, :: Any) = error("Not implemented")

"""
    isvalidinput(layer, input)

Checks whether the size of `input` is valid via `isvalidinputsize`.
"""
isvalidinput(l :: Layer, x) = isvalidinputsize(l, size(x)[1:end-1]) # remove batch dim

"""
    outputsize(layer, insize)

Calculates the output size of `layer` if supplied with input of size `insize`.
"""
outputsize(:: Layer, :: Any) = error("Not implemented")

"""
    parameters(layer)

Return an iterable of (trainable) parameter arrays in layer.
"""
parameters(:: Layer) = error("Not implemented")

"""
    parametercount(layer)

Return the number of (trainable) parameters in `layer`.
"""
parametercount(layer) = sum(length, parameters(layer))


"""
    adapt(backend, layer)
    adapt(T, layer)

Adapt the neural layer `layer` to the backend `backend`. If an array type `T`
is passed, the backend of layer is adapted to `T`. Depending on the backend
implementation, this may change the type of layer.
"""
adapt(:: Backend, l :: Layer{<: Backend}) = error("Not implemented")
adapt(:: B, l :: Layer{B}) where {B <: Backend} = l

adapt(name :: Union{Symbol, AbstractString}, l) = adapt(getbackend(name), l)

function adapt(T :: Type{<: AbstractArray}, l :: Layer{B}) where {B}
  backend = adapt(T, B())
  adapt(backend, l)
end

"""
    releasememory!(array)

Mark the memory occupied by `array` as reusable.
"""
releasememory!(_) = nothing


"""
Dense neural network layer.
"""
struct Dense{T} <: PrimitiveLayer{DefaultBackend{T}}
  w :: T
  b :: T
  f :: Activation

  bias :: Bool # (trainable) bias?
end

@pack {<: Dense} (w, b) in BinArrayFormat

"""
    Dense(ni, no, f = identity; [bias, rng])

Create a `Dense` layer that accepts input from `ni` neurons and has `no`
output neurons, activated by `f`.

If `bias = true` (default), a bias vector is added before activation and can
be trained. A custom random number generator for weight initialization can be
passed via `rng`.
"""
function Dense( ni :: Int
              , no :: Int
              , f = identity
              ; bias = true
              , rng = Random.default_rng() )

  w = kaiming(rng, no, ni)
  b = zeros(Float32, no)
  f = resolve(Activation, f)
  Dense{Array{Float32}}(w, b, f, bias)
end

function (d :: Dense{T})(x :: T) where {T}
  @assert isvalidinput(d, x)
  tmp = d.w * reshape(x, :, size(x)[end])
  res = d.f(tmp .+ d.b)
  releasememory!(tmp)
  res
end

# Mutating interface
# function (d :: Dense{T})(out :: T, x :: T) where {T}
#   @assert isvalidinput(d, x)
#   insz = size(x)
#   outsz = (size(d.w, 1), insz[end])

#   x = reshape(x, :, insz[end])
#   out = reshape(out, :)
#   resize!(out, prod(outsz))
#   out = reshape(out, outsz)

#   mul!(out, d.w, x)
#   out .= d.f(out)
# end

isvalidinputsize(d :: Dense, s) = (prod(s) == size(d.w, 2))

function outputsize(d :: Dense, s)
  @assert isvalidinputsize(d, s) "Dense layer does not accept input of size $s"
  (size(d.w, 1),)
end

parameters(d :: Dense) = d.bias ? [d.w, d.b] : [d.w]

function adapt(:: DefaultBackend{T}, d :: Dense) where {T}
  Dense{T}(convert(T, d.w), convert(T, d.b), d.f, d.bias)
end

function Base.copy(d :: Dense{T}) where {T}
  Dense{T}(copy(d.w), copy(d.b), d.f, d.bias)
end

function Base.show(io :: IO, d :: Dense)
  print(io, "Dense($(size(d.w, 1)), $(activationname(d.f)))")
end

function Base.show(io :: IO, ::MIME"text/plain", d :: Dense{T}) where {T}
  n = size(d.w, 1)
  name = activationname(d.f)
  print(io, "Dense{$T} layer with $n neurons and $name activation")
end


"""
Convolutional neural network layer.
"""
struct Conv{T} <: PrimitiveLayer{DefaultBackend{T}}
  w :: T
  b :: T
  f :: Activation

  bias :: Bool          # (trainable) bias?
  p :: Tuple{Int, Int}  # pad
  s :: Tuple{Int, Int}  # stride
end

@pack {<: Conv} (w, b) in BinArrayFormat

"""
    Conv(ci, co, f = identity; window = 3, pad = 0, stride = 1, [bias, rng])

Create a `Conv` layer with `ci` input and `co` output channels, activated by
`f`.

If `bias = true` (default), a bias vector (along the output channel dimension)
is added before activation and can be trained. A custom random number generator
for weight initialization can be passed via `rng`.
"""
function Conv( ci :: Int
             , co :: Int
             , f = identity
             ; window = 3
             , pad = 0
             , stride = 1
             , bias = true
             , rng = Random.default_rng() )

  k = isa(window, Int) ? (window, window) : window
  p = isa(pad, Int) ? (pad, pad) : pad
  s = isa(stride, Int) ? (stride, stride) : stride

  w = kaiming(rng, k[1], k[2], ci, co)
  b = zeros(Float32, co)
  f = resolve(Activation, f)

  Conv{Array{Float32}}(w, b, f, bias, p, s)
end

function (c :: Conv{T})(x :: T) where {T}
  @assert isvalidinput(c, x)
  tmp = NNlib.conv(x, c.w, pad = c.p, stride = c.s)
  b = reshape(c.b, 1, 1, :, 1)
  res = c.f(tmp .+ b)
  releasememory!(tmp)
  res
end

# Mutating interface
# function (c :: Conv{T})(out :: T, x :: T) where {T}
#   @assert isvalidinput(d, x)
#   insz = size(x)
#   outsz = (outputsize(c, insz[1:end-1])..., insz[end])
#   resize!(out)
#   # TODO: use NNlib.conv!
#   tmp = NNlib.conv(x, c.w, pad = c.p, stride = c.s)
#   res = c.f(tmp .+ c.b)
#   releasememory!(tmp)
# end

parameters(d :: Conv) = d.bias ? [d.w, d.b] : [d.w]

function isvalidinputsize(c :: Conv, s)
  all([
    length(s) == 3,
    s[1] >= size(c.w, 1) - c.p[1],
    s[2] >= size(c.w, 2) - c.p[2],
    s[3] == size(c.w, 3),
  ])
end

function outputsize(c :: Conv, s)
  @assert isvalidinputsize(c, s) "Conv layer does not accept input of size $s"
  ( 1 + (div(s[1] + 2c.p[1] - size(c.w, 1), c.s[1]) |> floor),
    1 + (div(s[2] + 2c.p[2] - size(c.w, 2), c.s[2]) |> floor),
    size(c.w, 4),
  )
end

function adapt(:: DefaultBackend{T}, c :: Conv) where {T}
  Conv{T}(convert(T, c.w), convert(T, c.b), c.f, c.bias, c.p, c.s)
end

function Base.copy(c :: Conv{T}) where {T}
  Conv{T}(copy(c.w), copy(c.b), c.f, c.bias, c.p, c.s)
end

function Base.show(io :: IO, c :: Conv)
  window = size(c.w)[1:2]
  channels = size(c.w, 4)
  print(io, "Conv($channels, $window, $(activationname(c.f)))")
end

function Base.show(io :: IO, ::MIME"text/plain", c :: Conv{T}) where {T}
  name = activationname(c.f)
  oc = size(c.w)[4]
  window = size(c.w)[1:2]
  println(io, "Conv{$T} layer with $oc out-channels and $name activation:")
  println(io, " window:  $window")
  println(io, " pad: $(c.p)")
  print(io, " stride:  $(c.s)")
end


"""
Batch normalization layer.
"""
struct Batchnorm{T} <: PrimitiveLayer{DefaultBackend{T}}
  mean :: T
  var :: T
  bias :: T
  scale :: T
  f :: Activation
end

@pack {<: Batchnorm} (mean, var, bias, scale) in BinArrayFormat

"""
    Batchnorm(c, f = identity)

Create a `Batchnorm` layer that operates on `c` channels, activated by `f`.
"""
function Batchnorm(c :: Int, f = identity)
  mean = zeros(Float32, c)
  var = ones(Float32, c)
  bias = zeros(Float32, c)
  scale = ones(Float32, c)
  f = resolve(Activation, f)
  Batchnorm{Array{Float32}}(mean, var, bias, scale, f)
end

function bnsize(s)
  if length(s) == 1
    (s[1], 1)
  elseif length(s) == 3
    (1, 1, s[3], 1)
  else
    error("Batchnorm layer only handles input with 1 or 3 dimensions")
  end
end

function (b :: Batchnorm{T})(x :: T) where {T}
  @assert isvalidinput(b, x)

  F = eltype(T)
  sz = bnsize(size(x)[1:end-1])
  eps = F(1e-5)

  mean = reshape(b.mean, sz)
  var = reshape(b.var, sz)
  bias = reshape(b.bias, sz)
  scale = reshape(b.scale, sz)

  b.f(((x .- mean) ./ sqrt.(eps .+ var)) .* scale .+ bias)
end

isvalidinputsize(b :: Batchnorm, s) = length(b.mean) == prod(bnsize(s))
outputsize(:: Batchnorm, s) = s

parameters(b :: Batchnorm) = [b.bias, b.scale]

function adapt(:: DefaultBackend{T}, b :: Batchnorm) where {T}
  Batchnorm{T}(
    convert(T, b.mean),
    convert(T, b.var),
    convert(T, b.bias),
    convert(T, b.scale),
    b.f
  )
end

function Base.copy(b :: Batchnorm{T}) where {T}
  Batchnorm{T}(copy(b.mean), copy(b.var), copy(b.bias), copy(b.scale), b.f)
end

function Base.show(io :: IO, b :: Batchnorm)
  print(io, "Batchnorm($(activationname(b.f)))")
end

function Base.show(io :: IO, ::MIME"text/plain", b :: Batchnorm{T}) where {T}
  name = activationname(b.f)
  print(io, "Batchnorm{$T} layer with $name activation")
end


"""
Composite neural layer that sequentially evaluates a vector of layers.
"""
struct Chain{T} <: CompositeLayer{DefaultBackend{T}}
  layers :: Vector{Layer{DefaultBackend{T}}}
end

"""
    Chain(layers)

Create a chain of the given neural network layers `layers`.
"""
function Chain(layers :: AbstractVector{L}) where {T, L <: Layer{DefaultBackend{T}}}
  @assert length(layers) > 0 "Empty chains are invalid"
  layers = Layer{DefaultBackend{T}}[l for l in layers]
  Chain{T}(layers)
end

function (c :: Chain{T})(x :: T) where {T}
  @assert isvalidinput(c, x)

  xold = c.layers[1](x)
  for l in c.layers[2:end]
    x = l(xold)
    releasememory!(xold)
    xold = x
  end
  xold
end

layers(c :: Chain) = c.layers

function isvalidinputsize(c :: Chain, s)
  for l in c.layers
    valid = isvalidinputsize(l, s)
    !valid && return false
    s = outputsize(l, s)
  end
  true
end

function outputsize(c :: Chain, s)
  for l in c.layers
    s = outputsize(l, s)
  end
  s
end

parameters(c :: Chain) = mapreduce(parameters, vcat, c.layers)

function adapt(backend :: DefaultBackend{T}, c :: Chain) where {T}
  Chain{T}(adapt.(Ref(backend), c.layers))
end

Base.copy(c :: Chain{T}) where {T} = Chain{T}(copy.(c.layers))

showcomposite(io :: IO, l) = show(io, l)
showcomposite(io :: IO, mime, l) = show(io, mime, l)

function showcomposite(io :: IO, c :: CompositeLayer{DefaultBackend{T}}) where {T}
  name = nameof(typeof(c))
  ls = layers(c)
  if length(ls) > 2
    ls = [ls[1], nothing, ls[end]]
  end
  print(io, "$name($(length(ls)), ")
  for l in ls[1:end-1]
    name = l |> typeof |> nameof
    name == :Nothing ? print(io, "..") : print(io, name)
    print(io, " -> ")
  end
  name = ls[end] |> typeof |> nameof
  print(io, name)
  print(io, ")")
end

function showcomposite( io :: IO
                      , mime :: MIME"text/plain"
                      , c :: CompositeLayer{DefaultBackend{T}}
                      , indent ) where {T}

  name = nameof(typeof(c))
  ls = layers(c)
  istr = repeat(" ", indent)
  istrp = repeat(" ", indent + 1)
  print(io, istr)
  
  print(io, "$name{$T} with $(length(ls)) layer(s):")
  for l in ls
    println()
    if l isa PrimitiveLayer
      print(io, istrp)
      showcomposite(io, l)
    else
      showcomposite(io, mime, l, indent + 1)
    end
  end
end

function Base.show(io :: IO, c :: CompositeLayer{DefaultBackend{T}}) where {T}
  showcomposite(io, c)
end

function Base.show( io :: IO
                  , mime :: MIME"text/plain"
                  , c :: CompositeLayer{DefaultBackend{T}} ) where {T}
  showcomposite(io, mime, c, 0)
end

"""
Residual neural network layer.
"""
struct Residual{T} <: CompositeLayer{DefaultBackend{T}}
  chain :: Chain{T}
  f :: Activation
end

"""
    Residual(layers, f = identity)
    Residual(chain, f = identity)

Create a residual neural network layer that wraps a chain `chain` with
activation `f`. The chain must conserve the shape of the input.
"""
function Residual(chain :: Chain{T}, f = identity) where {T}
  f = resolve(Activation, f)
  Residual{T}(chain, f)
end

function Residual(layers :: AbstractVector{L}, f = identity) where {T, L <: Layer{DefaultBackend{T}}}
  f = resolve(Activation, f)
  Residual{T}(Chain(layers), f)
end

function (r :: Residual{T})(x :: T) where {T}
  @assert isvalidinput(r, x)
  tmp = r.chain(x)
  res = r.f(tmp .+ x)
  releasememory!(tmp)
  res
end

layers(r :: Residual) = layers(r.chain)

function isvalidinputsize(r :: Residual, s)
  valid = isvalidinputsize(r.chain, s)
  if valid
    outsz = outputsize(r.chain, s)
    @assert outsz == s "Residual chain is not shape-conserving: $outsz != $s."
  end
  valid
end

function outputsize(r :: Residual, s)
  outsz = outputsize(r.chain, s)
  @assert outsz == s "Residual chain is not shape-conserving: $outsz != $s."
  outsz
end

function adapt(backend :: DefaultBackend{T}, r :: Residual) where {T}
  Residual{T}(adapt(backend, r.chain), r.f)
end

parameters(r :: Residual) = parameters(r.chain)

Base.copy(r :: Residual{T}) where {T} = Residual{T}(copy(r.chain), r.f)


##
## Chain macro 
##

inputsize(t :: Int) = (t,)
inputsize(t :: Tuple) = t
inputsize(val :: Any) = size(val)

function composite_arguments(symbol :: Symbol, size)
  if symbol in (:Dense,)
    (prod(size),)
  elseif symbol in (:Conv,)
    (size[3],)
  elseif symbol in (:Batchnorm,)
    (size[end],)
  elseif symbol in (:Chain, :Residual)
    ()
  else
    error("$symbol is no supported layer constructor")
  end
end

function composite_macro_body(c, ex, layers...)

  s = :(Jtac.Model.inputsize($ex))

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

  # Check for activation function
  if layers[1] isa QuoteNode && layers[1].value isa Symbol
    activation = layers[1]
    layers = layers[2:end]
  else
    activation = nothing
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
    push!(body, :(local $ssym = Jtac.Model.outputsize($name, $ssym)))
  end

  name = gensym("l")

  # Prepare layer and keyword arguments
  layersex = Expr(:vect, names...)
  kwargsex = Expr(:vect, kwargs...)
  
  # Join all defined layers to a chain
  if !isnothing(activation)
    push!(body, :(local $name = $c($layersex, $activation; $kwargsex...)))
  else
    push!(body, :(local $name = $c($layersex; $kwargsex...)))
  end
  push!(body, :(@assert Jtac.Model.isvalidinputsize($name, $s) "Macro failed to respect input size. This should not happen."))
  push!(body, :($name))

  # Return the created block of code
  esc(Expr(:block, body...))
end


"""
    @chain(gametype, [kwoptions...], partial_layer_constructors...)

Macro to comfortably create chains that are consistent with `gametype`.

The layer constructors in the arguments following `gametype` are supposed to
lack the input dimensions, as they are auto-derived from the gametype and the
previous layers. Keyword options to the call to `Chain` in the macro can also be
given.

# Examples
```julia
# The following two calls will create the same networks
# Both of them are compatible with the game TicTacToe

Model.Chain([ Model.Conv(1, 32, :relu), Model.Dense(32, 50) ])
Model.@chain ToyGames.TicTacToe Conv(32, :relu) Dense(50)
```
"""
macro chain(ex, layers...)
  composite_macro_body(:(Jtac.Model.Chain), ex, layers...)
end

"""
    @residual(gametype, [kwoptions...], partial_layer_constructors...) 

Macro to comfortably create residual layers that are consistent with `gametype`.

See also [`Chain`](@ref).
"""
macro residual(ex, layers...)
  composite_macro_body(:(Jtac.Model.Residual), ex, layers...)
end

