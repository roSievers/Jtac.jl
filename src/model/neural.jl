
"""
Tensorizors are singleton types that define how a game state is converted to its
array representation. This is needed for models that operate on tensorized
versions of the game state, like [`NeuralModel`](@ref).

Games that have a canonical array representation only need to specialize
the behavior of [`DefaultTensorizor`](@ref). To support multiple array
representations for a single game type, a custom singleton subtype of
[`Tensorizer`](@ref) must be defined.
"""
abstract type Tensorizor{G <: AbstractGame} end

@pack {T <: Tensorizor} in TypedFormat{MapFormat}

Base.size(:: Tensorizor) = error("Not implemented")

"""
    buffer([T], t :: Tensorizor, batchsize)

Returns a zero-initialized array of type `T` that can hold `batchsize` game
states in the representation of `t`.
"""
function buffer(T :: Type{<: AbstractArray}, t :: Tensorizor, batchsize)
  buf = zeros(Float32, size(t)..., batchsize)
  convert(T, buf)
end

buffer(t :: Tensorizor, batchsize) = buffer(Array{Float32}, t, batchsize)

"""
    (t :: Tensorizor)([T,] game)
    (t :: Tensorizor)([T,] games)
    (t :: Tensorizor)(buffer, games)

Use `t` to create an array representation of `game` or `games` with array type `T`.
If a suitable buffer is passed, the representation is written into `buffer`.

See [`Model.buffer`](@ref) for the creation of buffers with the correct size.
"""
function (t :: Tensorizor)(buf, games)
  @assert !isempty(games) """
  Cannot produce array representation of empty game vector.
  """
  @assert size(buf)[1:3] == size(t) """
  Buffer size is incompatible with tensorizor $t.
  """
  @assert size(buf, 4) >= length(games) """
  Buffer size too small to store $(length(games)) game states.
  """
  T = arraytype(buf)
  t(T, buf, games)
end

function (t :: Tensorizor)(T :: Type{<: AbstractArray}, games :: Vector)
  buf = buffer(T, t, length(games))
  t(buf, games)
  buf
end

function (t :: Tensorizor)(T :: Type{<: AbstractArray}, game :: AbstractGame)
  dropdims(t(T, [game]), dims = 4)
end

(t :: Tensorizor)(game) = t(Array{Float32}, game)

"""
    (t :: Tensorizor)(T, buffer, games)

This method for tensorizors has to be provided by game implementations.
"""
function (t :: Tensorizor{G})(T, buf, games) where {G}
  error("This tensorizor does not support games of type $G")
end



"""
Default tensorizor that is used if no explicit tensorizer is specified.

For a game type implementation `G <: AbstractGame` to work with
[`NeuralModel`](@ref), it should at least define

    Base.size(:: DefaultTensorizor{G})

and

    (:: DefaultTensorizor{G})(game :: G)
"""
struct DefaultTensorizor{G} <: Tensorizor{G} end


"""
   createhead(head, insize, nout) 

Check if the neural layer `head` is compatible with the input size `insize` and
number of output neurons `nout`. If `head = nothing`, return a single
[`Dense`](@ref) layer.
"""
function createhead(head, insize, nout)
  if isnothing(head)
    head = Dense(prod(insize), nout, identity)
  end
  @assert isvalidinputsize(head, insize) "Head is incompatible with trunk"
  @assert prod(outputsize(head, insize)) == nout "Head is incompatible with target"
  head
end

# TODO: Due to weird Documenter.jl behavior, types with constructors of the same
# name should not get their own docstring
struct NeuralModel{ G <: AbstractGame,
                    B <: Backend,
                    R <: Tensorizor{G} } <: AbstractModel{G}
  trunk :: Layer{B}
  targets :: Vector{AbstractTarget{G}}
  target_names :: Vector{Symbol}
  target_heads :: Vector{Layer{B}}
  target_activations :: Vector{Activation}
end

"""
Neural network model. Depending on the layer backend, the neural model can be
trained or only used for inference.

---

    NeuralModel(G, trunk; kwargs...)

Create a `NeuralModel` for games of type `G` with neural network trunk layer
`trunk`.

## Arguments
* `targets`: Named tuple of [`AbstractTarget`] that the network should support.
* `heads`: Named tuple of neural layer heads for the specified `targets`. \
  Heads that are not specified default to single dense layers.
* `activations`: Named tuple of activations for the specified `targets`. Falls
  back to the activations returned by [`Target.defaultactivation`](@ref).
* `backend`: The backend of the neural layers. Derived from `trunk` by default.
* `tensorizor`: The tensorizor used to convert games into their array representation. \
  Defaults to [`DefaultTensorizor`](@ref).
* Additional keyword arguments are passed to [`Model.configure`](@ref).
"""
function NeuralModel( :: Type{G}
                    , trunk :: Layer{B}
                    ; targets = (;)
                    , heads = (;)
                    , activations = (;)
                    , tensorizor = DefaultTensorizor{G}()
                    , kwargs...
                    ) where {G, B <: DefaultBackend{Array{Float32}}}

  @assert isvalidinputsize(trunk, size(tensorizor)) """
  Trunk incompatible with gametype $G
  """

  targets = (;
    value = DefaultValueTarget(G),
    policy = DefaultPolicyTarget(G),
    targets...
  )

  activations = (;
    map(Target.defaultactivation, targets)...,
    activations...
  )
  activations = map(a -> resolve(Activation, a), activations)

  @assert length(targets.value) == 1 """
    Target with name :value must have length 1.
  """
  @assert length(targets.policy) == policylength(G) """
    Target with name :policy must have length `policylength($G)`.
  """
  @assert issubset(keys(activations), keys(targets)) """
    Not all activation names correspond to targets.
  """
  @assert issubset(keys(heads), keys(targets)) """
    Not all head names correspond to targets.
  """

  insize = outputsize(trunk, size(tensorizor))
  heads = map(keys(targets)) do name
    target = getproperty(targets, name)
    head = get(heads, name, nothing)
    createhead(head, insize, length(target))
  end

  R = typeof(tensorizor)
  model = NeuralModel{G, B, R}(
    trunk,
    AbstractTarget{G}[values(targets)...],
    Symbol[keys(targets)...],
    Layer{B}[heads...],
    Activation[activations...],
  )
  configure(model; kwargs...)
end

"""
    model(data; targets = targetnames(model), activate = true)
    model(games; targets = targetnames(model), activate = true)

Apply a neural model `model` to a 4d array `data` or a vector `games`.

By default, all supported targets are evaluated. If only a selection of the
targets should be evaluated, this can be specified via passing an iterable of
target names via `targets` (see [`Target.targetnames`](@ref) and
[`Target.targets`](@ref) for all available targets). If `activate = false`, no
target activation (like `tanh` or `softmax`) is applied to the network output.
"""
function (model :: NeuralModel{G, B})( data :: T
                                     ; targets = targetnames(model)
                                     , activate = true
                                     ) where {G, T, B <: Backend{T}}
  @assert isvalidinput(model.trunk, data) """
  Game array data is not compatible with the model trunk.
  """
  @assert issubset(targets, targetnames(model)) """
  Some of the specified targets are not supported.
  """

  batchsize = size(data)[end]
  trunkout = model.trunk(data)
  indices = [findfirst(isequal(name), targetnames(model)) for name in targets]

  out = map(indices) do index
    head = model.target_heads[index]
    tmp = reshape(head(trunkout), :, batchsize)
    if activate
      f = model.target_activations[index]
      res = f(tmp)
      releasememory!(tmp)
    else
      res = tmp
    end
    res
  end

  releasememory!(trunkout)
  out
end

function (model :: NeuralModel{G, B, R})( games :: Vector{G}
                                        ; kwargs...
                                        ) where {G, T, B <: Backend{T}, R} 
  tensorizor = R()
  data = tensorizor(T, games)
  out = model(data; kwargs...)
  releasememory!(data)
  out
end

"""
    apply(model, game; targets = targetnames(model), activate = true)  

Evaluate a single game state `game` with the neural model `model`. In contrast
to `model([game])`, a named tuple `(; name = output, ...)` is returned.

This function is meant for use in forward mode only, so it may contain
optimizations that the training backend cannot handle.
"""
function Model.apply( model :: NeuralModel{G, B}
                    , game :: G
                    ; targets = targetnames(model)
                    , activate = true
                    ) where {G, B <: Backend}

  outputs = model([game]; targets, activate)
  outputs = map(outputs) do output
    out = convert(Array{Float32}, reshape(output, :))
    length(out) == 1 ? out[1] : out
  end
  (; zip(targets, outputs)...)
end

"""
    tensorizor(model :: NeuralModel)

Return the tensorizor utilized by `model`.
"""
tensorizor(:: NeuralModel{G, B, R}) where {G, B, R} = R()


"""
    addtarget!(model, name, target; [head, activation])

Add the target `target` to the neural model `model` under name `name`.
If no explicit `head` is provided, a shallow dense head is used. If no
`activation` is provided, it is set to `:identity`.
"""
function addtarget!( model :: NeuralModel{G, B, R}
                   , name :: Symbol
                   , target :: AbstractTarget{G}
                   ; head :: Union{Nothing, Layer} = nothing
                   , activation = Target.defaultactivation(target)
                   ) where {G, B, R}

  @assert !(name in model.target_names) "Target with name :$name already exists"

  insz = outputsize(model.trunk, size(R()))
  head = createhead(head, insz, length(target))
  head = adapt(B(), head)
  activation = resolve(Activation, activation)

  push!(model.targets, target)
  push!(model.target_names, name)
  push!(model.target_heads, head)
  push!(model.target_activations, activation)
  nothing
end

"""
     adapt(backend, model) 

Switch the backend of a neural model `model` to `backend`.
"""
function adapt(backend :: B , model :: NeuralModel{G}) where {G, B <: Backend}
  R = typeof(tensorizor(model))
  trunk = adapt(backend, model.trunk)
  heads = adapt.(Ref(backend), model.target_heads)
  NeuralModel{G, B, R}(
    trunk,
    model.targets,
    model.target_names,
    heads,
    model.target_activations
  )
end

function adapt(name :: Union{Symbol, String}, model :: NeuralModel)
  adapt(getbackend(name), model)
end

function adapt(T, model :: NeuralModel{G, B}) where {G, B}
  backend = adapt(T, B())
  adapt(backend, model)
end

"""
     parameters(model) 

Return an iterable of the (trainable) parameter arrays in the neural model
`model`.
"""
function parameters(model :: NeuralModel)
  [parameters(model.trunk); mapreduce(parameters, vcat, model.target_heads)]
end

function Base.copy(model :: NeuralModel{G, B}) where {G, B}
  NeuralModel{G, B}(
    copy(model.trunk),
    model.targets,
    copy(model.target_names),
    copy.(model.target_heads),
    copy(model.target_activations),
  )
end

function Target.targets(model :: NeuralModel)
  (; zip(model.target_names, model.targets)...)
end

Target.targetnames(model :: NeuralModel) = model.target_names

"""
    getbackend(model)

Returns the backend of the neural model `model`.
"""
getbackend(:: NeuralModel{G, B}) where {G, B} = B()

function Base.show(io :: IO, m :: NeuralModel{G, B}) where {G, B}
  bname = isregistered(Backend, B()) ? lookupname(Backend, B()) : string(B)
  print(io, "NeuralModel{$(Game.name(G))")
  print(io, ", ")
  show(io, bname)
  print(io, "}(")
  show(io, m.trunk)
  print(io, ")")
end

function Base.show(io :: IO, :: MIME"text/plain", m :: NeuralModel{G, B}) where {G, B}
  bname = isregistered(Backend, B()) ? lookupname(Backend, B()) : B
  print(io, "NeuralModel{$(Game.name(G)), ")
  show(io, bname)
  println(io, "}")
  print(io, " trunk: "); show(io, m.trunk)
  for (name, head) in zip(m.target_names, m.target_heads)
    println(io); print(io, " $name: "); show(io, head)
  end
end

"""
    aligndevice!(model)  

This function is called when `model` is to be used after context-switches
(e.g., in new tasks). Can be used by GPU-based backends to set the active device
consistent with the model.
"""
aligndevice!(model) = nothing

"""
     arraytype(model) 

Returns the array type of the backend of the neural model `model`.
"""
arraytype(model :: NeuralModel) = arraytype(getbackend(model))

function save(io :: IO, model :: NeuralModel, :: DefaultFormat)
  model = adapt(:default, model)
  Pack.pack(io, model)
end

function save(fname :: AbstractString, model :: NeuralModel, fmt :: DefaultFormat)
  open(io -> save(io, model, fmt), fname, "w")
end

