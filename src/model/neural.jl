
"""
   createhead(head, insize, nout) 

Check if the neural layer `head` is compatible with the input size `insize` and
number of output neurons `nout`. If `head = nothing`, return a single
[`Dense`](@ref) layer.
"""
function createhead(head, insize, nout)
  if isnothing(head)
    head = Dense(prod(insize), nout, :id)
  end
  @assert isvalidinputsize(head, insize) "Head is incompatible with trunk"
  @assert prod(outputsize(head, insize)) == nout "Head is incompatible with target"
  head
end


"""
Neural network model. Depending on the layer backend, the neural model can be
trained or only used for inference during selfplay.
"""
struct NeuralModel{G <: AbstractGame, B <: Backend} <: AbstractModel{G}
  trunk :: Layer{B}
  targets :: Vector{AbstractTarget{G}}
  target_names :: Vector{Symbol}
  target_heads :: Vector{Layer{B}}
  target_activations :: Vector{Activation}
end

Pack.@typed NeuralModel


"""
    NeuralModel(G, trunk; [targets, heads, activations, backend])

Create a `NeuralModel` for games of type `G` with neural network trunk layer
`trunk`.

### Arguments
* `targets`: Named tuple of [`AbstractTarget`] that the network should support.
* `heads`: Named tuple of neural layer heads for the specified `targets`.
* `activations`: Named tuple of activations for the specified `targets`. Falls
  back to the activations returned by [`Target.defaultactivation`].
* `backend`: The backend of the neural layers.

Heads that are not specified default to single dense layers. Activations that
are not specified default to `:identity`.
"""
function NeuralModel( :: Type{G}
                    , trunk :: Layer{B}
                    ; targets = (;)
                    , heads = (;)
                    , activations = (;)
                    , backend = nothing
                    ) where {G, B <: DefaultBackend{Array{Float32}}}

  @assert isvalidinputsize(trunk, size(G)) "Trunk incompatible with gametype $G"

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

  insize = outputsize(trunk, size(G))
  heads = map(keys(targets)) do name
    target = getproperty(targets, name)
    head = get(heads, name, nothing)
    createhead(head, insize, length(target))
  end

  model = NeuralModel{G, B}(
    trunk,
    AbstractTarget{G}[values(targets)...],
    Symbol[keys(targets)...],
    Layer{B}[heads...],
    Activation[activations...],
  )
  isnothing(backend) ? model : adapt(backend, model)
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
  @assert isvalidinput(model.trunk, data)
  @assert issubset(targets, targetnames(model))

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

function (model :: NeuralModel{G, B})( games :: Vector{G}
                                     ; kwargs...
                                     ) where {G, T, B <: Backend{T}} 
  data = Game.array(games)
  data = convert(T, data)
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
                    ) where {G, T, B <: Backend{T}}

  outputs = model([game]; targets, activate)
  outputs = map(outputs) do output
    out = convert(Array{Float32}, reshape(output, :))
    length(out) == 1 ? out[1] : out
  end
  (; zip(targets, outputs)...)
end


"""
    addtarget!(model, name, target; [head, activation])

Add the target `target` to the neural model `model` under name `name`.
If no explicit `head` is provided, a shallow dense head is used. If no
`activation` is provided, it is set to `:identity`.
"""
function addtarget!( model :: NeuralModel{G, B}
                   , name :: Symbol
                   , target :: AbstractTarget{G}
                   ; head :: Union{Nothing, Layer} = nothing
                   , activation = Target.defaultactivation(target)
                   ) where {G, B}

  @assert !(name in model.target_names) "Target with name :$name already exists"

  insz = outputsize(model.trunk, size(G))
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
function adapt(backend :: B, model :: NeuralModel{G}) where {G, B <: Backend}
  trunk = adapt(backend, model.trunk)
  heads = adapt.(Ref(backend), model.target_heads)
  NeuralModel{G, B}(
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
  print(io, "NeuralModel(")
  show(io, m.trunk)
  print(io, ", ")
  show(io, B())
  print(io, ")")
end

function Base.show(io :: IO, :: MIME"text/plain", m :: NeuralModel{G, B}) where {G, B}
  print(io, "NeuralModel{$(Game.name(G))}(")
  show(io, B()); println(")")
  print(io, " trunk: "); show(io, m.trunk); println(io)
  print(io, " heads:")
  for (name, head) in zip(m.target_names, m.target_heads)
    println(io); print(io, "  $name: "); show(io, head)
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

function save(fname, model :: NeuralModel, :: DefaultFormat)
  model = adapt(:default, model)
  open(io -> Pack.pack(io, model), fname, "w")
end

