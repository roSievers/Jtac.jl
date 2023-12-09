
"""
Abstract type for models that evaluate game states.

Models can be applied to game states via `apply(model, game)`, which returns a
named tuple of infered properties about `game`. Each of these properties
corresponds to a [`Target.AbstractTarget`](@ref). All models evaluate value
targets (key `:value`) and policy targets (key `:policy`), which are necessary
for Monte-Carlo tree search (MCTS). When exploring the tree, the proposed
`policy` is used to prioritize actions. When reaching an unexplored game state,
the returned `value` is used for backpropagation to inform and improve the
policy.

Any model that implement the [`apply`](@ref) method can be used for playing
matches after it has been attached to an [`Player.IntuitionPlayer`](@ref) or an
[`Player.MCTSPlayer`](@ref). Specific model implementations can also be trained
on datasets generated by players (see [`Player.NeuralModel`](@ref)).
"""
abstract type AbstractModel{G <: AbstractGame} end

@pack {M <: AbstractModel} in TypedFormat{MapFormat}

"""
    apply(model, game; targets = targetnames(model))

Apply `model` to `game`, returning a named tuple with target names as keys and
target predictions as values. By explicitly specifying `targets`, only selected
targets may be calculated.

Models must implement this method (at least with support for `targets =
[:value, :policy]`) in order to be used as engines for [`Player.IntuitionPlayer`](@ref)
and [`Player.MCTSPlayer`](@ref).
"""
apply(model :: AbstractModel, game) = error("not implemented")

"""
     assist(model, game)

Returns a named tuple that may contain none, one, or both of the entries `value`
and `policy`.

The returned information is used by an assisted model (see
[`AssistedModel`](@ref) for a wrapper to assist models) to influence its
predictions. For example, a model that implements a solver for checks in chess
could assist another model by returning `(value = 1, policy = [0, ..., 1, ...,
0])` when a winning move is possible, and return an empty tuple `(;)` otherwise.

Models that are only meant for assistance, and not for playing, do not need to
implement [`apply`](@ref).
"""
assist(model :: AbstractModel, game) = error("not implemented")

"""
    gametype(model)

Returns the game type `G <: AbstractGame` that `model` can be applied to.
"""
gametype(model :: AbstractModel{G}) where {G <: AbstractGame} = G

"""
    ntasks(model)

Returns a suggestion for the number of tasks if the model is applied in
asynchronous contexts.
"""
ntasks(:: AbstractModel) = 1

"""
    isasync(model)

Whether `model` can batch calls to [`apply`](@ref) in asynchronous contexts.
See also [`ntasks`](@ref) and [`AsyncModel`](@ref), the default async model
wrapper.
"""
isasync(model) = false

"""
    basemodel(model)
    basemodel(player)

Extract the lowermost model on which `model` or `player` is based. Except for
explicit wrapper types like [`AsyncModel`](@ref), this usually returns the
model itself.

This method calls [`childmodel`](@ref) repeatedly until a childless model is
found.

See also [`trainingmodel`](@ref) and [`playingmodel`](@ref).
"""
basemodel(m :: AbstractModel) = m


"""
    childmodel(model)
    childmodel(player)

Extract the child model on which `model` or `player` is based on. Except for
explicit wrapper types like [`AsyncModel`](@ref), this usually returns the
model itself.

See also [`basemodel`](@ref), [`trainingmodel`](@ref), [`playingmodel`](@ref).
"""
childmodel(m :: AbstractModel) = m

"""
    trainingmodel(model)
    trainingmodel(player)

Extract the child model of `model` or `player` that can be trained (usually a
[`NeuralModel`](@ref)). Returns nothing if no child of `model` is trainable.

See also [`basemodel`](@ref), [`childmodel`](@ref), [`playingmodel`](@ref).
"""
trainingmodel(m :: AbstractModel) = nothing

"""
    playingmodel(model)
    playingmodel(player)

Extract the model in `model` or `player` that is most suitable for playing.
Applied to models, this always returns the model itself. Can return `nothing` if
`player` does not use a model.

See also [`basemodel`](@ref), [`childmodel`](@ref), [`trainingmodel`](@ref).
"""
playingmodel(m :: AbstractModel) = m

"""
    configure(; backend, async, cache, assist)
    configure(model; backend, async, cache, assist)

Auxiliary methods to conveniently configure a model `model`.

The first method returns a function mapping a model to the given configuration.
The second method returns the configured model.

## Arguments
- `backend`: Set the backend of neural models (see [`NeuralModel`](@ref)).
- `async`: Wrap `model` in [`AsyncModel`](@ref). Can be `true/false` or the \
batchsize of the async model.
- `cache`: Wrap `model` in [`CachingModel`](@ref) if `cache > 0`.
- `assist`: Wrap `model` in [`AssistedModel`](@ref) with assistant model \
`assist`. If `model` is an [`AssistedModel`](@ref) and the option `assist` is \
not specified, then `assist = model.assistant`.
"""
function configure( model
                  ; backend = nothing
                  , async = nothing
                  , cache = nothing
                  , assist = nothing )

    
  base = basemodel(model)
  if model isa AssistedModel && isnothing(assist)
    assist = model.assistant
  end

  if !isnothing(backend)
    base = adapt(backend, base)
  end
  if !isnothing(async) && !(async == false)
    batchsize = (async == true) ? 64 : async
    base = AsyncModel(base; batchsize)
  end
  if !isnothing(cache) && !(cache == false)
    cachesize = (cache == true) ? 100_000 : cache
    base = CachingModel(base; cachesize)
  end
  if !isnothing(assist) && !(assist == false)
    base = AssistedModel(base, assist)
  end
  
  base
end

configure(; kwargs...) = model -> configure(model; kwargs...)

"""
    targets(model) 

Return the targets supported by `model`.
"""
targets(args...; kwargs...) = Target.targets(args...; kwargs...)
Target.targets(model :: AbstractModel{G}) where {G} = Target.defaulttargets(G)

"""
    targetnames(model) 

Return the names of the targets supported by `model`.
"""
targetnames(args...; kwargs...) = Target.targetnames(args...; kwargs...)


"""
Model file format. Formats implement ways of serializing Jtac models. Each
format must have a unique file extension (see [`extension`](@ref)).
"""
abstract type Format end

"""
    extension(format)  

Returns the file extension of the model format `format`.
"""
extension(fmt :: Format) = lookupname(Format, fmt)

"""
    extension(path)  

Returns the file extension of the file at `path`.
"""
function extension(path :: String)
  ext = splitext(path)[2]
  length(ext) <= 1 ? nothing : Symbol(ext[2:end])
end

"""
Default Jtac model format with file ending `.jtm`. Uses the generic
serialization methods provided by `Jtac.Pack` to store models via msgpack.
"""
struct DefaultFormat <: Format end

"""
     isformatextension(ext) 

Returns whether the file extension `ext` is associated to a model file format.
"""
isformatextension(ext) = isregistered(Format, ext)

"""
    save(filename, model; format = DefaultFormat())

Save `model` to the file `filename` in the format `format`. If the
file extension does not match the format extension, the latter is added
automatically.

See also [`load`](@ref). Note that saving and then again loading a
[`NeuralModel`](@ref) may change the backend.
"""
function save(fname :: String, model :: AbstractModel; format = DefaultFormat())
  format = resolve(Format, format)
  if extension(fname) != extension(format)
    fname = fname * "." * String(extension(format))
  end
  save(fname, model, format)
end

"""
    save(filename, model, format)  

Method to be extended by format implementations.
"""
save(fname, :: AbstractModel, :: Format) = error("Not implemented")

function save(io :: IO, model :: AbstractModel, :: DefaultFormat)
  Pack.pack(io, model)
end

function save(fname :: AbstractString, model :: AbstractModel, fmt :: DefaultFormat)
  open(io -> save(io, model, fmt), fname, "w")
end

"""
    load(filename; [format, kwargs...])

Load the model at `filename`. If `format` is not passed, the format is derived
from the file extension. The remaining keyword arguments are passed to
[`configure`](@ref).

See also [`save`](@ref).
"""
function load(fname; format = nothing, kwargs...)
  if isnothing(format)
    ext = extension(fname)
    if isnothing(ext) || !isformatextension(ext)
      error("Cannot determine model format from file extension")
    end
    format = resolve(Format, ext)
  else
    format = resolve(Format, format)
    if extension(format) != extension(fname)
      fname = fname * "." * String(extension(format))
    end
  end
  model = load(fname, format)
  configure(model; kwargs...)
end

"""
    load(filename, format)  

Method to be extended by format implementations.
"""
load(fname, :: Format) = error("Not implemented")

function load(io :: IO, :: DefaultFormat)
  Pack.unpack(io, AbstractModel)
end

function load(fname :: AbstractString, fmt :: DefaultFormat)
  open(io -> load(io, fmt), fname, "r")
end
