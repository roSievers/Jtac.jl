
# TODO: features should get own module

# -------- Feature ----------------------------------------------------------- #

"""
A feature that describes additional information about a game state.

Subtypes of this abstract type must be callable as

    (f :: MyFeature)(game :: G, history :: Vector{G})

on instances `game` of games `G <: Game` that are supported by the feature, and
must return a feature vector. The second argument `history` lists the full
series of game states until it finished, so that the feature can, e.g., know
about the final state that the game ended in. Additionally, the functions 

    feature_length(:: MyFeature, :: Type{G}) :: Int
    feature_conv(:: MyFeature, Vector) :: Vector of size feature_length
    feature_loss(:: MyFeature, :: Vector, :: Vector)

must be implemented.

"""
abstract type Feature end

# Default choices: squared error loss and identity conversion
feature_loss(:: Feature, a, b) = sum(abs2, a .- b)
feature_conv(:: Feature, flabel) = flabel

# Warning if several copies of features are given
function check_features(fs :: Vector{Feature})
  length(unique(fs)) != length(fs) && @warn "Features are not unique"
  fs
end

function feature_length(fs :: Vector{Feature}, G)
  sum(Int[feature_length(f, G) for f in fs])
end

function feature_indices(fs :: Vector{Feature}, G)
  isempty(fs) && return []

  clengths = cumsum(feature_length.(fs, G))
  start_indices = [1; clengths[1:end-1] .+ 1]
  end_indices = clengths

  map((i,j) -> i:j, start_indices, end_indices)
end

feature_name(f :: Feature) = "<unknown>"

MsgPack.msgpack_type(::Type{G}) where G <: Feature = MsgPack.StructType()

# -------- Feature Compability ----------------------------------------------- #

function feature_compatibility(l, model)
  features(l) == features(model) && !isempty(features(model))
end

function feature_compatibility(l, model, dataset)
  fl = features(l)
  fm = features(model)
  fd = features(dataset)

  if isempty(fm)        return false
  elseif fm == fl == fd return true
  elseif fm == fd       @warn "Loss does not support model features"
  elseif fm == fl       @warn "Dataset does not support model features"
  else                  @warn "Loss and dataset do not support model features"
  end

  false
end

# -------- Register new features --------------------------------------------- #

"""
Constant variable that maps registered feature names to a function that maps
template arguments to a fearute type.
"""
const FEATURES = Dict{String, Function}()

"""
    register_feature!(F)
    register_feature!(f, F)

Register a new jtac feature type `F`. If `F` takes template arguments, the function
`f` must be provided. Applying `f` to the template arguments must yield the
concrete feature type. For example, to register a feature with type `Feat{A, B}`, you
can use the code
```
register!(Feat) do a, b
  eval(Expr(:curly, :Feat, a, b))
end
"""
function register_feature!(f :: Function, F :: Type{<:Feature})
  key = Util.nameof(F)
  FEATURES[key] = f
end

register_feature!(F :: Type{<:Feature}) = register_feature!(() -> F, F)


# -------- Constant Dummy Feature -------------------------------------------- #

"""
Dummy feature for testing and debugging purposes.

This feature produces a constant feature vector on all game states of all game
types. It uses the l2 loss function to evaluate the feature prediction quality.
"""
struct ConstantFeature <: Feature 
  name :: String
  data :: Vector{Float32}
end

register_feature!(ConstantFeature)

function ConstantFeature(data; name = "const")
  ConstantFeature(name, data)
end

(f :: ConstantFeature)(:: G, :: Vector{G}) where {G <: AbstractGame} = f.data

feature_name(f :: ConstantFeature) = f.name
feature_length(f :: ConstantFeature, :: Type{<:AbstractGame}) = length(f.data)

Base.show(io :: IO, f :: ConstantFeature) = print(io, "ConstantFeature($(f.name))")

