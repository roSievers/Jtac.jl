
"""
Dictionary that holds named values that have been registered via
[`Util.register!`](@ref).
"""
const _nametable = Dict{Any, Dict{Symbol, Any}}()

"""
    register!(T, value, names...)

Register the value `convert(T, value)` under the names provided by `names`.
The type `T` is called the "scope type" and is (besides conversion) also
responsible for scoped naming. It must always be provided when looking up a
previously registered named value.

See also [`lookup`](@ref), [`resolve`](@ref), and [`lookupname`](@ref).
"""
function register!(T, value, names :: Symbol ...)
  for name in names
    if haskey(_nametable, T) && haskey(_nametable[T], name)
      error("A value with name :$name is already registered in scope $T")
    elseif haskey(_nametable, T)
      _nametable[T][name] = convert(T, value)
    else
      _nametable[T] = Dict{Symbol, Any}(name => convert(T, value))
    end
  end
end

"""
    isregistered(T, name)
    isregistered(T, value)

Check whether a value `value` or name `name` is registered under scope type `T`.

See also [`register!`](@ref), [`lookup`](@ref), and [`resolve`](@ref).
"""
function isregistered(T, name :: Symbol)
  haskey(_nametable, T) && haskey(_nametable[T], name)
end

isregistered(T, name :: AbstractString) = isregistered(T, Symbol(name))

function isregistered(:: Type{T}, value :: T) where {T}
  haskey(_nametable, T) && value in values(_nametable[T])
end

"""
    lookup(T)

Return a dictionary containing all named values with scope type `T`.
"""
function lookup(T)
  if !haskey(_nametable, T)
    throw(ArgumentError("Named value scope $T does not exist."))
  else
    _nametable[T]
  end
end

"""
    lookup(T, name)
    lookup(T, value)

Return the value with scope type `T` and name `name`.

See also [`resolve`](@ref), [`register!`](@ref), and [`lookupname`](@ref).
"""
function lookup(T, name :: Symbol)
  if !haskey(_nametable, T)
    throw(ArgumentError("Named value scope $T does not exist."))
  elseif !haskey(_nametable[T], name)
    throw(ArgumentError("Name :$name is not registered in scope $T"))
  else
    _nametable[T][name]
  end
end

lookup(T, name :: AbstractString) = lookup(T, Symbol(name))

"""
    lookupname(T, value)

Return the name that `value` with scope type `T` has been registered as.

See also [`register!`](@ref) and [`lookup`](@ref).
"""
function lookupname(T, value :: Any)
  if !haskey(_nametable, T)
    throw(ArgumentError("Named value scope $T does not exist."))
  else
    key = findfirst(isequal(value), _nametable[T])
    if isnothing(key)
      throw(ArgumentError("Value $value is not registered in scope $T"))
    else
      key
    end
  end
end


"""
    resolve(T, name)
    resolve(T, value)

Return `lookup(T, name)` if `name` is a [`Symbol`](@ref) or [`String`](@ref) and
`convert(T, value)` otherwise.

See also [`lookup`](@ref) and [`register!`](@ref).
"""
resolve(T, name :: Union{AbstractString, Symbol}) = lookup(T, name)
resolve(T, value) = convert(T, value)

@pack MyType in NamedValueFormat{MyType}

@pack MyType (a in BinArrayFormat, b in BinArrayFormat)

struct NamedValueFormat{S} <: Pack.Format end

function Pack.pack(io :: IO, value, :: NamedValueFormat{S}) where {S}
  name = lookupname(S, value)
  Pack.pack(io, name, Pack.StringFormat())
end

function Pack.unpack(io :: IO, :: Type{T}, :: NamedValueFormat{S}) :: T where {T, S}
  name = Pack.unpack(io, Pack.StringFormat())
  lookup(S, name)
end
