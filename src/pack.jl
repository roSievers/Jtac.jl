
import MsgPack: BinaryType, StringType, MapType, StructType
import MsgPack: msgpack_type, to_msgpack, from_msgpack

import TranscodingStreams: TOKEN_END
import CodecZstd: ZstdCompressorStream, ZstdDecompressorStream

pack(args...) = MsgPack.pack(args...)
unpack(args...) = MsgPack.unpack(args...)

function pack_compressed(io :: IO, value)
  stream = ZstdCompressorStream(io)
  pack(stream, value)
  write(stream, TOKEN_END)
  flush(stream)
end

function unpack_compressed(io :: IO, T)
  stream = ZstdDecompressorStream(io)
  unpack(stream, T)
end


# -------- Register new types ------------------------------------------------ #

"""
Constant variable that assigns registered type names to a function that can
recreate the type if given the type parameters. Essentially, this enables
automatic and safe unpacking for registered types.
"""
const TYPES = Dict{String, Function}()

function typestring(T)
  @assert T isa Type "$T not a valid DataType"
  name = Base.nameof(T) |> string
  mod = Base.parentmodule(T) |> string
  mod * "." * name
end

"""
    register(T)
    register(f, T)

Register a new concrete type `T` in the Jtac packing system. If `T` takes
template arguments, the function `f` must be provided. Applying `f` to the
arguments must yield the concrete type.

For example, to register a game with type `Game{A, B}`, you can use the code
```
register(Game) do a, b
  eval(Expr(:curly, :Game, a, b))
end
"""
function register(f :: Function, T :: Type{<:Any})
  name = typestring(T)
  @assert !(haskey(TYPES, name))
  TYPES[name] = f
end

register(T :: Type{<:Any}) = register(T) do args...
  isempty(args) ? T : T{args...}
end

#
# Freezing and unfreezing objects
#
# Should be overwritten for types that need intervention before / after
# packing / unpacking (for example, objects relying on pointers or tasks).
#

freeze(x :: Any) = x
unfreeze(x :: Any) = x
is_frozen(x :: Any) = false

#
# Generic MsgPack serialization of structs as MapType
#
# Compared to declaring a type as MsgPack.StructType, using MsgPack.MapType
# together with these functions in from_msgpack and to_msgpack has the
# following advantages and disadvantages:
#
# * We explicitly save the type, so we can reconstruct
#   an object even if we only know an abstract type.
#   For example, we can reconstruct a vector of layers via
#     MsgPack.unpack(_, Vector{Model.Layer})
#   even though Model.Layer is abstract. This would not work
#   if we simply declare subtypes of Model.Layer as MsgPack.MapType.
#
# * We need to do more bookkeeping. In particular, we have to keep
#   a dictionary of registered types
#
# * This approach is way slower, so it should not be used for small objects
#   that are packed / unpacked in great quantity. In particular, we do not
#   want to use it for subtypes of Game.AbstractGame, since we will need
#   to pack/unpack them in great quantities. In contrast, we can easily use
#   this strategy for Layers, Models, or Datasets, where the performance
#   when packing large amounts of binary data is more important (for this
#   reason, we overwrite the method for the type Vector{Float32} above).
#

fields(T) = Base.fieldnames(T)

destruct_typeinfo(x) = x
construct_typeinfo(x) = x

function destruct_typeinfo(T :: DataType)
  params = hasproperty(T, :parameters) ? collect(T.parameters) : []
  tinfo = Any[typestring(T)]
  append!(tinfo, map(destruct_typeinfo, params))
  tinfo
end

function construct_typeinfo(typeinfo :: Array)
  params = map(construct_typeinfo, typeinfo[2:end])
  TYPES[typeinfo[1]](params...)
end

function typed_destruct(value :: T) where {T}
  value = freeze(value)
  dict = destruct(value)
  dict["typeinfo"] = destruct_typeinfo(T)
  dict
end

function typed_construct(dict)
  T = construct_typeinfo(dict["typeinfo"])
  construct(T, dict)
end

function destruct(value :: T) where {T}
  values = [Base.getfield(value, n) for n in fields(T)]
  Dict{String, Any}(String(n) => val for (n, val) in zip(fields(T), values))
end

function construct(T :: Type, dict)
  names = Base.fieldnames(T)
  types = Base.fieldtypes(T)
  tn = Dict(n => F for (n, F) in zip(names, types))
  args = [from_msgpack(tn[n], dict[String(n)]) for n in fields(T)]
  value = T(args...)
  unfreeze(value)
end

macro mappack(T)
  quote
    Pack.msgpack_type(:: Type{<: $T}) = Pack.MapType()
    Pack.to_msgpack(:: Pack.MapType, x :: $T) = Pack.typed_destruct(x)
    Pack.from_msgpack(:: Type{<: $T}, dict) = Pack.typed_construct(dict)
    Pack.from_msgpack(:: Type{Vector{F}}, v :: Vector) where {F <: $T} =
      F[Pack.from_msgpack(F, x) for x in v]
  end |> esc
end

macro mappack(T, symbols)
  @assert symbols isa Expr
  @assert symbols.head == :vect
  @assert all(x -> x isa QuoteNode && x.value isa Symbol, symbols.args)
  quote
    Pack.fields(:: Type{<: $T}) = $symbols
    Pack.@mappack $T
  end |> esc
end


# Faster, but no automatic saving of type information. This means in particular
# that we cannot construct instances from abstract type information only.
macro structpack(T)
  :(Pack.msgpack_type(:: Type{<: $T}) = Pack.StructType()) |> esc
end


# -------- Auxiliary type for storing raw data ------------------------------- #

struct Bytes
  data :: Vector{UInt8}
end

msgpack_type( :: Type{Bytes}) = BinaryType()
to_msgpack(:: BinaryType, bytes :: Bytes) = bytes.data
from_msgpack(:: Type{Bytes}, data :: Vector{UInt8}) = Bytes(data)

#
# Knet parameters and batchnorm moments
#
# Knet parameters are essentially saved as Float32 arrays.
# This code relies on internals of Knet...

#import Knet

#const KnetParam{T, N} = Knet.Param{Array{T, N}} where {T <: Binary, N}
#const KnetBNMoments = Knet.Ops20.BNMoments

#msgpack_type(:: Type{<: KnetParam}) = MapType()
#to_msgpack(:: MapType, x :: KnetParam) = Dict("param" => Knet.value(x))

#from_msgpack(:: Type{KnetParam{T, N}}, d :: Dict) where {T, N} =
#  from_msgpack(Array{T, N}, d["param"]) |> Knet.Param

#msgpack_type(:: Type{KnetBNMoments}) = MapType()
#to_msgpack(:: MapType, bn :: KnetBNMoments) =
#  Dict("momentum" => bn.momentum, "mean" => bn.mean, "var" => bn.var)

#function from_msgpack(:: Type{KnetBNMoments}, dict)
#  T = Array{Float32, 1} # hack: use N = 1 since we save size explicitly
#  mean = isnothing(dict["mean"]) ? nothing : from_msgpack(T, dict["mean"])
#  var = isnothing(dict["var"]) ? nothing : from_msgpack(T, dict["var"])
#  momentum = Float32(dict["momentum"])
#  Knet.bnmoments(; mean, var, momentum)


# NeuralModel(trunk, vhead, vfeat = Feature.Value(weight = 0.3, loss = ))

