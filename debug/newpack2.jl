
"""
Wishlist:
* support nothing, Int64, Float64, String, Bool natively
* support Symbol, Int32, Int16, Int8, UInt64, UInt32, UInt16, UInt on top
* support arbitrary tuples (array)
* support Union{Nothing, A}
* support Vector{T} where T is bitstype
* support Array{T} where T is bitstype (map: data, size)
* support Vector{T} where T is no bistype
* can say that whole type should be packed "binary" (e.g. all arrays)
* can say that certain fields should be packed "binary"
"""




module Pack

"""
Abstract format type.

Formats are responsible for reducing the packing and unpacking of julia values
to primitives that are natively supported.
"""
abstract type Format end

"""
    format(T)

Return the format of `T`. Must be implemented for any type `T` for which
[`pack`] (@ref) and [`unpack`](@ref) are not defined explicitly.

See also [`Format`](@ref).
"""
function format(T :: Type)
  error("no format has been specified for type $T")
end

"""
    construct(T, val, format)

Postprocess a value `val` unpacked in the format `format` and return an object
of type `T`. The type of `val` depends on `format`.

Defaults to `T(val)`.

See [`Format`](@ref) and its subtypes for more information.
"""
construct(:: Type{T}, val, :: Format) where {T} = T(val)

"""
    destruct(val, format)

Preprocess a value `val` for packing it in the format `format`. Each format has
specific requirements regarding the output type of `destruct`.

Defaults to `val`.

See [`Format`](@ref) and its subtypes for more information.
"""
destruct(val, :: Format) = val

"""
    keytype(T, index)

Return the type of the key at `index` in `T`.
"""
keytype(:: Type, :: Any) = error("not implemented")

"""
    keyformat(T, index)

Return the format of the key at `index` in `T`
"""
keyformat(:: Type{T}, index) where {T} = format(keytype(T, index))

"""
    valuetype(T, index)

Return the type of the value at `index` in `T`.
"""
valuetype(:: Type, :: Any) = error("not implemented")

"""
    valueformat(T, index)

Return the format of the value at `index` in `T`.
"""
valueformat(:: Type{T}, index) where {T} = format(valuetype(T, index))


"""
    pack(value, [format]) :: Vector{UInt8}
    pack(io, value, [format]) :: Nothing

Create a binary representation of `value` in the given `format`. If an input
stream is passed, the representation is written into `io`. If no format is
provided, it is derived from the type of value via [`Pack.format`](@ref).
"""
function pack(io :: IO, value :: T, fmt :: Format = format(T)) :: Nothing where {T}
  pack(io, value, fmt)
end

function pack(value :: T, fmt :: Format = format(T)) :: Vector{UInt8} where {T}
  io = IOBuffer(write = true, read = false)
  pack(io, value, fmt)
  take!(io)
end


"""
    unpack(io / bytes, T) :: T

Unpack a binary msgpack representation of a value of type `T` from a byte vector
`bytes` or an output stream `io`.
"""
function unpack(io :: IO, :: Type{T}) :: T where {T}
  fmt = format(T)
  unpack(io, T, fmt)
end

"""
  unpack(io / bytes, T, format)
  unpack(io / bytes, format = AnyFormat())

Unpack a value packed in format `format` from `io` or `bytes` and (optionally)
construct a value of type `T` via [`Pack.construct`](@ref).
"""
function unpack(io :: IO, T, fmt :: Format)
  val = unpack(io, fmt)
  construct(T, val, fmt)
end

function unpack(io :: IO, fmt :: Format)
  ArgumentError("Unpacking in format $fmt not supported") |> throw
end

unpack(io :: IO) = unpack(io, AnyFormat())

function unpack(bytes :: Vector{UInt8}, args...)
  io = IOBuffer(bytes, write = false, read = true)
  unpack(io, args...)
end


"""
    byteerror(byte, format)

Throw an error indicating that `byte` is not compatible with `format`.
"""
function byteerror(byte, :: F) where {F <: Format}
  msg = "invalid format byte $byte when unpacking value in format $F"
  throw(ArgumentError(msg))
end

#
# AnyFormat
# destruct: Any
# construct: Any
#
# msgpack: all (except date and extension)
#

struct AnyFormat <: Format end

"""
    peekformat(io)

Peek at `io` and return the [`Format`](@ref) that best fits the detected msgpack
format.
"""
function peekformat(io)
  byte = peek(io)
  if isformatbyte(byte, NilFormat())
    NilFormat()
  elseif isformatbyte(byte, BoolFormat())
    BoolFormat()
  elseif isformatbyte(byte, SignedFormat())
    SignedFormat()
  elseif isformatbyte(byte, UnsignedFormat())
    UnsignedFormat()
  elseif isformatbyte(byte, FloatFormat())
    FloatFormat()
  elseif isformatbyte(byte, StringFormat())
    StringFormat()
  elseif isformatbyte(byte, BinaryFormat())
    BinaryFormat()
  elseif isformatbyte(byte, VectorFormat())
    VectorFormat()
  elseif isformatbyte(byte, MapFormat())
    MapFormat()
  else
    byteerror(byte, AnyFormat())
  end
end

pack(io :: IO, value, :: AnyFormat) = pack(io, value)

function unpack(io :: IO, :: AnyFormat)
  fmt = peekformat(io)
  unpack(io, fmt)
end

format(:: Type{Any}) = AnyFormat()
construct(:: Type{Any}, val, :: AnyFormat) = val


#
# NilFormat
# 
# destruct: Any
# construct: Nothing
#
# msgpack: nil
#

struct NilFormat <: Format end

function isformatbyte(byte, :: NilFormat)
  byte == 0xc0
end

function pack(io :: IO, value, :: NilFormat) :: Nothing
  write(io, 0xc0)
  nothing
end

function unpack(io :: IO, :: NilFormat) :: Nothing
  byte = read(io, UInt8)
  if byte == 0xc0
    nothing
  else
    byteerror(byte, NilFormat())
  end
end

format(:: Type{Nothing}) = NilFormat()
construct(:: Type{Nothing}, :: Nothing, :: NilFormat) = nothing


#
# BoolFormat
#
# destruct: Bool
# construct: Bool
#
# msgpack: bool 
#

struct BoolFormat <: Format end

function isformatbyte(byte, :: BoolFormat)
  byte == 0xc2 || byte == 0xc3
end

function pack(io :: IO, value, :: BoolFormat) :: Nothing
  if destruct(value, BoolFormat())
    write(io, 0xc3)
  else
    write(io, 0xc2)
  end
  nothing
end

function unpack(io :: IO, :: BoolFormat) :: Bool
  byte = read(io, UInt8)
  if byte == 0xc3
    true
  elseif byte == 0xc2
    false
  else
    byteerror(byte, BoolFormat())
  end
end

format(:: Type{Bool}) = BoolFormat()

#
# SignedFormat
#
# destruct: Signed
# construct: Int64
#
# msgpack: negative fixint,
#          positive fixint,
#          signed 8,
#          signed 16,
#          signed 32,
#          signed 64
#

struct SignedFormat <: Format end

function isformatbyte(byte, :: SignedFormat)
  byte <= 0x7f ||  # positive fixint
  byte >= 0xe0 ||  # negative fixint
  0xd0 <= byte == 0xd3 # signed 8 to 64
end

function pack(io :: IO, value, :: SignedFormat) :: Nothing
  x = destruct(value, SignedFormat())
  if -32 <= x < 0 # negative fixint
    write(io, reinterpret(UInt8, Int8(x)))
  elseif 0 <= x < 128 # positive fixint
    write(io, UInt8(x))
  elseif typemin(Int8) <= x <= typemax(Int8) # signed 8
    write(io, 0xd0)
    write(io, Int8(x))
  elseif typemin(Int16) <= x <= typemax(Int16) # signed 16
    write(io, 0xd1)
    write(io, Int16(x) |> hton)
  elseif typemin(Int32) <= x <= typemax(Int32) # signed 32
    write(io, 0xd2)
    write(io, Int32(x) |> hton)
  elseif typemin(Int64) <= x <= typemax(Int64) # signed 64
    write(io, 0xd3)
    write(io, Int64(x) |> hton)
  else
    ArgumentError("invalid signed integer $x") |> throw
  end
  nothing
end

function unpack(io :: IO, :: SignedFormat) :: Int64
  byte = read(io, UInt8)
  if byte >= 0xe0 # negative fixint
    reinterpret(Int8, byte)
  elseif byte < 128 # positive fixint
    byte
  elseif byte == 0xd0 # signed 8
    read(io, Int8)
  elseif byte == 0xd1 # signed 16
    read(io, Int16) |> ntoh
  elseif byte == 0xd2 # signed 32
    read(io, Int32) |> ntoh
  elseif byte == 0xd3 # signed 64
    read(io, Int64) |> ntoh
  # For compability, also read unsigned values when signed is expected
  elseif byte == 0xcc # unsigned 8
    read(io, UInt8)
  elseif byte == 0xcd # unsigned 16
    read(io, UInt16) |> ntoh
  elseif byte == 0xce # unsigned 32
    read(io, UInt32) |> ntoh
  elseif byte == 0xcf # unsigned 64
    read(io, UInt64) |> ntoh
  else
    byteerror(byte, SignedFormat())
  end
end

format(:: Type{<: Signed}) = SignedFormat()
destruct(value, :: SignedFormat) = Base.signed(value)


#
# UnsignedFormat
#
# destruct: Unsigned
# construct: UInt64
#
# msgpack: positive fixint,
#          unsigned 8,
#          unsigned 16,
#          unsigned 32,
#          unsigned 64
#

struct UnsignedFormat <: Format end

function isformatbyte(byte, :: UnsignedFormat)
  byte <= 0x7f ||  # positive fixint
  0xcc <= byte == 0xcf # unsigned 8 to 64
end

function pack(io :: IO, value, :: UnsignedFormat) :: Nothing
  x = destruct(value, UnsignedFormat())
  if x < 128 # positive fixint
    write(io, UInt8(x))
  elseif x <= typemax(UInt8) # unsigned 8
    write(io, 0xcc)
    write(io, UInt8(x))
  elseif x <= typemax(UInt16) # unsigned 16
    write(io, 0xcd)
    write(io, UInt16(x) |> hton)
  elseif x <= typemax(UInt32) # unsigned 32
    write(io, 0xce)
    write(io, UInt32(x) |> hton)
  elseif x <= typemax(UInt64) # unsigned 64
    write(io, 0xcf)
    write(io, UInt64(x) |> hton)
  else
    ArgumentError("invalid unsigned integer $x") |> throw
  end
  nothing
end

function unpack(io :: IO, :: UnsignedFormat) :: UInt64
  byte = read(io, UInt8)
  if byte < 128 # positive fixint
    byte
  elseif byte == 0xcc # unsigned 8
    read(io, UInt8)
  elseif byte == 0xcd # unsigned 16
    read(io, UInt16) |> ntoh
  elseif byte == 0xce # unsigned 32
    read(io, UInt32) |> ntoh
  elseif byte == 0xcf # unsigned 64
    read(io, UInt64) |> ntoh
  else
    byteerror(byte, UnsignedFormat())
  end
end

format(:: Type{<: Unsigned}) = UnsignedFormat()
destruct(x, :: UnsignedFormat) = Base.unsigned(x)


#
# FloatFormat
#
# destruct: Float16, Float32, Float64
# construct: Float64
#
# msgpack: float 32, float 64
#

struct FloatFormat <: Format end

function isformatbyte(byte, :: FloatFormat)
  byte == 0xca || byte == 0xcb 
end

function pack(io :: IO, value, :: FloatFormat) :: Nothing
  val = destruct(value, FloatFormat())
  if isa(val, Float16) || isa(val, Float32) # float 32
    write(io, 0xca)
    write(io, Float32(val) |> hton)
  else # float 64
    write(io, 0xcb)
    write(io, Float64(val) |> hton)
  end
  nothing
end

function unpack(io :: IO, :: FloatFormat) :: Float64
  byte = read(io, UInt8)
  if byte == 0xca ## float 32
    read(io, Float32) |> ntoh
  elseif byte == 0xcb # float 64
    read(io, Float64) |> ntoh
  else
    byteerror(byte, FloatFormat())
  end
end

format(:: Type{<: AbstractFloat}) = FloatFormat()
destruct(value, :: FloatFormat) = Base.float(value)


#
# StringFormat
#
# destruct: sizeof(.), write(io, .)
# construct: String
#
# msgpack: fixstr, str 8, str 16, str 32
#

struct StringFormat <: Format end

function isformatbyte(byte, :: StringFormat)
  0xa0 <= byte <= 0xbf || # fixstr
  0xd9 <= byte <= 0xdb # str 8 to 32
end

function pack(io :: IO, value, :: StringFormat) :: Nothing
  str = destruct(value, StringFormat())
  n = sizeof(str)
  if n < 32 # fixstr format
    write(io, 0xa0 | UInt8(n))
  elseif n <= typemax(UInt8) # str 8 format
    write(io, 0xd9)
    write(io, UInt8(n))
  elseif n <= typemax(UInt16) # str 16 format
    write(io, 0xda)
    write(io, UInt16(n) |> hton)
  elseif n <= typemax(UInt32) # str 32 format
    write(io, 0xdb)
    write(io, UInt32(n) |> hton)
  else
    ArgumentError("invalid string length $n") |> throw
  end
  write(io, str)
  nothing
end

function unpack(io :: IO, :: StringFormat) :: String
  byte = read(io, UInt8)
  n = if 0xa0 <= byte <= 0xbf # fixstr  format
    byte & 0x1f
  elseif byte == 0xd9 # str 8 format
    read(io, UInt8)
  elseif byte == 0xda # str 16 format
    read(io, UInt16) |> ntoh
  elseif byte == 0xdb # str 32 format
    read(io, UInt32) |> ntoh
  else
    byteerror(byte, StringFormat())
  end
  String(read(io, n))
end

#
# Default destruct / construct
#

destruct(value, :: StringFormat) = Base.string(value)
construct(:: Type{T}, x, :: StringFormat) where {T} = convert(T, x)

#
# String / Symbol support
#

format(:: Type{<: AbstractString}) = StringFormat()

#
# Symbol support
#

format(:: Type{Symbol}) = StringFormat()
construct(:: Type{Symbol}, x, :: StringFormat) = Symbol(x)

#
# BinaryFormat
#
# destruct: length(.), write(io, .)
# construct: Vector{UInt8}
#
# msgpack: bin 8, bin 16, bin 32
#

struct BinaryFormat <: Format end

function isformatbyte(byte, :: BinaryFormat) 
  0xc4 <= byte <= 0xc6
end

function pack(io :: IO, value, :: BinaryFormat) :: Nothing
  bin = destruct(value, BinaryFormat()) 
  n = sizeof(bin)
  if n <= typemax(UInt8) # bin8
    write(io, 0xc4)
    write(io, UInt8(n))
  elseif n <= typemax(UInt16) # bin16
    write(io, 0xc5)
    write(io, UInt16(n) |> hton)
  elseif n <= typemax(UInt32) # bin32
    write(io, 0xc6)
    write(io, UInt32(n) |> hton)
  else
    ArgumentError("invalid binary length $n") |> throw
  end
  write(io, bin)
  nothing
end

function unpack(io :: IO, :: BinaryFormat) :: Vector{UInt8}
  byte = read(io, UInt8)
  n = if byte == 0xc4 # bin8
    read(io, UInt8)
  elseif byte == 0xc5 # bin16
    read(io, UInt16) |> ntoh
  elseif byte == 0xc6 # bin32
    read(io, UInt32) |> ntoh
  else
    byteerror(byte, BinaryFormat())
  end
  read(io, n)
end

"""
Simple struct that implements [`BinaryFormat`](@ref).
"""
struct Bytes
  bytes :: Vector{UInt8}
end

destruct(x :: Bytes, :: BinaryFormat) = x.bytes
construct(:: Type{Bytes}, bytes, :: BinaryFormat) = Bytes(bytes)

#
# Vector support for bitstype elements
#

function destruct(value :: Vector{F}, :: BinaryFormat) where {F}
  @assert isbitstype(F) """
  Only vectors with bitstype elements can be saved in BinVectorFormat.
  """
  value
end

function construct(:: Type{Vector{F}}, bytes, :: BinaryFormat) where {F}
  @assert isbitstype(F) """
  Only vectors with bitstype elements can be loaded from BinVectorFormat.
  """
  convert(Vector{F}, bytes)
end


#
# VectorFormat
#
# destruct: length(.), iterate(.)
# construct: Generator
# methods: valuetype, [valueformat]
#
# msgpack: fixarray, array 16, array 32
#

struct VectorFormat <: Format end

function isformatbyte(byte, :: VectorFormat) 
  0x90 <= byte <= 0x9f || # fixarray
  byte == 0xdc || # array 16
  byte == 0xdd # array 32
end

function pack(io :: IO, value :: T, :: VectorFormat) :: Nothing where {T}
  val = destruct(value, VectorFormat())
  n = length(val)
  if n < 16 # fixarray
    write(io, 0x90 | UInt8(n))
  elseif n <= typemax(UInt16) # array16
    write(io, 0xdc)
    write(io, UInt16(n) |> hton)
  elseif n <= typemax(UInt32) # array32
    write(io, 0xdd)
    write(io, UInt32(n) |> hton)
  else
    ArgumentError("invalid array length $n") |> throw
  end
  for (index, x) in enumerate(val)
    fmt = valueformat(T, index)
    pack(io, x, fmt)
  end
  nothing
end

function unpack(io :: IO, :: VectorFormat) :: Vector
  byte = read(io, UInt8)
  n = if byte & 0xf0 == 0x90 # fixarray
    byte & 0x0f
  elseif byte == 0xdc # array 16
    read(io, UInt16) |> ntoh
  elseif byte == 0xdd # array 32
    read(io, UInt32) |> ntoh
  else
    byteerror(byte, VectorFormat())
  end
  map(1:n) do index
    unpack(io, AnyFormat())
  end
end

function unpack(io :: IO, :: Type{T}, :: VectorFormat) :: T where {T}
  byte = read(io, UInt8)
  n = if byte & 0xf0 == 0x90 # fixarray
    byte & 0x0f
  elseif byte == 0xdc # array 16
    read(io, UInt16) |> ntoh
  elseif byte == 0xdd # array 32
    read(io, UInt32) |> ntoh
  else
    byteerror(byte, VectorFormat())
  end
  values = Iterators.map(1:n) do index
    S = valuetype(T, index)
    fmt = valueformat(T, index)
    unpack(io, S, fmt)
  end
  construct(T, values, VectorFormat())
end

# Struct support

function destruct(value :: T, :: VectorFormat) where {T}
  n = Base.fieldcount(T)
  Iterators.map(1:n) do index
    Base.getfield(value, index)
  end
end

construct(:: Type{T}, vals, :: VectorFormat) where {T} = T(vals...)
valuetype(:: Type{T}, index) where {T} = Base.fieldtype(T, index)

# Tuple support (default)

format(:: Type{<: T}) where {T <: Tuple} = VectorFormat()

function construct(:: Type{T}, vals, :: VectorFormat) where {T <: Tuple}
  convert(T, (vals...,))
end

# NamedTuple support (default MapFormat)

function construct(:: Type{T}, vals, :: VectorFormat) where {T <: NamedTuple}
  T(vals)
end

# Vector support (default)

format(:: Type{<: AbstractVector}) = VectorFormat()
destruct(value :: AbstractArray, :: VectorFormat) = value

function construct(:: Type{T}, vals, :: VectorFormat) where {T <: AbstractVector}
  convert(T, collect(vals))
end

valuetype(:: Type{T}, _) where {T <: AbstractArray} = eltype(T)

#
# MapFormat
#
# destruct: length(.), iterate(.)
# construct: Base.Generator
# requires: keytype, valuetype, [keyformat, valueformat]
#
# msgpack: fixmap, map 16, map 32
#

struct MapFormat <: Format end

function isformatbyte(byte, :: MapFormat) 
  0x80 <= byte <= 0x8f || # fixmap
  byte == 0xde || # map 16
  byte == 0xdf # map 32
end

function pack(io :: IO, value :: T, :: MapFormat) :: Nothing where {T}
  val = destruct(value, MapFormat())
  n = length(val)
  if n < 16 # fixmap
    write(io, 0x80 | UInt8(n))
  elseif n <= typemax(UInt16) # map 16
    write(io, 0xde)
    write(io, UInt16(n) |> hton)
  elseif n <= typemax(UInt32) # map 32
    write(io, 0xdf)
    write(io, UInt32(n) |> hton)
  else
    ArgumentError("invalid map length $n") |> throw
  end
  for (index, (key, val)) in enumerate(val)
    fmt_key = keyformat(T, index)
    fmt_val = valueformat(T, index)
    pack(io, key, fmt_key)
    pack(io, val, fmt_val)
  end
  nothing
end

function unpack(io :: IO, :: MapFormat) :: Dict
  byte = read(io, UInt8)
  n = if byte & 0xf0 == 0x80
    byte & 0x0f
  elseif byte == 0xde 
    read(io, UInt16) |> ntoh
  elseif byte == 0xdf 
    read(io, UInt32) |> ntoh
  else
    byteerror(byte, MapFormat())
  end
  pairs = Iterators.map(1:n) do index
    key = unpack(io, AnyFormat())
    value = unpack(io, AnyFormat())
    (key, value)
  end
  Dict(pairs)
end

function unpack(io :: IO, :: Type{T}, :: MapFormat) :: T where {T}
  byte = read(io, UInt8)
  n = if byte & 0xf0 == 0x80
    byte & 0x0f
  elseif byte == 0xde 
    read(io, UInt16) |> ntoh
  elseif byte == 0xdf 
    read(io, UInt32) |> ntoh
  else
    byteerror(byte, MapFormat())
  end
  pairs = Iterators.map(1:n) do index
    K = keytype(T, index)
    V = valuetype(T, index)
    fmt_key = keyformat(T, index)
    fmt_val = valueformat(T, index)
    key = unpack(io, K, fmt_key)
    value = unpack(io, V, fmt_val)
    (key, value)
  end
  construct(T, pairs, MapFormat())
end

#
# Generic struct support
#

function destruct(value :: T, :: MapFormat) where {T}
  n = Base.fieldcount(T)
  Iterators.map(1:n) do index
    key = Base.fieldname(T, index)
    val = Base.getfield(value, index)
    (key, val)
  end
end

function construct(:: Type{T}, pairs, :: MapFormat) where {T}
  values = Iterators.map(last, pairs)
  T(values...)
end

keytype(:: Type{T}, _) where {T} = Symbol
valuetype(:: Type{T}, index) where {T} = Base.fieldtype(T, index)

#
# NamedTuple support (default)
#

format(:: Type{<: NamedTuple}) = MapFormat()

function construct(:: Type{T}, pairs, :: MapFormat) where {T <: NamedTuple}
  values = Iterators.map(last, pairs)
  T(values)
end

#
# Dict support (default)
#

format(:: Type{<: Dict}) = MapFormat()
destruct(value :: Dict, :: MapFormat) = value
construct(:: Type{<: Dict}, pairs, :: MapFormat) = Dict(pairs)
keytype(:: Type{<: Dict{K, V}}, _) where {K, V} = K
valuetype(:: Type{<: Dict{K, V}}, _) where {K, V} = V

## TODO
# - AliasFormat
# - BinVectorFormat
# - ArrayFormat
# - BinArrayFormat
# - TypedFormat


#
# BinVector format
#
# destruct: BinaryFormat
# construct: Vector{UInt8}
#

struct BinVectorFormat <: Format end

function pack(io :: IO, value, :: BinVectorFormat) :: Nothing
  val = destruct(value, BinVectorFormat())
  pack(io, val, BinaryFormat())
end

function unpack(io :: IO, :: Type{T}, :: BinVectorFormat) :: T where {T}
  bytes = unpack(io, BinaryFormat())
  construct(T, bytes, BinVectorFormat())
end

function construct(:: Type{Vector{F}}, bytes, :: BinVectorFormat) where {F}
  vals = reinterpret(F, bytes)
  convert(Vector{F}, vals)
end


#
# Array format
#
# destruct: size(.), VectorFormat
# construct: eltype(T), ArrayValue
#

struct ArrayFormat <: Format end

struct ArrayValue{T}
  data :: T
  size :: Vector{Int}
end

format(:: Type{<: ArrayValue}) = MapFormat()
valuetype(:: Type{ArrayValue{T}}, index) where {T} = index == 1 ? T : Vector{Int}
valueformat(:: Type{<: ArrayValue}, index) = VectorFormat()

function pack(io :: IO, value, :: ArrayFormat) :: Nothing
  val = destruct(value, ArrayFormat())
  pack(io, ArrayValue(val, collect(size(val))))
end

function unpack(io :: IO, :: Type{T}, :: ArrayFormat) :: T where {T}
  V = Vector{eltype(T)}
  val = unpack(io, ArrayValue{V})
  construct(T, val, ArrayFormat())
end

# ND Array support

format(:: Type{<: AbstractArray}) = ArrayFormat()

function construct(:: Type{A}, val, :: ArrayFormat) where {A <: AbstractArray}
  convert(A, reshape(val.data, val.size...))
end


#
# BinArrayFormat
#
# destruct: size, BinVectorFormat
# construct: BinArrayValue
#

struct BinArrayFormat <: Format end

struct BinArrayValue{T}
  data :: T
  size :: Vector{Int}
end

format(:: Type{<: BinArrayValue}) = MapFormat()

function valuetype(:: Type{BinArrayValue{T}}, index) where {T}
  index == 1 ? T : Vector{Int}
end

function valueformat(:: Type{<: BinArrayValue}, index)
  index == 1 ? BinVectorFormat() : VectorFormat()
end

function pack(io :: IO, value, :: BinArrayFormat)
  val = destruct(value, BinArrayFormat())
  pack(io, BinArrayValue(val, collect(size(val))))
end

function unpack(io :: IO, :: Type{T}, :: BinArrayFormat) :: T where {T}
  V = Vector{eltype(T)}
  val = unpack(io, BinArrayValue{V})
  construct(T, val, BinArrayFormat())
end

# ND Array support

function construct(:: Type{A}, val, :: BinArrayFormat) where {A <: AbstractArray}
  convert(A, reshape(val.data, val.size...))
end

#
# TypeFormat
#

struct TypeFormat <: Format end

struct TypeValue
  name :: Symbol
  path :: Vector{Symbol}
  params :: Vector{TypeValue}
end

"""
    TypeValue(T)

Construct a [`TypeValue`](@ref) out of the type `T`.

Currently, the following limitation for parameterized types applies: If `T` has
more type parameters than are explicitly specified (e.g., `T = Array{Float32}`),
the specified type parameters must come first (e.g., `T = Array{F, 1} where {F}`
would fail).
"""
function TypeValue(T :: Type)
  name = Base.nameof(T)
  path = string(Base.parentmodule(T))
  path = Symbol.(split(path, "."))
  params = extracttypeparams(T)
  params = TypeValue.(params)
  TypeValue(name, path, params)
end



function extracttypeparams(T)
  params = nothing
  vars = []
  body = T
  while isnothing(params)
    if hasproperty(body, :parameters)
      params = collect(body.parameters)
    elseif hasproperty(body, :body) && hasproperty(body, :var)
      push!(vars, body.var)
      body = body.body
    else
      error("Failed to understand structure of type $T")
    end
  end
  for R in reverse(vars)
    @assert pop!(params) == R "Cannot extract type parameters from type $T"
  end
  params
end

function composetype(value :: TypeValue) :: Type
  # Resolve type from module path and name
  T = Main
  for m in value.path
    T = getfield(T, m) :: Module
  end
  T = getfield(T, value.name) :: Type
  # attach type parameters
  if isempty(value.params)
    T
  else
    params = map(composetypeparams, value.params)
    T{params...}
  end
end

format(:: Type{TypeValue}) = MapFormat()

pack(io :: IO, value, :: TypeFormat) :: Nothing = pack(io, TypeValue(value))

function unpack(io :: IO, :: TypeFormat) :: Type
  value = unpack(io, TypeValue)
  composetype(value)
end

format(:: Type{<: Type}) = TypeFormat()


#
# ForcedFormat
#

"""
Format wrapper that is used to resolve ambiguities for the type
[`ForcedFormat`](@ref).
"""
struct ForcedFormat <: Format end

"""
Wrapper that enforces packing / unpacking of a value in a certain format.
"""
struct ForcedValue{F <: Format, T}
  value :: T
end

function pack(io :: IO, value :: ForcedValue{F, T}, :: ForcedFormat) where {F, T}
  pack(io, value.value, F())
end

function unpack(io, :: Type{ForcedValue{F, T}}, :: ForcedFormat) where {F, T}
  unpack(io, T, F())
end

ForcedValue(val :: T, :: F) where {F, T} = ForcedValue{F, T}(val)

format(:: Type{<: ForcedValue}) = ForcedFormat()


struct Alias{S} <: Format end

function pack(io :: IO, value :: T, :: Alias{S}) :: Nothing where {S, T}
  val = destruct(value, Alias{S}())
  pack(io, val)
end

function unpack(io :: IO, :: Type{T}, :: Alias{S}) :: T where {S, T}
  val = unpack(io, S)
  construct(T, val, Alias{S}())
end

construct(:: Type{T}, val, :: Alias) where {T} = T(val)
destruct(val :: T, :: Alias{S}) where {S, T} = S(val)


#
# Value that forces packing / unpacking in a certain format
#


struct BinVectorFormat <: Format end
struct BinArrayFormat <: Format end

end

using Random
using MsgPack
using BenchmarkTools

function packtest1(x, T = typeof(x))
  bin = Pack.pack(x)
  y = Pack.unpack(bin, T)
  all(bin .== Pack.pack(y))
end

function packtest2(x, T = typeof(x))
  bin1 = Pack.pack(x)
  bin2 = MsgPack.pack(x)
  v1 = MsgPack.unpack(bin1, T)
  v2 = Pack.unpack(bin2, T)
  bin11 = Pack.pack(v1)
  bin22 = Pack.pack(v2)
  all(bin11 .== bin11) && all(bin2 .== bin22)
end

function speedtest(x, T = typeof(x))
  bin = Pack.pack(x)
  println("Pack")
  @btime Pack.pack($x)
  @btime Pack.unpack($bin, $T)
  println("MsgPack")
  @btime MsgPack.pack($x)
  @btime MsgPack.unpack($bin, $T)
end

using Test

# @testset "Nothing" begin
#   @test packtest1(nothing)
#   @test packtest2(nothing)
# end
# speedtest(nothing)

# @testset "Bool" begin
#   @test packtest1(true)
#   @test packtest2(true)
#   @test packtest1(false)
#   @test packtest2(false)
# end
# speedtest(true)

# @testset "Integer" begin
#   for i in -100:100
#     @test packtest1(i)
#     @test packtest2(i)
#   end
#   for T in [Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64]
#     i = rand(T)
#     @test packtest1(i)
#     i = typemin(T)
#     # MsgPack also uses unsigned format for signed types, therefore loading
#     # fails.
#     @test packtest2(i)
#   end
# end
# speedtest(-13)
# speedtest(12)

# @testset "String" begin
#   @test packtest1("haha")
#   @test packtest2("haha")
#   @test packtest1(:haha)
#   @test packtest2(:haha)
#   for len in [2, 2^8-1, 2^8 + 1, 2^16 + 1]
#     str = String(rand(UInt8, len))
#     @test packtest1(str)
#     @test packtest2(str)
#   end
# end
# speedtest("hahaha")

# @testset "Float" begin
#   f32 = rand(Float32)
#   f64 = rand(Float64)
#   @test packtest1(f32)
#   @test packtest2(f32)
#   @test packtest1(f64)
#   @test packtest2(f64)
# end
# speedtest(f32)
# speedtest(f64)

# @testset "Vector" begin
#   for n in [1, 10, 100, 1000]
#     a = rand(10)
#     @test packtest1(a)
#     @test packtest2(a)
#   end
#   b = (-5, "haha", 3.0, true, nothing) 
#   @test packtest1(b)
#   @test packtest2(b)
# end
b = (-5, "haha", 3.0, true, nothing, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) 
# b = (3, 3.0, true)
# speedtest(b)

# @testset "Map" begin
#   a = 5
# end

v1 = (a = 3, b = 7.0, hehehuvle = "best time ever")
# speedtest(v1)
# speedtest(tuple(v1...))

struct TestStruct
  a :: Int
  b :: Float64
  hehehuvle :: String
end

# Pack.format(:: Type{TestStruct}) = Pack.VectorFormat()

# v2 = TestStruct(3, 7.0, "best time ever")
# speedtest(v2)

Pack.format(:: Type{TestStruct}) = Pack.MapFormat()

v2 = TestStruct(3, 7.0, "best time ever")
# speedtest(v2)


val = [randstring(100) for _ in 1:100]
# speedtest(val)
bytes = Pack.pack(val, Pack.ArrayFormat())
# Pack.unpack(bytes, Pack.ArrayFormat())
Pack.unpack(bytes, Vector{String}, Pack.ArrayFormat())
# @btime Pack.unpack($bytes)
# @btime MsgPack.unpack($bytes)

# array = rand(5, 5)
# bytes = Pack.pack(array)

# v2 = Dict(:a => 3, :b => 7, :c => 8)

# packtest1(v)


