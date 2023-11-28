
"""
Abstract format type.

Formats are responsible for reducing the packing and unpacking of julia values
to primitives that are natively supported.
"""
abstract type Format end

"""
    format(T)

Return the default format associated to type `T`. Must be implemented in order
for `pack(io, value :: T)` and `unpack(io, T)` to work.

See also [`Format`](@ref) and [`DefaultFormat`](@ref).
"""
function format(T :: Type)
  error("no default format has been specified for type $T")
end

format(value) = format(typeof(value))

"""
Special format that acts as placeholder for `Pack.format(T)` in situations where
the type `T` is not yet known.

!!! Never define `Pack.format(T)` for your type `T` in terms of `DefaultFormat`.
!!! This will lead to infinite recursion.
"""
struct DefaultFormat <: Format end

pack(io :: IO, val, :: DefaultFormat) = pack(io, val)
unpack(io :: IO, :: Type{T}, :: DefaultFormat) where {T} = unpack(io, T)

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
keytype(:: Type, :: Any) = Symbol

"""
    keyformat(T, index)

Return the format of the key at `index` in `T`
"""
keyformat(:: Type{T}, index) where {T} = DefaultFormat()

"""
    valuetype(T, index)

Return the type of the value at `index` in `T`.
"""
valuetype(:: Type{T}, index) where {T} = Base.fieldtype(T, index)

"""
    valueformat(T, index)

Return the format of the value at `index` in `T`.
"""
valueformat(:: Type{T}, index) where {T} = DefaultFormat()

"""
    pack(value, [format]) :: Vector{UInt8}
    pack(io, value, [format]) :: Nothing

Create a binary representation of `value` in the given `format`. If an input
stream is passed, the representation is written into `io`.

If no format is provided, it is derived from the type of value via
[`Pack.format`](@ref).
"""
function pack(io :: IO, value :: T) :: Nothing where {T}
  pack(io, value, format(T))
end

function pack(value :: T, args...) :: Vector{UInt8} where {T}
  io = IOBuffer(write = true, read = false)
  pack(io, value, args...)
  take!(io)
end

"""
    unpack(io / bytes) :: Any
    unpack(io / bytes, T, [format]) :: T

Unpack a binary msgpack representation of a value of type `T` from a byte vector
`bytes` or an output stream `io`.

If no format and no type is provided, the format [`AnyFormat`](@ref) is used.
If no format is provided, it is derived from `T` via [`Pack.format`](@ref).
[`Pack.format`](@ref).
"""
function unpack(io :: IO, :: Type{T}) :: T where {T}
  fmt = format(T)
  unpack(io, T, fmt)
end

function unpack(io :: IO, T, fmt :: Format)
  val = unpack(io, fmt)
  construct(T, val, fmt)
end

function unpack(:: IO, fmt :: Format)
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
  0xd0 <= byte <= 0xd3 # signed 8 to 64
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
  0xcc <= byte <= 0xcf # unsigned 8 to 64
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
  Only vectors with bitstype elements can be saved in BinaryFormat.
  """
  value
end

function construct(:: Type{Vector{F}}, bytes, :: BinaryFormat) where {F}
  @assert isbitstype(F) """
  Only vectors with bitstype elements can be loaded from BinaryFormat.
  """
  value = reinterpret(F, bytes)
  convert(Vector{F}, value)
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
  map(1:n) do _
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

#
# Auxiliary structfor valuetype injection
#
# Currently used for ArrayFormat
#

"""
Auxiliary structure that can be used to inject the [`valuetype`](@ref) method of
some type into a call to [`unpack`](@ref).
"""
struct ValueTypeOf{T}
  value
end

construct(:: Type{ValueTypeOf{T}}, val, :: Format) where {T} = ValueTypeOf{T}(val)
valuetype(:: Type{ValueTypeOf{T}}, index) where {T} = valuetype(T, index)
construct(:: Type{ValueTypeOf{T}}, val, :: VectorFormat) where {T} = ValueTypeOf{T}(val)
construct(:: Type{ValueTypeOf{T}}, val, :: MapFormat) where {T} = ValueTypeOf{T}(val)



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
  datatype :: Symbol
  size :: Vector{Int}
  data :: T
end

format(:: Type{<: ArrayValue}) = MapFormat()

function valueformat(:: Type{<: ArrayValue}, index)
  index == 3 ? VectorFormat() : DefaultFormat()
end

function pack(io :: IO, value, :: ArrayFormat) :: Nothing
  val = destruct(value, ArrayFormat())
  datatype = Base.eltype(val) |> string |> Symbol
  pack(io, ArrayValue(datatype, collect(size(val)), val))
end

function unpack(io :: IO, :: Type{T}, :: ArrayFormat) :: T where {T}
  val = unpack(io, ArrayValue{ValueTypeOf{T}})
  val = ArrayValue(val.datatype, val.size, val.data.value)
  construct(T, val, ArrayFormat())
end

# ND Array support

format(:: Type{<: AbstractArray}) = ArrayFormat()

function construct(:: Type{T}, val, :: ArrayFormat) where {T <: AbstractArray}
  data = collect(val.data)
  convert(T, reshape(data, val.size...))
end


#
# BinArrayFormat
#
# destruct: size, BinVectorFormat
# construct: BinArrayValue
#

struct BinArrayFormat <: Format end

struct BinArrayValue{T}
  datatype :: Symbol
  size :: Vector{Int}
  data :: T
end

format(:: Type{<: BinArrayValue}) = MapFormat()

function valueformat(:: Type{<: BinArrayValue}, index)
  index == 3 ? BinVectorFormat() : DefaultFormat()
end

function pack(io :: IO, value, :: BinArrayFormat)
  val = destruct(value, BinArrayFormat())
  datatype = Base.eltype(val) |> string |> Symbol
  pack(io, BinArrayValue(datatype, collect(size(val)), val))
end

function unpack(io :: IO, :: Type{T}, :: BinArrayFormat) :: T where {T}
  val = unpack(io, BinArrayValue{Vector{UInt8}})
  construct(T, val, BinArrayFormat())
end

# ND Array support

function construct(:: Type{T}, val, :: BinArrayFormat) where {F, T <: AbstractArray{F}}
  data = reinterpret(F, val.data)
  convert(T, reshape(data, val.size...))
end

#
# ValFormat
#

"""
Format wrapper that is used to resolve ambiguities for the type
[`ValFormat`](@ref).
"""
struct ValFormat <: Format end

"""
Wrapper that enforces packing / unpacking of a value in a certain format.
"""
struct Val{F <: Format, T}
  value :: T
end

Val(val :: T, :: F) where {F, T} = Val{F, T}(val)

format(:: Type{<: Val}) = ValFormat()

function pack(io :: IO, value :: Val{F, T}, :: ValFormat) where {F, T}
  pack(io, value.value, F())
end

function unpack(io :: IO, :: Type{Val{F, T}}, :: ValFormat) where {F, T}
  Val(unpack(io, T, F()), F())
end

function unpack(io :: IO, :: Type{Val{F}}, :: ValFormat) where {F <: Format}
  Val(unpack(io, F()), F())
end


#
# TypeFormat
#

struct TypeParamFormat <: Format end

pack(io :: IO, value, :: TypeParamFormat) = pack(io, value)

function unpack(io :: IO, :: TypeParamFormat)
  if peekformat(io) == MapFormat()
    unpack(io, TypeValue)
  elseif peekformat(io) == VectorFormat()
    Tuple(unpack(io, Vector))
  else
    unpack(io, Any)
  end
end

struct TypeFormat <: Format end

struct TypeValue
  name :: Symbol
  path :: Vector{Symbol}
  params :: Vector{Val{TypeParamFormat}}
end

# TODO: check if there is a way to find the type of a bitstype type parameter!
"""
    TypeValue(T)

Construct a [`TypeValue`](@ref) out of the type `T`.

Currently, the following limitations for parameterized types apply:
* If `T` has more type parameters than are explicitly specified \
(e.g., `T = Array{Float32}`), the specified type parameters must come first \
(e.g., `T = Array{F, 1} where {F}` would fail).
* Only certain primitive bitstypes are supported as type parameters, like \
`Bool` or `Int64`. Other bitstypes (like `UInt64` or `Int16`) are converted \
to Int64 type parameters when unpacking.
"""
function TypeValue(T :: Type)
  name = Base.nameof(T)
  path = string(Base.parentmodule(T))
  path = Symbol.(split(path, "."))
  params = typeparams(T)
  params = map(params) do param
    if param isa Type
      Val(TypeValue(param), TypeParamFormat())
    else
      Val(param, TypeParamFormat())
    end
  end
  TypeValue(name, path, params)
end

format(:: Type{TypeValue}) = MapFormat()

pack(io :: IO, value, :: TypeFormat) :: Nothing = pack(io, TypeValue(value))

function unpack(io :: IO, :: TypeFormat) :: Type
  value = unpack(io, TypeValue)
  composetype(value)
end

function typeparams(T)
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
    params = map(value.params) do param
      composetypeparam(param.value)
    end
    T{params...}
  end
end

function composetypeparam(value)
  if value isa TypeValue
    composetype(value)
  else
    value
  end
end

format(:: Type{<: Type}) = TypeFormat()

function construct(:: Type{Type}, S, :: TypeFormat)
  @assert isa(S, Type) "unpacked value $S was expected to be a type"
  S
end


#
# TypedFormat
#

"""
Format reserved to the type [`Typed`](@ref)
"""
struct TypedFormat{F <: Format} <: Format end

TypedFormat() = TypedFormat{DefaultFormat}()

function pack(io :: IO, value, :: TypedFormat{F}) where {F <: Format}
  write(io, 0x82) # fixmap of length 2
  pack(io, :type)
  pack(io, typeof(value))
  pack(io, :value)
  pack(io, Val(value, F()))
end

function pack(io :: IO, value :: T, :: TypedFormat{DefaultFormat}) where {T}
  F = typeof(format(T))
  pack(io, value, TypedFormat{F}())
end

function unpack(io :: IO, :: TypedFormat{F}) where {F <: Format}
  byte = read(io, UInt8)
  if byte == 0x82 # expect fixmap of length 2
    key = unpack(io, Symbol)
    @assert key == :type "Expected map key :type when unpacking typed value"
    T = unpack(io, Type)
    key = unpack(io, Symbol)
    @assert key == :value "Expected map key :value when unpacking typed value"
    unpack(io, T, F())
  else
    byteerror(byte, TypedFormat{F}())
  end
end

function unpack(io :: IO, :: Type{T}, :: TypedFormat{F}) :: T where {T, F <: Format}
  val = unpack(io, TypedFormat{F}())
  @assert val isa T "Expected value type $T when unpacking typed value"
  val
end

function unpack(io :: IO, :: TypedFormat{DefaultFormat})
  byte = read(io, UInt8)
  if byte == 0x82 # expect fixmap of length 2
    key = unpack(io, Symbol)
    @assert key == :type "Expected map key :type when unpacking typed value"
    T = unpack(io, Type)
    key = unpack(io, Symbol)
    @assert key == :value "Expected map key :value when unpacking typed value"
    unpack(io, T)
  else
    byteerror(byte, TypedFormat{DefaultFormat}())
  end
end

#
# StreamFormat
#

struct StreamFormat{S, F <: Format} <: Format end

StreamFormat(S) = StreamFormat{S, DefaultFormat}()

function pack(io :: IO, value, :: StreamFormat{S, F}) where {S, F}
  stream = S(io)
  pack(stream, value, F)
  write(stream, TOKEN_END)
end

function unpack(io :: IO, :: StreamFormat{S, F}) where {S, F <: Format}
  stream = S(io)
  unpack(stream, F())
end

function unpack(io :: IO, :: Type{T}, :: StreamFormat{S, F}) where {T, S, F <: Format}
  stream = S(io)
  unpack(stream, T, F())
end


#
# AliasFormat
#

struct AliasFormat{S} <: Format end

function pack(io :: IO, value :: T, :: AliasFormat{S}) :: Nothing where {S, T}
  val = destruct(value, AliasFormat{S}())
  pack(io, val)
end

function unpack(io :: IO, :: Type{T}, :: AliasFormat{S}) :: T where {S, T}
  val = unpack(io, S)
  construct(T, val, AliasFormat{S}())
end

construct(:: Type{T}, val, :: AliasFormat) where {T} = T(val)
destruct(val :: T, :: AliasFormat{S}) where {S, T} = S(val)

