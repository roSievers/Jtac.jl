
"""
    pack(x)

Return the serialized value of `x` in the msgpack format.

See also [`pack_compressed`](@ref) and [`unpack`](@ref).
"""
function pack(x)
  buf = IOBuffer()
  pack(buf, x)
  take!(buf)
end

"""
    unpack(data, T)

Unpack the packed data `data` as value of type `T`.

See also [`pack`](@ref).
"""
function unpack(bin :: Vector{UInt8}, T :: Type)
  unpack(IOBuffer(bin), T)
end

pack(args...) = MsgPack.pack(args...)
unpack(args...) = MsgPack.unpack(args...)

"""
    pack_compressed([io, ] value)

Serialize `value` with additional Zstd compression.

See also [`pack`](@ref) and [`unpack_compressed`](@ref).
"""
function pack_compressed(io :: IO, value)
  stream = ZstdCompressorStream(io)
  pack(stream, value)
  write(stream, TOKEN_END)
  flush(stream)
end

function pack_compressed(value)
  io = IOBuffer()
  pack_compressed(io, value)
  io.data
end

"""
    unpack_compressed(io / data, value)

Unpack a value that was packed via [`pack_compressed`](@ref).

See also [`unpack`](@ref).
"""
unpack_compressed(io :: IO, T) =
  unpack(ZstdDecompressorStream(io), T)

unpack_compressed(data :: Vector{UInt8}, T) =
  unpack_compressed(IOBuffer(data), T)


"""
    freeze(x)

Freeze the value `x`. This function is called before custom packing is applied,
and the return value is used for packing instead of `x`. For example, this can
be used to automatically convert models to the CPU before saving or to resolve
pointers in game states.
"""
freeze(x :: Any) = x

"""
    unfreeze(x)

Undo `freeze(x)`. This function is called after custom unpacking.
"""
unfreeze(x :: Any) = x


"""
    array_length(io)

Get the length of a msgpack array stored in `io`. Consumes all bytes with length
information.
"""
function array_length(io)
  byte = read(io, UInt8)
  if byte >> 4 == 0x09 # fixed array 16
    (byte << 4 >> 4) |> Base.ntoh |> Int # remove first 4 bytes and handle endianness
  elseif byte == 0xdc # array with length <= 2^16 - 1
    read(io, UInt16) |> Base.ntoh |> Int
  elseif byte == 0xdd # array with length <= 2^32 - 1
    read(io, UInt32) |> Base.ntoh |> Int
  else
    ArgumentError("array_length: invalid msgpack array byte") |> throw
  end
end

function write_array_format(io, n :: Int)
  if n <= 15
    write(io, 0x90 | UInt8(n))
  elseif n <= typemax(UInt16)
    write(io, 0xdc); write(io, hton(UInt16(n)))
  elseif n <= typemax(UInt32)
    write(io, 0xdd); write(io, hton(UInt32(n)))
  else
    ArgumentError("write_array_format: invalid number $n of fields") |> throw
  end
end

"""
    map_length(io)

Get the length of a msgpack map stored in `io`. Consumes all bytes with length
information.
"""
function map_length(io)
  byte = read(io, UInt8)
  if byte >> 4 == 0x08 # fixed map 16
    (byte << 4 >> 4) |> Base.ntoh |> Int
  elseif byte == 0xde # map with length <= 2^16 - 1
    read(io, UInt16) |> Base.ntoh |> Int
  elseif byte == 0xdf # map with length <= 2^32 - 1
    read(io, UInt32) |> Base.ntoh |> Int
  else
    ArgumentError("map_length: invalid msgpack array byte") |> throw
  end
end

function write_map_format(io, n :: Int)
  if n <= 15
    write(io, 0x80 | UInt8(n))
  elseif n <= typemax(UInt16)
    write(io, 0xde); write(io, hton(UInt16(N)))
  elseif n <= typemax(UInt32)
    write(io, 0xdf); write(io, hton(UInt32(N)))
  else
    ArgumentError("write_map_format: invalid number $n of fields") |> throw
  end
end

"""
    bin_length(io)

Get the length of a msgpack byte vector stored in `io`. Consumes all bytes with
length information.
"""
function bin_length(io)
  byte = read(io, UInt8)
  if byte == 0xc4 # bin with length <= 2^8 - 1
    read(io, UInt8) |> Base.ntoh |> Int
  elseif byte == 0xc5 # bin with length <= 2^16 - 1
    read(io, UInt16) |> Base.ntoh |> Int
  elseif byte == 0xc6 # bin with length <= 2^32 - 1
    read(io, UInt32) |> Base.ntoh |> Int
  else
    ArgumentError("bin_length: invalid msgpack bin byte") |> throw
  end
end

function write_bin_format(io, n :: Int)
  if n <= typemax(UInt8)
    write(io, 0xc4); write(io, hton(UInt8(n)))
  elseif n <= typemax(UInt16)
    write(io, 0xc5); write(io, hton(UInt16(n)))
  elseif n <= typemax(UInt32)
    write(io, 0xc6); write(io, hton(UInt32(n)))
  else
    ArgumentError("write_bin_format: invalid number $n of fields") |> throw
  end
end

"""
Auxiliary type that allows storing data structures in binary format. Relies on
the `binary` format of the msgpack protocol.
"""
struct Bytes
  data :: Vector{UInt8}
end

function pack(io :: IO, val :: Bytes)
  write_bin_format(io, length(val.data))
  write(io, val.data)
end

function unpack(io :: IO, :: Type{Bytes})
  n = bin_length(io)
  Bytes(read(io, n))
end


Bytes(arr :: Array) = Bytes(reinterpret(UInt8, reshape(arr, :)))
Base.convert(:: Type{Bytes}, bytes :: Vector{UInt8}) = Bytes(bytes)


"""
Auxiliary type that allows storing julia Float32 arrays in binary format. Relies
on the `binary` format of the msgpack protocol.
"""
struct BinArray{F <: Real}
  data :: Array{F}
end

BinArray(arr :: Array) = BinArray(arr)

function pack(io :: IO, val :: BinArray{F}) where {F}
  write_map_format(io, 3)
  pack(io, :type)
  pack(io, nameof(F))
  pack(io, :size)
  pack(io, size(val.data))
  pack(io, :data)
  data = reinterpret(UInt8, reshape(val.data, :))
  write_bin_format(io, length(data))
  write(io, data)
end

function unpack(io :: IO, :: Type{BinArray})
  @assert map_length(io) == 3
  @assert unpack(io, Symbol) == :type
  type = unpack(io, Symbol)
  F = Base.eval(type)
  @assert F <: Real
  @assert unpack(io, Symbol) == :size
  size = unpack(io, Vector{Int})
  @assert unpack(io, Symbol) == :data
  n = bin_length(io)
  data = read(io, n)
  arr = reshape(reinterpret(F, data), Tuple(size))
  arr = convert(Array{F}, arr)
  BinArray(arr)
end

function unpack(io :: IO, :: Type{BinArray{F}}) where {F}
  unpack(io, BinArray) :: BinArray{F}
end


"""
    fieldnames(T)

Get the field names of an instance of type `T` that are included in (typed or
untyped) packing. See also the convenience macro `Pack.@only`.

This function can be specialized (together with `Pack.fieldvalues` and
`Pack.fieldytpes`) in order to provide custom packing / unpacking.
"""
fieldnames(T :: Type) = Base.fieldnames(T)

"""
    binarrayfieldnames(T)

Returns a vector of field names of `T` that are to be packed in binary format.
The corresponding fields must be of type `Array{<: Real}`.
"""
binarrayfieldnames(_) = Symbol[]

"""
    fieldvalues(val)

Return a vector of entries that are packed when `val` is packed (typed or
untyped). The entries of this vector correspond to the names returned by
`fieldnames(typeof(val))`.

This function can be specialized (together with `Pack.fieldnames` and
`Pack.fieldytpes`) in order to provide custom packing / unpacking.
"""
function fieldvalues(val)
  names = fieldnames(typeof(val))
  (Base.getfield(val, name) for name in names)
end

"""
    fieldtypes(T)

Returns a vector of types that are used as type information for unpacking
instances of `T`.

This function can be specialized (together with `Pack.fieldnames` and
`Pack.fieldvalues`) in order to provide custom packing / unpacking.
"""
function fieldtypes(T :: Type)
  names = fieldnames(T)
  (Base.fieldtype(T, name) for name in names)
end

"""
    construct(T, args...)

Construct values of type `T` based on arguments `args...`. This function should
be defined for any type that implements custom packing / unpacking via
specializing `fieldnames`, `fieldtypes`, and `fieldvalues`.
"""
construct(T :: Type, args...) = T(args...)

"""
    @only T fields

When packing instances of type `T`, ignore fields other than the ones provided
in `fields`. In this case, a constructor that accepts the respective fields as
positional arguments (in the order specified in `fields`) must be provided.

Note that this macro must only be applied to (semi-)concrete subtypes of a
type on which `@untyped` or `@typed` has been applied.
"""
macro only(T, fields)
  quote
    Pack.fieldnames(:: Type{<: $T}) = $fields
  end |> esc
end

"""
    @binarray T fields 

Mark the fields `fields` of type `T` as arrays to be saved in binary mode. The
respective fields must be of type `Array{<: Real}`.
"""
macro binarray(T, fields)
  quote
    Pack.binarrayfieldnames(:: Type{<: $T}) = $fields
  end |> esc
end


"""
    @vector T

Enables packing / unpacking of values with type `Vector{S}` as long as instances
of type `S <: T` can be packed / unpacked.
"""
macro vector(T)
  S = Base.gensym(:S)
  quote
    Pack.pack(io :: IO, val :: Vector{<: $T}) = Pack.pack_vector(io, val)
    Pack.unpack(io :: IO, :: Type{Vector{$S}}) where {$S <: $T} =
      Pack.unpack_vector(io, Vector{$S})
  end |> esc
end

function pack_vector(io :: IO, vec :: AbstractVector)
  write_array_format(io, length(vec))
  for v in vec
    pack(io, v)
  end
end

function unpack_vector(io :: IO, :: Type{Vector{S}}) where {S}
  n = array_length(io)
  vec = Vector{S}(undef, n)
  for i in 1:n
    vec[i] = unpack(io, S)
  end
  vec
end


"""
    @nullable T

Enables unpacking of values of type `Union{Nothing, <: T}`.
"""
macro nullable(T)
  S = Base.gensym(:S)
  quote
    Pack.unpack(io :: IO, :: Type{Union{Nothing, $S}}) where {$S <: $T} =
      Pack.unpack_nullable(io, $S)
  end |> esc
end

function unpack_nullable(io :: IO, S :: Type)
  if peek(io, UInt8) == 0xc0
    read(io, UInt8)
    nothing
  else
    unpack(io, S)
  end
end


"""
    @named T

Enable packing of named values with type `T` as their scope type.
As a consequence, only values of type `T` that have previously been registered
via [`Util.register!`](@ref) can be serialized / deserialized.
"""
macro named(T)
  quote
    Pack.@vector $T

    function Pack.pack(io :: IO, val :: $T)
      name = Util.lookupname($T, val)
      Pack.pack(io, name)
    end

    function Pack.unpack(io :: IO, :: Type{<: $T})
      name = Pack.unpack(io, Symbol)
      Util.lookup($T, name)
    end
  end |> esc
end

"""
    @untyped T

Enable packing of all concrete subtypes `S <: T` as msgpack map type. Note that
unpacking requires prior knowledge of the concrete type `S`. For more flexible
(albeit slower) unpacking, see `@typed`, which additionally stores type
information.
"""
macro untyped(T)
  S = Base.gensym(:S)
  quote
    Pack.@vector $T
    Pack.@nullable $T
    Pack.pack(io :: IO, val :: $T) = Pack.pack_untyped(io, val)
    Pack.unpack(io :: IO, $S :: Type{<: $T}) = Pack.unpack_untyped(io, $S)
  end |> esc
end

function pack_untyped(io :: IO, val :: S) where {S}
  val = freeze(val)
  fields = fieldvalues(val)
  names = fieldnames(S)
  binnames = binarrayfieldnames(S)
  @assert length(names) == length(fields)
  write_map_format(io, length(names))
  for (name, field) in zip(names, fields)
    pack(io, name)
    if name in binnames
      pack(io, BinArray(field))
    else
      pack(io, field)
    end
  end
end

function unpack_untyped(io :: IO, T :: Type)
  types = fieldtypes(T)
  names = fieldnames(T)
  binnames = binarrayfieldnames(T)
  @assert map_length(io) == length(names) == length(types)

  args = map(names, types) do name, type
    @assert unpack(io, Symbol) == name
    if name in binnames
      F = eltype(type)
      unpack(io, BinArray{F}).data
    else
      unpack(io, type)
    end
  end

  val = construct(T, args...)
  Pack.unfreeze(val)
end


"""
    @typed T

Enable typed packing for all subtypes of the type `T`. Instead of only storing
the fields of `val :: T` (see `@untyped T`), typed packing stores a msgpack map
of the following layout:

    { type : typeinfo, (fields of val)... }

The entry `typeinfo` is created via `decompose(typeof(val))` during packing, and
converted into a type again via `compose(typeinfo)` during unpacking.

Since the type is explicitly stored and can be retrieved, the following pack
/ unpack cycle also works when `S <: T` is abstract:

    bytes = pack(val :: S)
    bytes .== pack(unpack(bytes, S)) |> all
"""
macro typed(T)
  S = Base.gensym(:S)
  R = Base.gensym(:R)
  quote
    Pack.@vector $T 
    Pack.@nullable $T
    Pack.pack(io :: IO, val :: $T) = Pack.pack_typed(io, val)
    Pack.unpack(io :: IO, $S :: Type{<: $T}) = Pack.unpack_typed(io, $S)
  end |> esc
end

function pack_typed(io :: IO, val)
  val = Pack.freeze(val)
  S = typeof(val)
  fields = fieldvalues(val)
  names = fieldnames(S)
  binnames = binarrayfieldnames(S)
  @assert length(names) == length(fields)

  write_map_format(io, length(names) + 1)
  pack(io, "type")
  pack(io, Pack.decompose(S))

  for (name, field) in zip(names, fields)
    pack(io, name)
    if name in binnames
      pack(io, BinArray(field))
    else
      pack(io, field)
    end
  end
end

function unpack_typed(io :: IO, T :: Type)
  n = map_length(io) - 1
  @assert unpack(io, Symbol) == :type
  S = compose(unpack(io), T)
  @assert S <: T "unexpected type $S when unpacking $T"
  
  types = fieldtypes(S)
  names = fieldnames(S)
  binnames = binarrayfieldnames(S)
  @assert n == length(names) == length(types)

  args = map(names, types) do name, type
    @assert unpack(io, Symbol) == name
    if name in binnames
      F = eltype(type)
      unpack(io, BinArray{F}).data
    else
      unpack(io, type)
    end
  end

  val = construct(S, args...)
  Pack.unfreeze(val)
end

"""
    typename(T)

Snake-case representation of the type `T`. Used for nicer type entries only.
"""
function typename(T :: Type)
  name = Base.nameof(T) |> string
  join(lowercase.(split(name, r"(?=[A-Z])")), "_")
end

function typepath(T :: Type)
  mod = Base.parentmodule(T)
  name = Base.nameof(T)
  string(mod) * "." * string(name)
end

function parsetypepath(str, T = Main)
  syms = map(Symbol, split(str, "."))
  for sym in syms
    T = getfield(T, sym)
  end
  T :: Type
end


"""
    typeparams(T)

Return the type parameters of a type `T`.

If `T` has more type parameters than are specified (e.g., `T = Array{Float32}`),
the specified type parameters must come first (e.g., `T = Array{F, 1} where {F}`
would fail).
"""
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


"""
    decompose(T)

Generic decomposition of a type `T` into a dictionary that can be packed and
unpacked. Type parameters are also stored. Note that `T` as well as all type
parameters must be accessible from the module `Main` in the context of
unpacking.
"""
function decompose(T :: Type)
  d = Dict()
  d["name"] = typename(T)
  d["path"] = typepath(T)
  params = typeparams(T)
  if !isempty(params)
    d["params"] = params .|> decompose
  end
  d
end

decompose(x :: Union{Int, Bool, Symbol}) = x

"""
    compose(dict [, T])

Compose the type stored as a dictionary `dict` that was created by `decompose`.
The type argument `T` can be specified to make sure that the composed type is in
fact a subtype of `T`.

Note that composition only works if all types and type parameters stored in
`dict` are accessible from the module `Main`.
"""
function compose(d :: Dict, T :: Type = Any)
  S = parsetypepath(d["path"])
  if haskey(d, "params") && !isempty(d["params"])
    params = map(x -> compose(x), d["params"])
    S = S{params...}
  end
  @assert S <: T "$S is not a subtype of $T"
  S
end

compose(x :: Bool) = x
compose(x :: Integer) = Int(x)
compose(x :: String) = Symbol(x) # a packing / unpacking cycle converts symbols to strings
