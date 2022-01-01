
# By default, MsgPack does currently not use binary formats for vectors.
MsgPack.msgpack_type( :: Type{Vector{UInt8}}) = MsgPack.BinaryType()
MsgPack.to_msgpack( :: MsgPack.BinaryType, x :: Vector{UInt8}) = x
MsgPack.from_msgpack( :: Type{Vector{UInt8}}, bytes :: Vector{UInt8}) = bytes


const BinType = Union{Int8, Int32, Int64, UInt32, UInt64, Float32, Float64}

MsgPack.msgpack_type( :: Type{Vector{T}}) where T <: BinType = MsgPack.BinaryType()
function MsgPack.to_msgpack( :: MsgPack.BinaryType, x :: Vector{T}) where T <: BinType
  # TODO: collect may be left away after a performance regression in julia has been
  # fixed!
  collect(reinterpret(UInt8, x))
end

function MsgPack.from_msgpack( :: Type{Vector{T}}, bytes :: Vector{UInt8}) where T <: BinType
  collect(reinterpret(T, bytes))
end

