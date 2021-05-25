
# This algorithm worked (by far) the best to get small representations on
# MetaTac datasets, and encoding / decoding did not take considerably longer
# than with the others
Blosc.set_compressor("zstd")

"""
    compress(value) :: Vector{UInt8}

Compresses a generic julia value with the zstd algorithm.
"""
function compress(value) :: Vector{UInt8}
  buf = IOBuffer()
  Serialization.serialize(buf, value)
  Blosc.compress(take!(buf))
end

"""
    decompress(data)

Restores a compressed julia type from data. Only apply this to data from trusted
sources, since arbitrary code may be executed. 
"""
function decompress(data)
  buf = IOBuffer()
  write(buf, Blosc.decompress(UInt8, data))
  seekstart(buf)
  Serialization.deserialize(buf)
end

