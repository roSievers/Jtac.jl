
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

"""
Specification of an MCTS or Intuition player used to communicate with jtac play
instances.
"""
struct PlayerSpec

  # Compressed reference model
  model :: Jtac.NeuralModel

  # Player parameters
  power       :: Int      # power = 0 is used for IntuitionPlayers
  temperature :: Float32
  exploration :: Float32
  dilution    :: Float32
  name        :: String

end

function PlayerSpec(player :: Jtac.MCTSPlayer)
  model = Jtac.base_model(player) |> Jtac.to_cpu
  PlayerSpec( model, player.power, player.temperature
            , player.exploration, player.dilution, player.name )
end

function PlayerSpec(player :: Jtac.IntuitionPlayer)
  model = Jtac.base_model(player) |> Jtac.to_cpu
  PlayerSpec(model, 0, player.temperature, 0., 0., player.name)
end

"""
    build_player(spec; gpu = false, async = false)

Derive a player from a specification `spec`. The model of the player is
transfered to the gpu or brought in async mode if the respective flags are set.
"""
function build_player(spec :: PlayerSpec; gpu = false, async = false)
  model = spec.model
  if model isa Jtac.NeuralModel 
    model = gpu   ? Jtac.to_gpu(model) : model
    model = async ? Jtac.Async(model)  : model
  end
  if spec.power <= 0
    Jtac.IntuitionPlayer( model
                        , temperature = spec.temperature
                        , name = spec.name )
  else
    Jtac.MCTSPlayer( model
                   , power = spec.power
                   , temperature = spec.temperature
                   , exploration = spec.exploration
                   , dilution = spec.dilution
                   , name = spec.name )
  end
end

