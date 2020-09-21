
const JTAC_SERVER_VERSION = "v0.1"

#
# Imports
#

using Jtac

import Sockets: @ip_str, connect, listen, TCPSocket
import Serialization
import Distributed
import Distributed: RemoteChannel
import SharedArrays
import Blosc

# This algorithm worked (by far) the best to get small representations on
# MetaTac datasets, and encoding / decoding did not take considerably longer
# than with the others
Blosc.set_compressor("zstd")

abstract type JtacService end
abstract type Train   <: JtacService end
abstract type Play    <: JtacService end
abstract type Control <: JtacService end

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
sources. 
"""
function decompress(data)
  buf = IOBuffer()
  write(buf, Blosc.decompress(UInt8, data))
  seekstart(buf)
  Serialization.deserialize(buf)
end

"""
Specification of an MCTS or Intuition player which is used to send data and
contest requests to play services.
"""
struct PlayerSpec

  # Compressed reference model
  _model :: Vector{UInt8}

  # Player parameters
  power         :: Int      # power = 0 is used for IntuitionPlayers
  temperature   :: Float32
  exploration   :: Float32
  dilution      :: Float32
  name          :: String

end

function PlayerSpec(player :: MCTSPlayer)

  _model = base_model(player) |> to_cpu |> compress
  PlayerSpec( _model, player.power, player.temperature
            , player.exploration, player.dilution, player.name )
end

function PlayerSpec(player :: IntuitionPlayer)

  ref_model_ = base_model(player) |> to_cpu |> compress
  PlayerSpec(ref_model_, 0, player.temperature, 0., 0., player.name)

end

"""
    get_player(spec; gpu = false, async = false)

Derive a player from a specification `spec`. The model of the player is
transfered to the gpu or brought in async mode if the respective flags are set.
"""
function get_player(spec :: PlayerSpec; gpu = false, async = false)

  model = spec._model |> decompress

  if model isa NeuralModel 

    model = gpu   ? to_gpu(model) : model
    model = async ? Async(model)  : model

  end

  if spec.power <= 0

    IntuitionPlayer( model
                   , temperature = spec.temperature
                   , name = spec.name )
  else

    MCTSPlayer( model
              , power = spec.power
              , temperature = spec.temperature
              , exploration = spec.exploration
              , dilution = spec.dilution
              , name = spec.name )
  end

end

#
# Communication between services
#

include("msg.jl")

#
# Default implementation of a Jtac play service
#

include("play.jl")

#
# Default implementation of a Jtac train service
#

include("train.jl")

end # module Server

