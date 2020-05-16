
module Server

#
# Imports
#

using ..Jtac

import Sockets: @ip_str, connect, listen, TCPSocket
import Serialization
import Distributed
import Distributed: RemoteChannel
import Blosc

function __init__()

  # This algorithm worked (by far) the best to get small representations on
  # MetaTac datasets, and encoding / decoding did not take considerably longer
  # as with the others
  Blosc.set_compressor("zstd")

end

#
# Messages to communicate between train server and play client
#

abstract type Msg end

# Sending an receiving messages. Note that receive is not safe, as it allows
# arbitrary code execution. So it must only be used on messages from trusted
# sources.
function send(socket, msg :: Msg)
  Serialization.serialize(socket, msg)
end

function receive(socket) :: Msg
  Serialization.deserialize(socket)
end

# The following functions can be used to compress parts of a Msg if it contains
# a large amount of data
function compress(value) :: Vector{UInt8}
  buf = IOBuffer()
  Serialization.serialize(buf, value)
  Blosc.compress(take!(buf))
end

# Like receive, this is not safe and must only be used on data from a trusted source.
function decompress(data)
  buf = IOBuffer()
  write(buf, Blosc.decompress(UInt8, data))
  seekstart(buf)
  Serialization.deserialize(buf)
end

#
# Client messages
#

abstract type ClientMsg <: Msg end

struct Login <: ClientMsg
  token   :: String
  name    :: String
  version :: Int

  accept_data_requests    :: Bool
  accept_contest_requests :: Bool

  function Login(; token, name, data = true, contest = true)
    new(token, name, CLIENT_VERSION, data, contest)
  end
end

struct Logout <: ClientMsg
  msg :: String
end

struct DataRecord <: ClientMsg
  data_  :: Vector{UInt8} # compressed Jtac datasets
  reqid  :: Int
  id     :: Int
  dtime  :: Float64
end

struct ContestRecord <: ClientMsg
  data   :: Array{Int, 3}
  reqid  :: Int
  id     :: Int
  dtime  :: Float64
end


#
# Server messages
#

abstract type ServerMsg <: Msg end

struct Reply <: ServerMsg
  accept :: Bool
  msg    :: String
end

struct Disconnect <: ServerMsg
  msg :: String
end

struct Idle <: ServerMsg
  msg :: String
end

struct Reconnect <: ServerMsg
  min_wait_time :: Float64
end

struct DataConfirmation <: ServerMsg
  id :: Int
end

struct ContestConfirmation <: ServerMsg
  id :: Int
end


#
# Server Request messages
#

struct PlayerSpec

  # Decomposed and compressed reference model
  ref_model_    :: Vector{UInt8}

  # Player parameters
  power         :: Int      # power = 0 is used for IntuitionPlayers
  temperature   :: Float32
  exploration   :: Float32
  dilution      :: Float32
  name          :: String

end

function PlayerSpec(player :: MCTSPlayer)

  ref_model_ = base_model(player) |> to_cpu |> Jtac.decompose |> compress
  PlayerSpec( ref_model_, player.power, player.temperature
            , player.exploration, player.dilution, player.name )
end

function PlayerSpec(player :: IntuitionPlayer)

  ref_model_ = base_model(player) |> to_cpu |> Jtac.decompose |> compress
  PlayerSpec(ref_model_, 0, player.temperature, 0., 0., player.name)

end


# Given a spec, a (gpu/async) player can be generated
function get_player(spec :: PlayerSpec; gpu = false, async = false)

  model = spec.ref_model_ |> decompress |> Jtac.compose

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


struct DataRequest <: ServerMsg

  spec :: PlayerSpec

  # Training options
  prepare_steps :: Tuple{Int, Int}
  branch_prob   :: Float64
  branch_steps  :: Tuple{Int, Int}
  augment       :: Bool

  # Dataset options
  min_playings  :: Int
  max_playings  :: Int

  # Hash of the instruction set
  id            :: Int

  function DataRequest( player :: MCTSPlayer{G}
                      ; id = rand(UInt16)    # TODO: change this for production
                      , augment = true
                      , prepare_steps = 0
                      , branch_prob = 0.
                      , branch_steps = 1
                      , min_playings = 1
                      , max_playings = 10000
                      ) where {G <: Game}

    @assert !isabstracttype(G)

    new( PlayerSpec(player)
       , Jtac.tup(prepare_steps)
       , branch_prob
       , Jtac.tup(branch_steps)
       , augment
       , min_playings
       , max_playings
       , id )

  end

end


struct ContestRequest <: ServerMsg

  specs  :: Vector{PlayerSpec}
  active :: Vector{Int} 
  length :: Int
  id     :: Int

end

function ContestRequest( players :: Vector{<: MCTSPlayer}
                       , length :: Int
                       , active = 1:length(players)
                       ; id = rand(UInt16) )

  # Make sure that a concrete, consistent game type can be derived
  Jtac.derive_gametype(players)

  specs = PlayerSpec.(players)
  ContestRequest(specs, active, length, id)

end



#
# Play Client
#

include("play.jl")


#
# Train Server
#

include("train.jl")

end # module Server

