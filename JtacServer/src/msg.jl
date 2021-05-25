
module Msg

import ..JtacServer: Events

import Serialization
import LazyJSON

abstract type Role end

abstract type Train <: Role end
abstract type Play  <: Role end
abstract type Watch  <: Role end

"""
Message between source S and destination D. The two modes of communication are
between jtac train and play instances, as well as between jtac train and
watch instances.

For efficiency, the former mode of communication (Train <--> Play) mostly uses
a binary format provided by julia's serialization capability. To make monitoring
jtac training transparent and extensible, the latter mode of communication
(Train <--> Watch) uses a simple JSON format. 
"""
abstract type Message{S <: Role, D <: Role} end

"""
    send(socket, message)

Sending a Jtac service message to socket.
"""
function send(socket, msg :: Message)
  Serialization.serialize(socket, msg)
end

"""
    receive(socket, type)

Receiving a Jtac service message from `socket` and asserting that it is of type
`type`. Only apply this to trusted sockets, since arbitrary code may be
executed.
"""
function receive(socket, M :: Type{<: Message})
  v = Serialization.deserialize(socket)
  v isa M ? v : nothing
end

"""
Login messages are always exchanged via JSON for both watch and play
instances.
"""
abstract type Login{S <: Role} end

send(socket, msg :: Login) = println(socket, LazyJSON.jsonstring(msg))

function receive(socket, :: Type{Login})
  try
    value = LazyJSON.value(readline(socket))
    if value["kind"] == "play"
      convert(FromPlay.Login, value)
    elseif value["kind"] == "watch"
      convert(FromWatch.Login, value)
    else
      Log.error("Login of kind $(value["kind"]) rejected")
      nothing
    end
  catch err
    # here we should provide some information about *what* is going wrong
    nothing
  end
end

struct LoginAuth
  accept :: Bool
  msg :: String
  session :: String
end

send(socket, msg :: LoginAuth) = println(socket, LazyJSON.jsonstring(msg))

function receive(socket, :: Type{LoginAuth})
  line = readline(socket)
  try
    value = LazyJSON.value(line)
    convert(LoginAuth, value)
  catch
    nothing
  end
end


#
# Messages train -> play
#

module ToPlay

  using Jtac
  import ...JtacServer: compress
  import ..Msg
  import ..Msg: Train, Play

  """
  Asking a jtac serve instance to disconnect. After some timeout, the connection
  should be closed if the serve instance does not react.
  """
  struct Disconnect <: Msg.Message{Train, Play}
    text :: String
  end

  """
  Asking a jtac serve instance to reconnect after some timeout. This can be used
  if the jtac train instance needs to restart / pause, but activity should be
  resumed automatically afterwards.
  """
  struct Reconnect <: Msg.Message{Train, Play}
    text :: String
    timeout :: Float64 # seconds before the reconnection should be attempted
  end

  """
  Asking a jtac serve instance to discard the current data request. Contest
  requests are not affected.
  """
  struct Idle <: Msg.Message{Train, Play}
    text :: String
  end

  """
  Requesting the generation of training data from a jtac serve instance.

  The message encapsulates the player to be used for self play (as `PlayerSpec`)
  and other information concerning the requested data (branching, number of
  playings, ...).
  """
  struct DataReq <: Msg.Message{Train, Play}

    _spec :: Vector{UInt8}

    # Training options
    init_steps   :: Tuple{Int, Int}
    branch       :: Float64
    branch_steps :: Tuple{Int, Int}
    augment      :: Bool

    # Dataset options
    min_playings :: Int
    max_playings :: Int

    # Id of the instruction set
    reqid        :: Int

  end

  option_tuple(a :: Int) = (a, a)
  option_tuple(b) = (b[1], b[2])

  function DataReq( reqid :: Int
                  , player :: Player.MCTSPlayer{G}
                  ; augment = true
                  , init_steps = 0
                  , branch = 0.
                  , branch_steps = 1
                  , min_playings = 1
                  , max_playings = 10000
                  ) where {G <: Game.AbstractGame}

    @assert !isabstracttype(G)

    DataReq( Player.PlayerSpec(player) |> compress
           , option_tuple(init_steps)
           , branch
           , option_tuple(branch_steps)
           , augment
           , min_playings
           , max_playings
           , reqid )
  end

  """
      DataReq(id, context, model)

  Derive a data request based on a model and training context.
  """
  function DataReq(id, ctx, model :: Model.AbstractModel{<: Game.AbstractGame, false})

    player = Player.MCTSPlayer( model |> Model.to_cpu
                              , power = ctx.power
                              , temperature = ctx.temperature
                              , exploration = ctx.exploration
                              , dilution = ctx.dilution
                              , name = ctx.name * "-$id" )

    DataReq( id, player
           ; augment = ctx.augment
           , init_steps = ctx.init_steps
           , branch = ctx.branch
           , branch_steps = ctx.branch_steps
           , min_playings = ctx.min_playings
           , max_playings = ctx.max_playings )
  end



  """
  Confirmation by the jtac train instance that the upload of a data record was
  successful.
  """
  struct DataConfirm <: Msg.Message{Train, Play}
    id :: Int
  end


  """
  Requesting the generation of contest data from a jtac serve instance.
  """
  struct ContestReq <: Msg.Message{Train, Play}
    _specs :: Vector{UInt8}
    active :: Vector{Int} 
    names  :: Vector{String}
    length :: Int
    era    :: Int
    reqid  :: Int
  end

  function ContestReq( reqid :: Int
                     , players :: Vector{<: Player.MCTSPlayer}
                     , length :: Int
                     , active = 1:length(players) )
    # Make sure that a concrete, consistent game type can be derived
    Player.derive_gametype(players)
    specs = Player.PlayerSpec.(players)
    names = Player.name.(players)
    ContestReq(specs |> compress, active, names, length, reqid)
  end


  """
  Confirmation from the train service that the upload of a contest record was
  successful.
  """
  struct ContestConfirm <: Msg.Message{Train, Play}
    id :: Int
  end

end # module ToPlay

#
# Messages play -> train
#

module FromPlay

  using Jtac
  import ...JtacServer: JTAC_VERSION, Events, compress
  import ..Msg
  import ..Msg: Play, Train


  """
  Login request of a play service.

  This must be the initial message to a train service. The train service then
  decides whether to accept the connection or not. If it accepts, a
  """
  struct Login <: Msg.Login{Play}
    name    :: String
    token   :: String
    version :: String
    kind    :: String

    accept_data    :: Bool
    accept_contest :: Bool
  end

  function Login(name, token, data = true, contest = true)
    Login(name, token, JTAC_VERSION, "play", data, contest)
  end

  """
  Logout message of a play service
  """
  struct Logout <: Msg.Message{Play, Train}
    text :: String
  end

  """
  Record of training data generated by the jtac serve instance.

  It contains the compressed data sets and the id of the corresponding data
  request.
  """
  mutable struct Data <: Msg.Message{Play, Train}
    _data    :: Vector{UInt8} # compressed Jtac datasets
    states   :: Int
    playings :: Int
    reqid    :: Int
    id       :: Int
    time     :: Float64
  end

  function Data(data :: Vector{<:Training.Dataset}, reqid, id, time)
    _data = compress(data)
    states = sum(length.(data))
    playings = length(data)
    Data(_data, states, playings, reqid, id, time)
  end

  """
      DataBody(client, data)

  Generate a data message body from the `data` answer received from `client`.
  """
  function Events.DataBody(client, d :: Data)
    mb = sizeof(d._data) / 1024 / 1024
    Events.DataBody(d.reqid, client, d.states, d.playings, mb, d.time)
  end


  """
  Record of contest data generated by the jtac serve instance.

  It contains the contest results and the id of the corresponding contest request.
  """
  mutable struct Contest <: Msg.Message{Play, Train}
    elo   :: Vector{Float64}
    data  :: Vector{Int}
    draw  :: Float64
    sadv  :: Float64
    reqid :: Int
    id    :: Int
    time  :: Float64
  end

  """
      ContestBody(client, req, data)

  Generate a contest message body from `req` and the `data` answer received from `client`.
  """
  function Events.ContestBody(client, r :: Msg.ToPlay.ContestReq, d :: Contest)
    balance = [d.data[i,:,1] .- d.data[i,:,3] for i in 1:size(d.data, 1)]
    Events.ContestBody(d.reqid, client, r.era, r.names,
                       sum(abs, d.data), balance, d.elo, d.draw, d.sadv)
  end

end # module FromPlay

#
# Custom serialization of messages between serve and train
#

#const BitsType = Union{UInt8, Int, Float64, Bool}
#
#serial(io, v :: BitsType) = write(io, v)
#serial(io, v :: String) = write(io, length(v), v)
#serial(io, v :: Tuple{Int, Int}) = write(io, v[1], v[2])
#serial(io, v :: Vector{T}) where {T <: BitsType} = write(io, length(v), v)
#serial(io, v :: Vector{String}) = (write(io, length(v)); for s in v serial(io, s) end)
#                                   
#
#deserial(io, :: Type{String}) = String(read(io, read(io, Int)))
#deserial(io, :: Type{T}) where {T <: BitsType} = read(io, T)
#deserial(io, :: Type{Tuple{Int, Int}}) = (read(io, Int), read(io, Int))
#deserial(io, :: Type{Vector{T}}) where {T <: BitsType} = convert(Vector{T}, read(io, sizeof(T)*read(io, Int)))
#deserial(io, :: Type{Vector{String}}) = (n = read(io, Int); [deserial(io, String) for _ in 1:n])
#
#const msgtypes = Dict(
#    0 => ServeLogout
#  , 1 => ServeData
#  , 2 => ServeContest
#  , 3 => TrainDisconnectServe
#  , 4 => TrainReconnectServe
#  , 5 => TrainIdleServe
#  , 6 => TrainDataServe
#  , 7 => TrainDataConfirmServe
#  , 8 => TrainContestServe
#  , 9 => TrainContestConfirmServe
#)
#
#function send(io, v :: Message)
#  tid = findfirst(isequal(typeof(v)), msgtypes)
#  # DEBUGGING
#  out = open("send.bin", "a")
#  for io in [out, io]
#    write(io, tid)
#    for field in Base.fieldnames(typeof(v))
#      serial(io, Base.getfield(v, field))
#    end
#  end
#  close(out)
#end
#
#function receive(io, T)
#  open("receive.bin", "a") do recv
#    tid = read(io, Int)
#    write(recv, tid)
#    t = msgtypes[tid]
#    @assert t <: T
#    args = map(Base.fieldnames(t)) do field
#      v = deserial(io, Base.fieldtype(t, field))
#      serial(recv, v)
#      v
#    end
#    t(args...)
#  end
#end


#
# Messages train <-> watch
#
# Note that communication to / from watch clients are conducted via plain json
# encoded events, so we do not introduce new types 

Msg.send(socket, ev :: Events.Event) = println(socket, Events.json(ev)) 

function Msg.receive(sock, :: Type{Events.Event})
  Events.parse(readline(sock), Events.Event)
end

module FromWatch

  import ..Msg
  import ..Msg: Watch

  """
  Login request of a watch service.
  """
  struct Login <: Msg.Login{Watch}
    name    :: String
    token   :: String
    version :: String
    kind    :: String
  end

end

end # module Msg
