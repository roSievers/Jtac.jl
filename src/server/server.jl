
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

  Blosc.set_compressor("zstd")

end

#
# Messages to communicate between train server and play client
#

abstract type Msg end

# TODO: the temporary buffers don't look particularly efficient, but I am not
# sure how to prevent them
function send(socket, msg :: Msg)
  buf = IOBuffer()
  Serialization.serialize(buf, msg)
  cmsg = Blosc.compress(take!(buf))
  Serialization.serialize(socket, cmsg)
end

function receive(socket) :: Msg
  buf = IOBuffer()
  cmsg = Serialization.deserialize(socket)
  write(buf, Blosc.decompress(UInt8, cmsg))
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

struct Logoff <: ClientMsg
  msg :: String
end

struct DataRecord <: ClientMsg
  data  :: Vector
  reqid :: Int
  id    :: Int
  dtime :: Float64
end

struct ContestRecord <: ClientMsg
  data  :: Array{Int, 3} 
  reqid :: Int
  id    :: Int
  dtime :: Float64
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
  wait_time :: Float64
end

struct DataConfirmation <: ServerMsg
  id  :: Int
end

struct ContestConfirmation <: ServerMsg
  id :: Int
end


#
# Server Request messages
#

struct PlayerSpec

  # Composed reference model
  ref_model     :: Dict{Symbol}

  # Player parameters
  power         :: Int      # power = 0 is used for IntuitionPlayers
  temperature   :: Float32
  exploration   :: Float32
  dilution      :: Float32
  name          :: String

end

function PlayerSpec(player :: MCTSPlayer)

  ref_model = base_model(player) |> to_cpu |> Jtac.decompose
  
  PlayerSpec( ref_model, player.power, player.temperature
            , player.exploration, player.dilution, player.name )
end

function PlayerSpec(player :: IntuitionPlayer)

  ref_model = base_model(player) |> to_cpu |> Jtac.decompose
  PlayerSpec(ref_model, 0, player.temperature, 0., 0., player.name)

end


# Given an instruction, a (gpu/async) player can be generated
function get_player(spec :: PlayerSpec; gpu = false, async = false)

  model = Jtac.compose(spec.ref_model)

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


#=

# -------- Training Task ----------------------------------------------------- #

function collect_games!(buffer, rb_queue, rbmax)

  sets = 0

  while isready(rb_queue)

    games = take!(rb_queue)

    # Get the length of every dataset in buffer and the length of the fresh games
    sizes, s = length.(buffer), length(games)

    # If there is not enough space in the buffer left, remove some entries first
    if sum(sizes) + s > rbmax

      k = findfirst(x -> x > s, cumsum(sizes))
      foreach(j -> deleteat!(buffer, j), 1:k)

    end

    push!(buffer, games)

    sets += 1

  end

  sets

end

function train_thread( player
                     , player_queue
                     , rb_queue
                     , rbmax
                     , rbmin
                     , steps_renewal
                     , selfplay_training_ratio
                     ; kwargs... )

  # The number of steps played since the playing model was last renewed
  steps = 0

  # The total number of training steps
  total_steps = 0

  # The total number of game datasets collected from clients 
  total_sets = 0

  # Replay buffer that collects the datasets recorded by the clients. The
  # datasets in this variable are available for training. As more sets are
  # generated over time, the replay buffer is updated: old games (played with
  # potentially older models) are replaced by more current ones.
  replay_buffer = DataSet[]

  while true

    # Update the replay buffer if clients pushed some new games
    total_sets += collect_games!(replay_buffer, rb_queue, rbmax)

    # Train the model for some steps
    steps += train_player!(player, replay_buffer, rbmin; kwargs...)
    total_steps += steps

    # After a certain number of training steps, we employ a new, hopefully
    # improved, player for self-playing
    if steps > steps_renewal

      # Serialize the player and put it in the player_queue. There, it
      # is managed by the server: If desired, a backup copy is saved,
      # and it is propagated to all clients as current player.
      put!(player_queue, _serialize(player))
      steps = 0

    end

  end

end


# -------- Playing Task ------------------------------------------------------ #

send_player(sock, player) = Serialization.serialize(sock, player)

function listen_to_client(c, sock, rb_queue)
  
  try

    while !eof(sock)

      ds = Serialization.deserialize(sock)
      put!(rb_queue, ds)

      @info "<play> Received set from client $c."

    end


  catch err

    @info "<play> Error reading from client $c: $err."
    close(sock)

  end

  ip, port = getsockname(sock)
  @info "<play> Client $c from $ip:$port has disconnected."

end

function play_task(player, player_queue, rb_queue, ip, port)

  clients = []
  current_player = Ref{Any}(_serialize(player))

  server = Sockets.listen(ip, port)
  @info "<play> Listening to $ip:$port."

  @sync begin

    # Accepting incoming client connections
    @async begin

      c = 0

      while true

        sock = Sockets.accept(server)
        ip, port = getsockname(sock)

        c += 1

        @info "<play> Client $c from $ip:$port connected."
        push!(clients, (c, sock))

        @info "<play> Sending current player to client $c."
        send_player(sock, current_player[])

        @async listen_to_client(rb_queue, sock)
        @info "<play> Begin listening to client $c for datasets."

      end

    end

    # Look for new players and tell the clients about them
    @async while true

      player = take(player_queue)
      current_player[] = player

      @info "<play> Received new player from <train>."

      foreach(c -> send_player(c[2], player), clients)

      @info "<play> Sent new player to clients $(first.(clients))."
      
    end

  end

end

# -------- Server ------------------------------------------------------------ #

function jtac_server( player :: MCTSPlayer{G}
                    ; buffer = 500000  # replay buffer size
                    , ip = ip"::1"
                    , port :: Int = 5112
                    , checkpoint :: String = ""
                    ) where {G <: Game}

  @sync begin

    # Channels to communicate between training and playing
    rb_queue = Channel{DataSet{G}}(10)
    player_queue = Channel(1)

    # Start the training activity in the background
    @info "Launch training thread..."
    train_task = @async train_thread(player, player_queue, rb_queue, args...)

    # Start the server, listen to the tcp socket for clients, and manage them
    @info "Launch playing thread..."
    play_task = @async play_thread(args...)

  end

end

=#
