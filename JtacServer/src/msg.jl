abstract type Role end

abstract type Train   <: Role end
abstract type Serve   <: Role end
abstract type Monitor <: Role end

"""
Message between source S and destination D. The two modes of communication are
between jtac train and serve instances, as well as between jtac train and
monitor instances.

For efficiency, the former mode of communication (Train <--> Serve) mostly uses
a binary format provided by julia's serialization capability. To make monitoring
jtac training transparent and extensible, the latter mode of communication
(Train <--> Monitor) uses a simple JSON format. 
"""
abstract type Message{S <: Role, D <: Role} end

"""
    send(socket, message)

Sending a Jtac service message to socket.
"""
send(socket, msg :: Message) = Serialization.serialize(socket, msg)

"""
    receive(socket, type)

Receiving a Jtac service message from `socket` and asserting that it is of type
`type`. Only apply this to trusted sockets, since arbitrary code may be
executed.
"""
function receive(socket, :: Type{M}) where {M <: Message}
  val = Serialization.deserialize(socket)
  val isa M ? val : nothing
end


"""
Login messages are always exchanged via JSON for both monitor and serve
instances.
"""
abstract type Login{S <: Role} end

send(socket, msg :: Login) = println(socket, LazyJSON.jsonstring(msg))

function receive(socket, :: Type{Login})
  try
    value = LazyJSON.value(readline(socket))
    if value["kind"] == "serve"
      convert(ServeLogin, value)
    elseif value["kind"] == "monitor"
      convert(MonitorLogin, value)
    end
  catch err
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
# Messages play -> train
#

"""
Login request of a play service.

This must be the initial message to a train service. The train service then
decides whether to accept the connection or not. If it accepts, a
"""
struct ServeLogin <: Login{Serve}
  name    :: String
  token   :: String
  version :: String
  kind    :: String

  accept_data    :: Bool
  accept_contest :: Bool
end

function ServeLogin(name, token, data = true, contest = true)
  ServeLogin(name, token, VERSION, "serve", data, contest)
end

"""
Logout message of a play service
"""
struct ServeLogout <: Message{Serve, Train}
  text :: String
end

"""
Record of training data generated by the jtac serve instance.

It contains the compressed data sets and the id of the corresponding data
request.
"""
mutable struct ServeData <: Message{Serve, Train}
  _data    :: Vector{UInt8} # compressed Jtac datasets
  states   :: Int
  playings :: Int
  reqid    :: Int
  id       :: Int
  time     :: Float64
end

function ServeData(data :: Vector{<:Jtac.DataSet}, reqid, id, time)
  _data = compress(data)
  states = sum(length.(data))
  playings = length(data)
  ServeData(_data, states, playings, reqid, id, time)
end

"""
Record of contest data generated by the jtac serve instance.

It contains the contest results and the id of the corresponding contest request.
"""
mutable struct ServeContest <: Message{Serve, Train}
  data  :: Array{Int, 3}
  elo   :: Vector{Float64}
  draw  :: Float64
  sadv  :: Float64
  reqid :: Int
  id    :: Int
  time  :: Float64
end


#
# Messages train -> play
#

"""
Asking a jtac serve instance to disconnect. After some timeout, the connection
should be closed if the serve instance does not react.
"""
struct TrainDisconnectServe <: Message{Train, Serve}
  text :: String
end

"""
Asking a jtac serve instance to reconnect after some timeout. This can be used
if the jtac train instance needs to restart / pause, but activity should be
resumed automatically afterwards.
"""
struct TrainReconnectServe <: Message{Train, Serve}
  text :: String
  timeout :: Float64 # seconds before the reconnection should be attempted
end

"""
Asking a jtac serve instance to discard the current data request. Contest
requests are not affected.
"""
struct TrainIdleServe <: Message{Train, Serve}
  text :: String
end

"""
Requesting the generation of training data from a jtac serve instance.

The message encapsulates the player to be used for self play (as `PlayerSpec`)
and other information concerning the requested data (branching, number of
playings, ...).
"""
struct TrainDataServe <: Message{Train, Serve}

  spec :: PlayerSpec

  # Training options
  init_steps   :: Tuple{Int, Int}
  branch       :: Float64
  branch_steps :: Tuple{Int, Int}
  augment      :: Bool

  # Dataset options
  min_playings :: Int
  max_playings :: Int

  # Id of the instruction set
  reqid         :: Int

  function TrainDataServe( reqid :: Int
                         , player :: Jtac.MCTSPlayer{G}
                         ; augment = true
                         , init_steps = 0
                         , branch = 0.
                         , branch_steps = 1
                         , min_playings = 1
                         , max_playings = 10000
                         ) where {G <: Jtac.Game}

    @assert !isabstracttype(G)

    new( PlayerSpec(player)
       , Jtac.tup(init_steps)
       , branch
       , Jtac.tup(branch_steps)
       , augment
       , min_playings
       , max_playings
       , reqid )

  end

end

"""
    TrainDataServe(id, context, model)

Derive a data request based on a model and training context.
"""
function TrainDataServe(id, ctx, model :: Jtac.Model{<: Jtac.Game, false})

  player = Jtac.MCTSPlayer( model |> Jtac.to_cpu
                          , power = ctx.power
                          , temperature = ctx.temperature
                          , exploration = ctx.exploration
                          , dilution = ctx.dilution
                          , name = ctx.name * "-$id" )

  TrainDataServe( id, player
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
struct TrainDataConfirmServe <: Message{Train, Serve}
  id :: Int
end


"""
Requesting the generation of contest data from a jtac serve instance.
"""
struct TrainContestServe <: Message{Train, Serve}
  specs  :: Vector{PlayerSpec}
  active :: Vector{Int} 
  names  :: Vector{String}
  length :: Int
  era    :: Int
  reqid  :: Int
end

function TrainContestServe( reqid :: Int
                          , players :: Vector{<: Jtac.MCTSPlayer}
                          , length :: Int
                          , active = 1:length(players) )
  # Make sure that a concrete, consistent game type can be derived
  Jtac.derive_gametype(players)
  specs = PlayerSpec.(players)
  names = Jtac.name.(players)
  ContestRequest(specs, active, names, length, reqid)
end


"""
Confirmation from the train service that the upload of a contest record was
successful.
"""
struct TrainContestConfirmServe <: Message{Train, Serve}
  id :: Int
end


#
# Messages train <-> watch
#

struct MonitorLogin <: Login{Monitor}
  name    :: String
  token   :: String
  version :: String
  kind    :: String

end

function MonitorLogin(name, token)
  MonitorLogin(name, token, VERSION, "monitor")
end

# The following types are informally specified by the implementation of jtab,
# a web-based monitoring tool for jtac training procedures.

abstract type Body end

"""
An event to be sent to every connected jtac monitor instance.
"""
struct Event <: Message{Train, Monitor}
  id   :: Int
  time :: Dates.DateTime
  body :: Body
end

Event(body) = Event(-1, Dates.now(), body)
Event(id, event :: Event) = Event(id, event.time, event.body)


"""
An epoch event body. Sent after the completion of single epochs.
"""
struct Epoch <: Body
  number    :: Int
  trainloss :: Vector{Tuple{String, Float64}}
  testloss  :: Vector{Tuple{String, Float64}}
  quality   :: Float64
  capacity  :: Float64
  size      :: Int
  ctxid     :: Int
  era       :: Int
end

"""
An era event body. Sent at the beginning of each era. Can contain metrics about
the model that the jtac train instance decides to expose.
"""
struct Era <: Body
  number :: Int
  epoch :: Int
  metrics :: Vector{Tuple{String, Float64}}
end

"""
A data event body. Sent if training data was received from an jtac serve
instance.
"""
struct Data <: Body
  reqid :: Int
  client :: String
  states :: Int
  playings :: Int
  mb :: Float64
  time :: Float64
end

"""
    Data(client, data)

Generate a data message from the `data` answer received from `client`.
"""
function Data(client, d :: ServeData)
  mb = sizeof(d._data) / 1024 / 1024
  Data(d.reqid, client, d.states, d.playings, mb, d.time)
end

"""
A data request event body. Sent if a data request was issued.
"""
struct Datareq <: Body
  reqid :: Int
  ctxid :: Int
end

"""
A contest event body. Sent if contest data was received from an jtac serve
instance.
"""
struct Contest <: Body
  reqid :: Int
  client :: String
  era :: Int
  names :: Vector{String}
  matches :: Int
  balance :: Vector{Vector{Int}}
  elo :: Vector{Float64}
  draw :: Float64
  sadv :: Float64
end

function Contest(client, r :: TrainContestServe, d :: ServeContest)
  balance = [d.data[i,:,1] .- d.data[i,:,3] for i in 1:size(d.data, 1)]
  Contest(d.reqid, client, r.era, r.names,
          sum(abs, d.data), balance, d.elo, d.draw, d.sadv)
end

"""
A contest request event body. Sent if a contest request was issued.
"""
struct Contestreq <: Body
  reqid :: Int
end

"""
A client connection, meaning that a jtac serve instance has connected.
"""
struct Client <: Body
  name :: String
  ip :: String
  port :: Int
  disconnect :: Bool
end

"""
A context body that describes the setting in which the training of a player
takes place. Contains information like the learning rate, batchsize, or power.
"""
mutable struct Context <: Body

  id :: Int

  # Player options for dataset generation
  name :: String
  power :: Int
  temperature :: Float32
  exploration :: Float32
  dilution :: Float32

  # Other options for dataset generation
  init_steps :: Tuple{Int, Int}
  branch :: Float64 
  branch_steps :: Tuple{Int, Int}
  augment :: Bool

  # Minimal or maximal playings required from data packages sent by jtac serve
  # instances
  min_playings :: Int
  max_playings :: Int

  # Selecting training data from the DataPool
  # Epoch: subset of the pool used for one training iteration
  epoch_size :: Int
  iterations :: Int
  test_frac :: Float64 # fraction of data to go to test set 
  max_age :: Int
  max_use :: Int

  # Data is only used when the quality of the datapool is above this threshold
  min_quality :: Float64 

  # Number between 0 and 1 to weight age vs. usage to assess the quality of
  # a dataset (convex combination)
  age_weight :: Float64
                        
  # Maximal number of game states stored in the test and trainpools
  capacity :: Int

  # Era: games used for training until the reference model gets its next update
  era_size :: Int

  # Options for training after an epoch has been selected from the DataPool
  batch_size :: Int
  learning_rate :: Float32
  momentum :: Float32
  loss_weights :: Vector{Float32}

  # number of backups
  backups :: Int

  # Additional meta-information for purposes of documentation
  msg :: String

end

function Context( id :: Int
                ; name = "default"
                , power = 50
                , temperature = 1.
                , exploration = 1.41
                , dilution = 0.
                , initial_steps = (0, 0)
                , branch = 0.
                , branch_steps = (0, 0)
                , augment = true
                , min_playings = 1
                , max_playings = 1000
                , epoch_size = 5000
                , iterations = 1
                , test_frac = 0.1
                , max_age = 3
                , max_use = 3
                , min_quality = 0.
                , age_weight = 0.5
                , capacity = 10^6
                , era_size = 20000
                , batch_size = 512
                , learning_rate = 1e-2
                , momentum = 0.9
                , loss_weights = [1., 1., 0.]
                , backups = 2
                , msg = "" )

  Context( id, name, power, temperature, exploration, dilution,
           initial_steps, branch, branch_steps, augment,
           min_playings, max_playings,
           epoch_size, iterations, test_frac, max_age, max_use, min_quality, age_weight,
           capacity, era_size, batch_size, learning_rate, momentum, loss_weights,
           backups, msg )
end


struct Model <: Body
  name :: String
  info :: String
  trunk :: Vector{Any}
  phead :: Vector{Any}
  vhead :: Vector{Any}
  fhead :: Union{Vector{Any}, Nothing}
  params :: Int
  base :: Union{String, Nothing}
end

function Model(model :: Jtac.Model, name, info, base)
  params = Jtac.count_params(model)
  trunk = arch_vec(model.trunk)
  phead = arch_vec(model.phead)
  vhead = arch_vec(model.vhead)
  fhead = arch_vec(model.fhead)
  Model(name, info, trunk, phead, vhead, fhead, params, base)
end


# Todo - this has to be improved for better understanding of the architecture
function arch_vec(layer :: Jtac.PrimitiveLayer)
  ["Primitive", typeof(layer) |> string]
end

function arch_vec(layer :: Jtac.CompositeLayer)
  ["Composite", typeof(layer) |> string, arch_vec.(Jtac.layers(layer))]
end

arch_vec(:: Nothing) = nothing

# Help LazyJSON with converting arrays to 2-tuples
function Base.convert( :: Type{Tuple{I, J}}
                     , arr :: LazyJSON.Array{Nothing, String}
                     ) where {I, J}
  @assert length(arr) == 2
  (I(arr[1]), J(arr[2]))
end

function json(ev :: Event)
  typ = typeof(ev.body) |> nameof |> string

  time = ( year = Dates.year(ev.time)
         , month = Dates.month(ev.time)
         , day = Dates.day(ev.time)
         , hour = Dates.hour(ev.time)
         , minute = Dates.minute(ev.time)
         , second = Dates.second(ev.time) )

  ( id = ev.id, time = time, body = [typ, ev.body] ) |> LazyJSON.jsonstring
end

function parse_json(str, :: Type{Event})
  val = LazyJSON.value(str)
  id = val["id"]
  t = val["time"]
  time = Dates.DateTime(t["year"], t["month"], t["day"], t["hour"], t["minute"], t["second"])
  bodytype = Symbol(val["body"][1])
  Event(id, time, convert(eval(bodytype), val["body"][2]))
end

send(socket, ev :: Event) = println(socket, json(ev)) 

function receive(socket, :: Type{Event})
  parse_json(readline(socket), Event)
end

