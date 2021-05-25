

module Events

import ..Context

using Jtac
import Dates
import LazyJSON

export Event

export Body,
       EpochBody,
       EraBody,
       DataBody,
       DataReqBody,
       ContestBody,
       ContestReqBody,
       ClientBody,
       ContextBody,
       ModelBody


"""
An event body determining the event type.
"""
abstract type Body end

"""
An event that takes place during the training process. Used for logging
/ debugging purposes and to inform jtac watch instances. Contains a time stamp,
an id, and a body with further information about the event.
"""
struct Event
  id   :: Int
  time :: Dates.DateTime
  body :: Body
end

Event(body :: Body) = Event(-1, Dates.now(), body)
Event(id :: Int, body :: Body) = Event(id, Dates.now(), body)
Event(id :: Int, event :: Event) = Event(id, event.time, event.body)

"""
An epoch event body. Indicates the completion of a single epoch of training.
"""
struct EpochBody <: Body
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
An era event body. Indicates the beginning of an era. Can contain metrics about
the model that the jtac-train instance decides to expose.
"""
struct EraBody <: Body
  number :: Int
  epoch :: Int
  metrics :: Vector{Tuple{String, Float64}}
end

"""
A data event body. Indicates that training data was received from an jtac-play
instance.
"""
struct DataBody <: Body
  reqid :: Int
  client :: String
  states :: Int
  playings :: Int
  mb :: Float64
  time :: Float64
end

"""
A data request event body. Indicates that a data request was issued.
"""
struct DataReqBody <: Body
  reqid :: Int
  ctxid :: Int
end

"""
A contest event body. Indicates that contest data was received from an jtac-play
instance.
"""
struct ContestBody <: Body
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

"""
A contest request event body. Sent if a contest request was issued.
"""
struct ContestReqBody <: Body
  reqid :: Int
end

"""
A client connection, meaning that a jtac serve instance has connected.
"""
struct ClientBody <: Body
  name :: String
  ip :: String
  port :: Int
  disconnect :: Bool
end

"""
A context body that holds information abou the setting in which the training
of a player takes place.
"""
mutable struct ContextBody <: Body
  ctx :: Context
end

ContextBody(args...; kwargs...) = ContextBody(Context(args...; kwargs...)) 

struct ModelBody <: Body
  name :: String
  info :: String
  trunk :: Vector{Any}
  phead :: Vector{Any}
  vhead :: Vector{Any}
  fhead :: Union{Vector{Any}, Nothing}
  params :: Int
  base :: Union{String, Nothing}
end

function ModelBody(model :: Model.AbstractModel, name, info, base)
  params = Model.count_params(model)
  trunk = arch_vec(model.trunk)
  phead = arch_vec(model.phead)
  vhead = arch_vec(model.vhead)
  fhead = arch_vec(model.fhead)
  ModelBody(name, info, trunk, phead, vhead, fhead, params, base)
end


# Todo - this has to be improved for better understanding of the architecture
function arch_vec(layer :: Model.PrimitiveLayer)
  ["Primitive", typeof(layer) |> string]
end

function arch_vec(layer :: Model.CompositeLayer)
  ["Composite", typeof(layer) |> string, arch_vec.(Model.layers(layer))]
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
  typ = typ[1:end-4]

  time = ( year = Dates.year(ev.time)
         , month = Dates.month(ev.time)
         , day = Dates.day(ev.time)
         , hour = Dates.hour(ev.time)
         , minute = Dates.minute(ev.time)
         , second = Dates.second(ev.time) )

  if typ == "Context"
    ( id = ev.id, time = time, body = [typ, ev.body.ctx] ) 
  else
    ( id = ev.id, time = time, body = [typ, ev.body] ) 
  end |> LazyJSON.jsonstring
end

function parse(str, :: Type{Event})
  val = LazyJSON.value(str)
  id = val["id"]
  t = val["time"]
  time = Dates.DateTime(t["year"], t["month"], t["day"], t["hour"], t["minute"], t["second"])
  bodytype = Symbol(val["body"][1] * "Body")
  if bodytype == :ContextBody
    Event(id, time, ContextBody(convert(Context, val["body"][2])))
  else
    Event(id, time, convert(eval(bodytype), val["body"][2]))
  end
end

end # module Events
