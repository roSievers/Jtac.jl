
import Base.@kwdef

"""
Event type. The jtac training server logs a series of events that can be
queried from external clients for visualization of the training progress. Each
subtype of `Event` possesses the fields

    timestamp
    client_id

that specify the event type (snake-case), the time of creation of the
event, and the id of the client that triggered the event. There are three
types of clients:

* The privileged client `"jtac"` with `client_id = 0`, which corresponds
  to the server itself. I.e., a `StartSession` event is triggerd by `"jtac"`.
* The privileged client `"admin"` with a randomly generated `client_id`.
  This client is used to manipulate the training procedure from an external
  source. It can use the full API.
* External non-privileged clients with custom names and randomly generated
  `client_id`.
"""
abstract type Event end

Pack.@typed Event

"""
Start of the training session
"""
@kwdef struct StartSession <: Event
  timestamp :: Float64 = time()
  client_id :: UInt = 0

  session_id :: UInt
  session_name :: String
end

"""
End of the training session
"""
@kwdef struct StopSession <: Event
  timestamp :: Float64 = time()
  client_id :: UInt = 0

  session_id :: UInt
  session_name :: String
end

"""
A model has been loaded
"""
@kwdef struct LoadModel <: Event
  timestamp :: Float64 = time()
  client_id :: UInt = 0

  source :: String
  arch :: String # TODO!!
  role :: String # "training", "playing", or "both"
end

"""
A client has logged in
"""
@kwdef struct LoginClient <: Event
  timestamp :: Float64 = time()
  client_id :: UInt

  client_name :: String
end

"""
A client has logged off
"""
@kwdef struct LogoutClient <: Event
  timestamp :: Float64 = time()
  client_id :: UInt

  client_name :: String
end

"""
A model generation started
"""
@kwdef struct StartGeneration <: Event
  timestamp :: Float64 = time()
  client_id :: UInt = 0

  generation :: Int
  config :: Dict{String, Dict{String, Any}}
end

"""
A model generation ended
"""
@kwdef struct StopGeneration <: Event
  timestamp :: Float64 = time()
  client_id :: UInt = 0

  generation :: Int
end

"""
The server status has been queried
"""
@kwdef struct QueryStatus <: Event
  timestamp :: Float64 = time()
  client_id :: UInt
end

"""
The server history has been queried
"""
@kwdef struct QueryHistory <: Event
  timestamp :: Float64 = time()
  client_id :: UInt

  starting_at :: Int
  max_entries :: Int
end

"""
The current model generation has been queried
"""
@kwdef struct QueryGeneration <: Event
  timestamp :: Float64 = time()
  client_id :: UInt

  wait :: Bool
end

"""
A model has been queried
"""
@kwdef struct QueryModel <: Event
  timestamp :: Float64 = time()
  client_id :: UInt

  generation :: Int
end

"""
The current player (and other selfplay information) has been queried
"""
@kwdef struct QueryPlayer <: Event
  timestamp :: Float64 = time()
  client_id :: UInt
end

"""
Training data was uploaded
"""
@kwdef struct UploadData <: Event
  timestamp :: Float64 = time()
  client_id :: UInt

  generation :: Int
  length :: Int
end

"""
A ranking was uploaded
"""
@kwdef struct UploadContest <: Event
  timestamp :: Float64 = time()
  client_id :: UInt

  ranking :: Ranking
end

"""
Config parameters were modified
"""
@kwdef struct SetParam <: Event
  timestamp :: Float64 = time()
  client_id :: UInt

  param :: Vector{String}
  value :: Vector{Any}
end

"""
Training mode was started
"""
@kwdef struct StartTraining <: Event
  timestamp :: Float64 = time()
  client_id :: UInt = 0
end

"""
Training mode was stopped
"""
@kwdef struct StopTraining <: Event
  timestamp :: Float64 = time()
  client_id :: UInt = 0
end

"""
A training step (which may contain several batches) was conducted
"""
@kwdef struct StepTraining <: Event
  timestamp :: Float64 = time()
  client_id :: UInt = 0

  batchsize :: Int
  stepsize :: Int
  targets :: Vector{String}
  weights :: Vector{Float64}
  values :: Vector{Float64}
end

"""
Recording mode was initiated
"""
@kwdef struct StartRecording <: Event
  timestamp :: Float64 = time()
  client_id :: UInt = 0
end

"""
Recording mode was ended
"""
@kwdef struct StopRecording <: Event
  timestamp :: Float64 = time()
  client_id :: UInt = 0
end

"""
A dataset was recorded by the server
"""
@kwdef struct StepRecording <: Event
  timestamp :: Float64 = time()
  client_id :: UInt = 0

  length :: Int
end

