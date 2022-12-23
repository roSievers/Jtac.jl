
using Sockets
import Base.@kwdef

abstract type Message end
abstract type Action <: Message end
abstract type Response <: Message end

Pack.@typed Message

struct ResponseOrError
  res :: Union{Nothing, Response}
  msg :: String
end

Pack.@untyped ResponseOrError

# We always obtain a response from the socket, even if authorization fails. If
# we don't, then it has to be an error and should be treated as such.
function handle(msg, rtype, port = 7238, host = ip"127.0.0.1", )
  try
    sock = Sockets.connect(host, port)
    Pack.pack(sock, msg)
    r = Pack.unpack(sock, ResponseOrError)
    if isnothing(r.res)
      @warn "Server $host:$port responded with error message: $(r.msg)"
      nothing
    else
      r.res :: rtype
    end
  catch err
    @warn "Error during communication with $host:$port: $err"
    nothing
  end
end

"""
    login(host, port; client_name, password)

Login action. Register the client under `client_name` at the train server.
A `password` may be required.

The response of the server contains the fields

    client_id :: UInt

which can be used for authentication.
"""
login(args...; kwargs...) =
  handle(Login(; kwargs...), LoginRes, args...)

@kwdef struct Login <: Action
  client_name :: String
  password :: String = ""
end

@kwdef struct LoginRes <: Response
  client_id :: UInt
end

"""
    logout(host, port; client_id)

Logout action. Unregister `client_name` at the train server.

The response of the server contains the fields

    client_id :: UInt

which can be used for authentication.
"""
logout(args...; kwargs...) =
  handle(Logout(; kwargs...), LogoutRes, args...)

@kwdef struct Logout <: Action
  client_id :: UInt
end

@kwdef struct LogoutRes <: Response
  success :: Bool
end

"""
    query_status(host, port; client_id)

Status query action. Obtain status information about the current session. Only
privileged clients have access to this functionality.

The response of the server contains the fields (TODO!)

    session_id :: UInt
    clients :: Vector{String}  # which clients are logged in

    training :: Bool           # is the network currently being trained?
    generation :: Int          # current generation

"""
query_status(args...; kwargs...) =
  handle(QueryStatus(; kwargs...), QueryStatusRes, args...)

@kwdef struct QueryStatus <: Action
  client_id :: UInt
end

@kwdef struct QueryStatusRes <: Response
  session_id :: UInt
  clients :: Vector{String}
  state :: Symbol
  generation :: Int
end

"""
    query_history(host, port; client_id [, starting_at, max_entries])

History query action. Get list of server events. May be censored if `client_id`
is not privileged. If a history event hash is provided for `starting_at`, then
only entries newer than this one are requested. The value of `max_entries`
serves as an upper bound on the number of history events returned.

The response of the server contains the fields

    history :: Vector{Event}

The event api is documented in the module `Events`.
"""
query_history(args...; kwargs...) =
  handle(QueryHistory(; kwargs...), QueryHistoryRes, args...)

@kwdef struct QueryHistory <: Action
  client_id :: UInt
  starting_at :: Int = 1
  max_entries :: Int = -1
end

@kwdef struct QueryHistoryRes <: Response
  history :: Vector{Event}
end

"""
    query_generation(host, port; client_id, wait = false)

Generation query action. Obtain the current model generation.
Clients can loop over this query in order to register generation changes.

The response of the server contains the fields
    generation :: Int
"""
query_generation(args...; kwargs...) =
  handle(QueryGeneration(; kwargs...), QueryGenerationRes, args...)

@kwdef struct QueryGeneration <: Action
  client_id :: UInt
  wait :: Bool = false
end

@kwdef struct QueryGenerationRes <: Response
  generation :: Int
end

"""
    query_model(host, port; client_id [, generation])

Model query action. Obtain the specified `generation` of the model. If negative,
obtain the most current generation.

The response of the server contains the fields

    model :: Model.AbstractModel
    generation :: Int
"""
query_model(args...; kwargs...) =
  handle(QueryModel(; kwargs...), QueryModelRes, args...)

@kwdef struct QueryModel <: Action
  client_id :: UInt
  generation :: Int = -1
end

@kwdef struct QueryModelRes <: Response
  model :: AbstractModel
  generation :: Int
end

"""
    query_player(host, port; client_id)

Player query action. Obtain the player based on `generation` of the model that
is to be used for data generation. Also returns other information, like
branching probabilities, needed to produce canonical self-play data used for
training. Specification of the generation works as in `query_model`.

The response of the server contains the fields

    player :: Player.MCTSPlayer
    instance_randomization :: Float64
    branch_probability :: Float64
    branch_step_min :: Int
    branch_step_max :: Int

"""
query_player(args...; kwargs...) =
  handle(QueryPlayer(; kwargs...), QueryPlayerRes, args...)

@kwdef struct QueryPlayer <: Action
  client_id :: UInt
  generation :: Int = -1
end

@kwdef struct QueryPlayerRes <: Response
  player :: MCTSPlayer
  generation :: Int
  instance_randomization :: Float64 = 0.
  branch_probability :: Float64 = 0.
  branch_step_min :: Int = 1
  branch_step_max :: Int = 10
end

"""
    upload_data(host, port; client_id, generation, data)

Data upload action. Upload a dataset that has been generated by a model of the
given `generation`.

The response of the server contains the fields

    success :: Bool
"""
upload_data(args...; kwargs...) =
  handle(UploadData(; kwargs...), UploadDataRes, args...)

@kwdef struct UploadData <: Action
  client_id :: UInt64
  generation :: Int
  data :: DataSet
end

@kwdef struct UploadDataRes <: Response
  success :: Bool
end

"""
    upload_contest(host, port; client_id, data :: Ranking)

Contest upload action. Upload contest data (i.e., a `Ranking`) that can be
accessed as part of the event history.

The response of the server contains the fields

    success :: Bool
"""
upload_contest(args...; kwargs...) =
  handle(UploadContest(; kwargs...), UploadContestRes, args...)

@kwdef struct UploadContest <: Action
  client_id :: UInt
  ranking :: Ranking
end

@kwdef struct UploadContestRes <: Response
  success :: Bool
end

"""
    set_param(host, port; client_id, param, value)

Parameter setting action. Set parameters that influences the training
environment. Only privileged clients are allowed to do this.

The response of the server contains the fields

  success :: Bool
"""
set_param(args...; kwargs...) =
  handle(SetParam(; kwargs...), SetParamRes, args...)

@kwdef struct SetParam <: Action
  client_id :: UInt
  param :: Vector{String}
  value :: Vector{Any}
end

@kwdef struct SetParamRes <: Response
  success :: Bool
end

"""
    stop_training(host, port; client_id)

Stop training action. Can be used to pause the training. Only privileged clients
are allowed to do this.

The response of the server contains the fields

  success :: Bool
"""
stop_training(args...; kwargs...) =
  handle(StopTraining(; kwargs...), StopTrainingRes, args...)

@kwdef struct StopTraining <: Action
  client_id :: UInt
end

@kwdef struct StopTrainingRes <: Response
  success :: Bool
end

"""
    start_training(host, port; client_id)

Start training action. Can be used to resume training after `stop_training` has
been called. Only privileged clients are allowed to do this.

The response of the server contains the fields

  success :: Bool
"""
start_training(args...; kwargs...) =
  handle(StartTraining(; kwargs...), StartTrainingRes, args...)

@kwdef struct StartTraining <: Action
  client_id :: UInt
end

@kwdef struct StartTrainingRes <: Response
  success :: Bool
end

"""
    stop_recording(host, port; client_id)

Stop recording action. Can be used to pause ongoing recording. Only privileged
clients are allowed to do this.

The response of the server contains the fields

  success :: Bool
"""
stop_recording(args...; kwargs...) =
  handle(StopRecording(; kwargs...), StopRecordingRes, args...)

@kwdef struct StopRecording <: Action
  client_id :: UInt
end

@kwdef struct StopRecordingRes <: Response
  success :: Bool
end

"""
    start_recording(host, port; client_id)

Start recording action. Can be used to resume recording after `stop_recording`
has been called. Only privileged clients are allowed to do this.

The response of the server contains the fields

  success :: Bool
"""
start_recording(args...; kwargs...) =
  handle(StartRecording(; kwargs...), StartRecordingRes, args...)

@kwdef struct StartRecording <: Action
  client_id :: UInt
end

@kwdef struct StartRecordingRes <: Response
  success :: Bool
end

