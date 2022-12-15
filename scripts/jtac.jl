

using Sockets

using Jtac
using .Game
using .Model
using .Player
using .Training

module Events

  import Base.@kwdef

  using Jtac

  """
  Event type. The jtac training server logs a series of events that can be
  queried from external clients for visualization of the training progress. Each
  subtype of `Event` possesses the fields

      type
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

  msgpack_type(:: Type{Event}) = Pack.MapType()

  function from_msgpack(:: Type{Event}, d :: Dict)
    sym = to_camlcase(d["type"])
    T = eval(Symbol(sym))
    args = map(fieldnames(T)) do name
      d[string(name)]
    end
    T(args...)
  end

  to_camlcase(str :: String) =
    join(uppercasefirst(word) for word in split(str, "_"))

  """
  Start of the training session
  """
  @kwdef struct StartSession <: Event
    type :: Symbol = :start_session
    timestamp :: Float64 = time()
    client_id :: UInt = 0

    session_id :: UInt
    session_name :: String
  end

  """
  End of the training session
  """
  @kwdef struct StopSession <: Event
    type :: Symbol = :stop_session
    timestamp :: Float64 = time()
    client_id :: UInt = 0

    session_id :: UInt
    session_name :: String
  end

  """
  A model has been loaded
  """
  @kwdef struct LoadModel <: Event
    type :: Symbol = :load_model
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
    type :: Symbol = :login_client
    timestamp :: Float64 = time()
    client_id :: UInt

    client_name :: String
  end

  """
  A client has logged off
  """
  @kwdef struct LogoutClient <: Event
    type :: Symbol = :logout_client
    timestamp :: Float64 = time()
    client_id :: UInt

    client_name :: String
  end

  """
  A model generation started
  """
  @kwdef struct StartGeneration <: Event
    type :: Symbol = :start_generation
    timestamp :: Float64 = time()
    client_id :: UInt = 0

    generation :: Int
    config :: Dict{String, Dict{String, Any}}
  end

  """
  A model generation ended
  """
  @kwdef struct StopGeneration <: Event
    type :: Symbol = :stop_generation
    timestamp :: Float64 = time()
    client_id :: UInt = 0

    generation :: Int
  end

  """
  The server status has been queried
  """
  @kwdef struct QueryStatus <: Event
    type :: Symbol = :query_status
    timestamp :: Float64 = time()
    client_id :: UInt
  end

  """
  The server history has been queried
  """
  @kwdef struct QueryHistory <: Event
    type :: Symbol = :query_history
    timestamp :: Float64 = time()
    client_id :: UInt

    starting_at :: Int
    max_entries :: Int
  end

  """
  The current model generation has been queried
  """
  @kwdef struct QueryGeneration <: Event
    type :: Symbol = :query_generation
    timestamp :: Float64 = time()
    client_id :: UInt

    wait :: Bool
  end

  """
  A model has been queried
  """
  @kwdef struct QueryModel <: Event
    type :: Symbol = :query_model
    timestamp :: Float64 = time()
    client_id :: UInt

    generation :: Int
  end

  """
  The current player (and other selfplay information) has been queried
  """
  @kwdef struct QueryPlayer <: Event
    type :: Symbol = :query_player
    timestamp :: Float64 = time()
    client_id :: UInt
  end

  """
  Training data was uploaded
  """
  @kwdef struct UploadData <: Event
    type :: Symbol = :upload_data
    timestamp :: Float64 = time()
    client_id :: UInt

    generation :: Int
    length :: Int
  end

  """
  A ranking was uploaded
  """
  @kwdef struct UploadContest <: Event
    type :: Symbol = :upload_contest
    timestamp :: Float64 = time()
    client_id :: UInt

    ranking :: Player.Ranking
  end

  """
  Config parameters were modified
  """
  @kwdef struct SetParam <: Event
    type :: Symbol = :set_param
    timestamp :: Float64 = time()
    client_id :: UInt

    param :: Vector{String}
    value :: Vector{Any}
  end

  """
  Training mode was started
  """
  @kwdef struct StartTraining <: Event
    type :: Symbol = :start_training
    timestamp :: Float64 = time()
    client_id :: UInt = 0
  end

  """
  Training mode was stopped
  """
  @kwdef struct StopTraining <: Event
    type :: Symbol = :stop_training
    timestamp :: Float64 = time()
    client_id :: UInt = 0
  end

  """
  A training step (which may contain several batches) was conducted
  """
  @kwdef struct StepTraining <: Event
    type :: Symbol = :step_training
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
    type :: Symbol = :start_recording
    timestamp :: Float64 = time()
    client_id :: UInt = 0
  end

  """
  Recording mode was ended
  """
  @kwdef struct StopRecording <: Event
    type :: Symbol = :stop_recording
    timestamp :: Float64 = time()
    client_id :: UInt = 0
  end

  """
  A dataset was recorded by the server
  """
  @kwdef struct StepRecording <: Event
    type :: Symbol = :step_recording
    timestamp :: Float64 = time()
    client_id :: UInt = 0

    length :: Int
  end

end # module Event

module Api

  using Sockets
  import Base.@kwdef

  import Jtac: Pack, Data, Model, Player
  import ..Events: Event

  # We always obtain a response from the socket, even if authorization fails. If
  # we don't, then it has to be an error and should be treated as such.
  function handle(msg, rtype, host = ip"127.0.0.1", port = 7238 )
    try
      sock = Sockets.connect(host, port)
      Pack.pack(sock, msg)
      Pack.unpack(sock, rtype)
    catch err
      @warn "Error during communication with $host:$port: $err"
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
    handle(LoginMsg(; kwargs...), LoginRes, args...)

  @kwdef struct LoginMsg
    action :: Symbol = :login
    client_name :: String
    password :: String = ""
  end

  @kwdef struct LoginRes
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
    handle(LogoutMsg(; kwargs...), LogoutRes, args...)

  @kwdef struct LogoutMsg
    action :: Symbol = :logout
    client_id :: UInt
  end

  @kwdef struct LogoutRes
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
    handle(QueryStatusMsg(; kwargs...), QueryStatusRes, args...)

  @kwdef struct QueryStatusMsg
    action :: Symbol = :query_status
    client_id :: UInt
  end

  @kwdef struct QueryStatusRes
    session_id :: UInt
    clients :: Vector{String}
    training :: Bool
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
    handle(QueryHistoryMsg(; kwargs...), QueryHistoryRes, args...)

  @kwdef struct QueryHistoryMsg
    action :: Symbol = :query_history
    client_id :: UInt
    starting_at :: Int
    max_entries :: Int
  end

  @kwdef struct QueryHistoryRes
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
    handle(QueryGenerationMsg(; kwargs...), QueryGenerationRes, args...)

  @kwdef struct QueryGenerationMsg
    action :: Symbol = :query_generation
    client_id :: UInt
    wait :: Bool
  end

  @kwdef struct QueryGenerationRes
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
    handle(QueryModelMsg(; kwargs...), QueryModelRes, args...)

  @kwdef struct QueryModelMsg
    action :: Symbol = :query_model
    client_id :: UInt
    generation :: Int = -1
  end

  @kwdef struct QueryModelRes
    model :: Model.NeuralModel
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
    handle(QueryPlayerMsg(; kwargs...), QueryPlayerRes, args...)

  @kwdef struct QueryPlayerMsg
    action :: Symbol = :query_player
    client_id :: UInt
    generation :: Int = 0
  end

  @kwdef struct QueryPlayerSig
    player :: Player.MCTSPlayer
    generation :: Int
    instance_randomization :: Float64
    branch_probability :: Float64
    branch_step_min :: Int
    branch_step_max :: Int
  end

  """
      upload_data(host, port; client_id, generation, data)

  Data upload action. Upload a dataset that has been generated by a model of the
  given `generation`.

  The response of the server contains the fields

      success :: Bool
  """
  upload_data(args...; kwargs...) =
    handle(UploadDataMsg(; kwargs...), UploadDataRes, args...)

  @kwdef struct UploadDataMsg
    action :: Symbol = :upload_data
    client_id :: UInt64
    generation :: Int
    data :: Data.DataSet
  end

  @kwdef struct UploadDataRes
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
    handle(UploadContestMsg(; kwargs...), UploadContestRes, args...)

  @kwdef struct UploadContestMsg
    action :: Symbol = :upload_contest
    client_id :: UInt
    ranking :: Player.Ranking
  end

  @kwdef struct UploadContestRes
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
    handle(SetParamMsg(; kwargs...), SetParamRes, args...)

  @kwdef struct SetParamMsg
    action :: Symbol = :set_param
    client_id :: UInt
    param :: Vector{String}
    value :: Vector{Any}
  end

  @kwdef struct SetParamRes
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
    handle(StopTrainingMsg(; kwargs...), StopTrainingRes, args...)

  @kwdef struct StopTrainingMsg
    action :: Symbol = :stop_training
    client_id :: UInt
  end

  @kwdef struct StopTrainingRes
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
    handle(StartTrainingMsg(; kwargs...), StartTrainingRes, args...)

  @kwdef struct StartTrainingMsg
    action :: Symbol = :start_training
    client_id :: UInt
  end

  @kwdef struct StartTrainingRes
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
    handle(StopRecordingMsg(; kwargs...), StopRecordingRes, args...)

  @kwdef struct StopRecordingMsg
    action :: Symbol = :stop_recording
    client_id :: UInt
  end

  @kwdef struct StopRecordingRes
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
    handle(StartRecordingMsg(; kwargs...), StartRecordingRes, args...)

  @kwdef struct StartRecordingMsg
    action :: Symbol = :start_recording
    client_id :: UInt
  end

  @kwdef struct StartRecordingRes
    success :: Bool
  end

end # module Api

module Config

  import TOML

  # default config sections

  function server(
    ; host :: String = "127.0.0.1"
    , port :: Int = 7238 
    , gpu :: Int = 0
    , snapshot_interval :: Int = 5
    , snapshot_path :: String = "" )

    Dict( "host" => host
        , "port" => port
        , "gpu" => gpu
        , "snapshot_interval" => snapshot_interval
        , "snapshot_path" => snapshot_path )
  end

  function training(
    ; model :: String = ""
    , batchsize :: Int = 512
    , stepsize :: Int = batchsize * 10
    , gensize :: Int = batchsize * 100
    , optimizer :: Symbol = :momentum
    , lr :: Float64 = 1e-2 )

    Dict( "model" => model
        , "batchsize" => batchsize
        , "stepsize" => stepsize
        , "gensize" => gensize
        , "optimizer" => optimizer
        , "lr" => lr )
  end

  function selfplay(
    ; model :: String = ""
    , power :: Int = 250
    , temperature :: Float64 = 1.
    , exploration :: Float64 = 1.41
    , dilution :: Float64 = 0.0
    , instance_randomization = 0.0
    , branch_probability = 0.0
    , branch_step_min = 1
    , branch_step_max = 10 )

    Dict( "model" => model
        , "power" => power
        , "temperature" => temperature
        , "exploration" => exploration
        , "dilution" => dilution
        , "instance_randomization" => instance_randomization
        , "branch_probability" => branch_probability
        , "branch_step_min" => branch_step_min
        , "branch_step_max" => branch_step_max )
  end

  function pool(
    ; capacity :: Int = 1_000_000
    , augment :: Bool = false
    , keep_generations :: Int = 3
    , keep_iterations :: Int = 1
    , quality_sampling :: Bool = true
    , quality_sampling_stop :: Float64 = quality_sampling ? 0.5 : -1.
    , quality_sampling_resume :: Float64 = quality_sampling ? 0.75 : -1. )

    Dict( "capacity" => capacity
        , "augment" => augment
        , "keep_iterations" => keep_iterations
        , "keep_generations" => keep_generations
        , "quality_sampling" => quality_sampling
        , "quality_sampling_stop" => quality_sampling_stop
        , "quality_sampling_resume" => quality_sampling_resume )
  end

  # Auxiliary functions

  function default()
    Dict( "server" => server()
        , "training" => training()
        , "selfplay" => selfplay()
        , "pool" => pool() )
  end

  struct ConfigKeyError <: Exception
    keys :: Vector{String}
  end

  struct ConfigValueError <: Exception
    keys :: Vector{String}
    values :: Vector{Any}
  end

  function check(cfg)
    # TODO: consistency checks for config values can go here
    # In case inconsistencies are found, throw ConfigValueErrors
  end

  # TODO: better error handling!
  function load(file)
    def = default()
    cfg = TOML.parsefile(file)
    diff = setdiff(keys(cfg), keys(def))
    if !isempty(diff)
      collect(diff) |> ConfigKeyError |> throw
    end
    for key in keys(cfg)
      diff = setdiff(keys(cfg[key]), keys(def[key]))
      if !isempty(diff)
        collect(diff) |> ConfigKeyError |> throw
      end
      def[key] = merge(def[key], cfg[key])
    end
    check(def)
    def
  end

  function save(file, cfg)
    open(file, "w") do io
      TOML.print(io, cfg, sorted = true, by = length)
    end
  end

  convert_param(:: Type{Int}, v :: String) = parse(Int, v)
  convert_param(:: Type{Float64}, v :: String) = parse(Float64, v)
  convert_param(:: Type{T}, v) where {T} = convert(T, v)

  function set_param!(cfg, param :: String, value)
    path = split(param, ".")
    try
      p = path
      T = typeof(cfg[p[1]][p[2]])
      cfg[p[1]][p[2]] = convert_param(T, value)
      check(cfg)
      true
    catch _
      false
    end
  end


end # module Config

function train( player :: MCTSPlayer{G}
              ; batchsize :: Int = 512
              , stepsize = batchsize * 10
              , gensize = batchsize * 100
              , generations = 0
              , capacity :: Int
              , snapshot_interval = 5
              , gpu = 0
              , port :: Int
              , ) where {G <: AbstractGame}

  model = training_model(player)
  @assert model isa NeuralModel{G} "Invalid training model: expected neural model"

  models = NeuralModel{G}[]
  push!(models, copy(model |> to_cpu))

  data = Tuple{DataSet{G}, Int}[]

  history = Dict{String, Any}[]

  locks = Dict{Symbol, ReentrantLock}
  locks[:data] = ReentrantLock()
  locks[:model] = ReentrantLock()
  locks[:history] = ReentrantLock()

  @sync begin

    @async begin
      # IO stuff
    end

  end


end



