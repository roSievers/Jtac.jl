
using ArgParse
using HTTP
using JSON2
using Jtac

include("rencoder.jl")

# -------- Request and Response Types ---------------------------------------- #

struct PlayerRequest
  model::String
  power::Int
  exploration::Float64
end

struct GameRequest
  game::Game
end

function GameRequest(; typ::String, kwargs...)
  if     typ == "TicTacToe" GameRequest(TicTacToe(; kwargs...))
  elseif typ == "MNKGame"   GameRequest(MNKGame(; kwargs...))
  elseif typ == "MetaTac"   GameRequest(MetaTac(; kwargs...))
  elseif typ == "Morris"    GameRequest(Morris(; kwargs...))
  end
end

JSON2.@format GameRequest keywordargs

struct ApplyRequest
  player::PlayerRequest
  game::GameRequest
end

struct ApplyResponse
  value::Float32
  policy::Vector{Float32}
  features::Dict{String,Vector{Float32}}
end

# -------- Auxiliary Functions ----------------------------------------------- #

wrap(f, args...) = HTTP.Response(200, JSON2.write(f(args...)))

function load_models(dir, gpu)

  # Get all files/directories in dir
  files = readdir(dir)

  # Extract the files with '.jtm' ending and load them as async (gpu) models
  models = map(filter(x->length(x) > 4 && x[end - 3:end] == ".jtm", files)) do fname
    name = fname[1:end - 4] # strip '.jtm'
    path = joinpath(dir, name)
    model = gpu ? load_model(path) |> to_gpu : load_model(path)
    name => Async(model)
  end

  Dict{String,Model}("rollout" => RolloutModel(), "random" => RandomModel(), models...)

end

# -------- HTTP Server ------------------------------------------------------- #

function get_reload(req::HTTP.Request, dir, gpu, models)

  # Remove previous models
  empty!(models)

  # Add new ones
  # TODO: We should be more efficient here and only look for newly added
  # / deleted models
  merge!(models, load_models(dir, gpu))

  "Successfully loaded $(length(models)) models."

end

function get_models(req::HTTP.Request, models)
  map(collect(models)) do (name, model)
    Dict( "name"     => name, "game"     => string(gametype(model))
        , "params"   => sum([0; length.(Knet.params(model))])
        , "features" => Jtac.feature_name.(Jtac.features(model)))
  end
end

function get_games(req::HTTP.Request, models)
  ["TicTacToe", "MNKGame", "MetaTac", "Morris"]
end

function post_apply(req::HTTP.Request, models)

  # Parse the json sent via POST
  req = JSON2.read(IOBuffer(HTTP.payload(req)), ApplyRequest)

  # Get model, game, power
  model  = models[req.player.model]
  tmodel = training_model(model)
  game   = req.game.game
  power  = req.player.power

  # A value power <= 1 means that we pick the IntuitionPlayer
  if power <= 1
    player = IntuitionPlayer(model) 
  else
    player = MCTSPlayer(model, power = power, exploration = req.player.exploration)
  end

  # TODO: Get (improved) value of state from player
  v, _, f = apply_features(tmodel, game) # get value and features
  p = think(player, game)                # get improved policy

  # Extract features in nicer dict-format
  feats = Jtac.features(tmodel)
  fdict = Dict{String,Vector{Float32}}(
            Jtac.feature_name(feat) => Jtac.feature_conv(feat, f[idx])
            for (feat, idx) in zip(feats, Jtac.feature_indices(feats, typeof(game))) )
  
  ApplyResponse(v, p, fdict)

end

# -------- Main -------------------------------------------------------------- #

function main(ip, port, dir, gpu)

  # Keyword Arguments

  s = ArgParseSettings()
  @add_arg_table s begin
      "--ip"
      help = "IP address that the server listens to"
      default = "127.0.0.1"
    "--port"
      help = "port that the server listens to"
      arg_type = Int
      default = 4242
    "--dir"
      help = "directory path that is searched for '.jtm' model files"
      default = "."
    "--gpu"
      help = "whether the GPU will be used if CUDA is supported"
      action = :store_true
  end

  args = parse_args(s)

  # Load models from model_dir
  models = load_models(dir, gpu)

  # Create a router that maps addresses to function calls
  router = HTTP.Router()

  # Register the addresses
  HTTP.@register router "GET"  "/reload" (r->wrap(get_reload, r, dir, gpu, models))
  HTTP.@register router "GET"  "/models" (r->wrap(get_models, r, models))
  HTTP.@register router "GET"  "/games"  (r->wrap(get_games, r, models))
  HTTP.@register router "POST" "/apply"  (r->wrap(post_apply, r, models))

  # Start the server
  println("Server listens at $ip:$port.")

  try
    HTTP.serve(router, ip, port)
  catch err
  end

end

main(args["ip"], args["port"], args["dir"], args["gpu"])


