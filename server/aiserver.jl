
using Printf
using ArgParse
using HTTP
using JSON2
using Jtac

# -------- List of supported games ------------------------------------------- #

const GAMES = Dict(
    "tictactoe" => [:TicTacToe]
  , "mnkgame"   => [:MNKGame, Int, Int, Int]
  , "metatac"   => [:MetaTac]
  , "morris"    => [:Morris]
)


# -------- En-/decoding visual / renderable content -------------------------- #

include("rencoder.jl")

# -------- En-/decoding game states ------------------------------------------ #

struct GameState
  typ :: String
  params :: Vector{String}
  game :: Game
end

function GameState(game :: Game)

  typ = string(typeof(game))
  ff = findfirst('{', typ)
  fl = findlast('}', typ)

  if !isnothing(ff) && !isnothing(fl)
    params = split(typ[ff+1:fl-1], ",")
    typ = lowercase(typ[1:ff-1])
  else
    typ = lowercase(typ)
    params = String[]
  end

  GameState(typ, params, game)
end

function JSON2.read(io :: IO, :: Type{GameState})

  fields = JSON2.read(io)
  typ = fields.typ

  gametype = GAMES[typ][1]
  paramtypes = GAMES[typ][2:end]
  params = map(parse, paramtypes, fields.params)

  cexpr = isempty(params) ? gametype : Expr(:curly, gametype, params...)
  game = eval(cexpr)()

  for field in fieldnames(typeof(game))
    value = convert(getfield(game, field) |> typeof, getfield(fields.game, field))
    setfield!(game, field, value)
  end

  GameState(game)

end


# -------- Requests and Responses -------------------------------------------- #

struct TurnRequest
  game :: GameState
  action :: Int
end

TurnRequest(; game, action) = TurnRequest(game, action)
JSON2.@format TurnRequest keywordargs

struct ApplyRequest
  game :: GameState
  model :: String
  power :: Int
  temperature :: Float32
  exploration :: Float32
end

function ApplyRequest(; game, model, power, temperature, exploration)
  ApplyRequest(game, model, power, temperature, exploration)
end
JSON2.@format ApplyRequest keywordargs

struct ApplyResponse
  value :: Float32
  policy :: Vector{Float32}
  features :: Dict{String,Vector{Float32}}
end

# -------- Auxiliary Functions ----------------------------------------------- #

wrap(f, args...) = HTTP.Response(200, JSON2.write(f(args...)))

function load_models(dir, gpu)

  # Get all files/directories in dir
  files = readdir(dir)

  # Extract the files with '.jtm' ending and load them as async (gpu) models
  models = map(filter(x->length(x) > 4 && x[end-3:end] == ".jtm", files)) do fname
    name = fname[1:end - 4] # strip '.jtm'
    path = joinpath(dir, name)
    model = gpu ? load_model(path) |> to_gpu : load_model(path)
    name => Async(model)
  end

  Dict{String,Model}( "rollout" => RolloutModel()
                    , "random" => RandomModel()
                    , models... )

end

# -------- Handling GET Requests --------------------------------------------- #

function GET_index(req::HTTP.Request, index)
  if index == ""
    content ="Welcome the the Jtac AI server. No index file was provided" 
  else
    content = read(index)
  end
  HTTP.Response(200, content)
end

function GET_reload(req::HTTP.Request, dir, gpu, models)

  # Remove previous models
  empty!(models)

  # Add new ones
  # TODO: We should be more efficient here and only look for newly added
  # / deleted models
  merge!(models, load_models(dir, gpu))

  "Successfully loaded $(length(models)) models."

end

function GET_models(req::HTTP.Request, models)
  map(collect(models)) do (name, model)
    Dict( "name"     => name, "game"     => string(gametype(model))
        , "params"   => sum([0; length.(Knet.params(model))])
        , "features" => Jtac.feature_name.(Jtac.features(model)))
  end
end

function GET_games(req::HTTP.Request)
  map(keys(GAMES), values(GAMES)) do k, v
    Dict("typ" => k, "params" => v[2:end])
  end
end

function GET_create(req::HTTP.Request)
  typ = HTTP.URIs.splitpath(req.target)[3]
  params = HTTP.URIs.splitpath(req.target)[4:end]

  gametype = GAMES[typ][1]
  paramtypes = GAMES[typ][2:end]
  params = map(parse, paramtypes, params)
  
  cexpr = isempty(params) ? gametype : Expr(:curly, gametype, params...)
  GameState(eval(cexpr)())
end

function GET_dummyimage(req::HTTP.Request)
  game = TicTacToe()
  for i in rand(1:5) random_turn!(game) end
  Image(game, rand(Float32, 9), 0.5f0) |> Typed
end

function POST_turn(req::HTTP.Request)
  req = JSON2.read(IOBuffer(HTTP.payload(req)), TurnRequest)
  apply_action!(req.game.game, req.action)
  GameState(req.game.game)
end

function POST_apply(req::HTTP.Request, models)

  # Parse target and json and sent via POST
  paths = HTTP.URIs.splitpath(req.target)

  req = JSON2.read(IOBuffer(HTTP.payload(req)), ApplyRequest)

  # Get model, game, power
  model  = models[req.model]
  game   = req.game.game
  power  = req.power

  # A value power <= 1 means that we pick the IntuitionPlayer
  if power <= 1
    player = IntuitionPlayer(model, temperature = req.temperature) 
  else
    player = MCTSPlayer( model
                       , power = power
                       , temperature = req.temperature
                       , exploration = req.exploration )
  end

  # TODO: Get (improved) value of state from player!
  v, _, f = apply_features(base_model(model), game) # get value and features
  p = think(player, game)                # get improved policy

  # Bring features in nicer format as dictionary
  feats = Jtac.features(base_model(model))
  fdict = Dict{String,Vector{Float32}}(
            Jtac.feature_name(feat) => Jtac.feature_conv(feat, f[idx])
            for (feat, idx) in zip(feats, Jtac.feature_indices(feats, typeof(game))) )
  
  # If a visual response is desired, return the corresponding image data
  if length(paths) == 3 && paths[3] == "visual"

    Image(game, p, v) |> Typed

  # Else, send value/policy/feature data directly
  else

    ApplyResponse(v, p, fdict)

  end

end

# -------- Main entry function ----------------------------------------------- #

function aiserver( ip = "127.0.0.1"
                 , port = "4242"
                 , dir = "."
                 , gpu = "false"
                 , index = "" )

  # Load models from model_dir
  models = load_models(dir, gpu)

  # Create a router that maps addresses to function calls
  router = HTTP.Router()

  # Register addresses
  HTTP.@register router "GET"  "/" (r->GET_index(r, index))

  HTTP.@register router "GET"  "/api/reload" (r->wrap(GET_reload, r, dir, gpu, models))
  HTTP.@register router "GET"  "/api/models" (r->wrap(GET_models, r, models))
  HTTP.@register router "GET"  "/api/games"  (r->wrap(GET_games, r))
  HTTP.@register router "GET"  "/api/create" (r->wrap(GET_create, r))

  HTTP.@register router "GET"  "/api/dummyimage" (r->wrap(GET_dummyimage, r))

  HTTP.@register router "POST" "/api/turn"   (r->wrap(POST_turn, r))
  HTTP.@register router "POST" "/api/apply" (r->wrap(POST_apply, r, models))


  # Start the server
  println("Server listens on $ip:$port.")

  try

    HTTP.serve(router, ip, port)

  catch err

    println(err)

  end

end

# -------- Parsing arguments and calling entry point ------------------------- #

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
  "--index"
    help = "statically served index page"
    default = ""
end

args = parse_args(s)

aiserver(args["ip"], args["port"], args["dir"], args["gpu"], args["index"])

