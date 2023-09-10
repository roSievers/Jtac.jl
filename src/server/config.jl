
# config sections

function train(
  ; host :: String = "127.0.0.1"
  , port :: Int = 7238 
  , name :: String = ENV["USER"] * "-" * string(rand(1:1000))
  , password :: String = ""
  , folder :: String = "./models/%s"
  , logfile :: String = "./logs/%s.txt"
  , snapshot_interval :: Int = 5
  )

  Dict{Symbol, Any}( 
    :host => (host, "training host address")
  , :port => (port, "port the training host listens on")
  , :name => (name, "host name")
  , :password => (password, "password that protects the session")
  , :folder => (folder, "output folder")
  , :logfile => (logfile, "logfile for the session")
  , :snapshot_interval => (snapshot_interval, "number of generations between model snapshots")
  )
end

function play(
  ; host :: String = "127.0.0.1"
  , port :: Int = 7248
  , name :: String = ENV["USER"] * "-" * string(rand(1:1000))
  , password :: String = ""
  , logfile :: String = "./logs/%s.txt"
  , retry_delay :: Float64 = 30.
  )

  Dict{Symbol, Any}( 
    :host => (host, "training host address")
  , :port => (port, "port the training host listens on")
  , :name => (name, "client name")
  , :password => (password, "password to authenticate client")
  , :logfile => (logfile, "logfile for the session")
  , :retry_delay => (retry_delay, "delay in seconds between connection attempts")
  )
end

function play_local(
  ; folder :: String = "./data/%s"
  , packsize :: Int = 10000
  , logfile :: String = "./logs/%s.txt"
  )

  Dict{Symbol, Any}(
    :folder => (folder, "folder the training data goes to")
  , :packsize => (packsize, "minimum size of a saved dataset")
  , :logfile => (logfile, "logfile for the session")
  )
end

function program_training(
  ; model :: String = ""
  , batchsize :: Int = 512
  , stepsize :: Int = 10
  , gensize :: Int = 100
  , gpu ::Int = -2
  , atype :: String = "knet"
  , pool :: Dict = program_training_pool()
  , optimizer :: Dict = program_training_optimizer()
  )

  Dict{Symbol, Any}(
    :model => (model, "path to the model to be trained")
  , :batchsize => (batchsize, "batchsize during training")
  , :stepsize => (stepsize, "number of batches that comprise one step")
  , :gensize => (gensize, "number of steps that comprise one generation")
  , :gpu => (gpu, "gpu device to use for training. -2: auto, -1: cpu, i>0: device i")
  , :atype => (atype, "gpu array type to use for training")
  , :pool => pool
  , :optimizer => optimizer
  )
end

function program_training_pool(
  ; size_min :: Int = 0
  , size_max :: Int = 1000000
  , size_min_test :: Int = 1000
  , size_max_test :: Int = 10000
  , keep_iterations :: Int = 10
  , keep_generations :: Int = 3
  )

  Dict{Symbol, Any}(
    :size_min => (size_min, "minimal size of the train pool for training to continue")
  , :size_max => (size_max, "maximal size (capacity) of the train pool")
  , :size_min_test => (size_min_test, "minimal size of the test pool")
  , :size_max_test => (size_max_test, "maximal size of the test pool")
  , :keep_iterations => (keep_iterations, "how many iterations to keep in the train pool")
  , :keep_generations => (keep_generations, "how many generations to keep in the train/test pool")
  )
end

function program_training_optimizer(
  ; name = "momentum"
  , lr = 0.05
  , gamma = 0.95
  )

  Dict{Symbol, Any}(
    :name => (name, "name of the optimizer. other options vary based on the chosen optimizer")
  , :lr => (lr, "learning rate of the momentum optimizer")
  , :gamma => (gamma, "gamma parameter of the momentum optimizer")
  )
end

function program_selfplay(
  ; model :: String = ""
  , augment :: Bool = false
  , batchsize :: Int = 64
  , ntasks :: Int = 2batchsize
  , spawn :: Bool = true
  , gpu :: Int = -1
  , atype :: String = "knet"
  , player :: Dict = program_selfplay_player()
  , randomize :: Dict = program_selfplay_randomize()
  )

  Dict{Symbol, Any}( 
    :model => (model, "path to the model used in the selfplay program")
  , :augment => (augment, "whether to augment generated datasets")
  , :batchsize => (batchsize, "batchsize of the async player")
  , :ntasks => (ntasks, "number of async selfplay tasks")
  , :spawn => (spawn, "whether the async worker is spawned on a new thread")
  , :gpu => (gpu, "gpu device to use for selfplays. -2: auto, -1: cpu, i>0: device i")
  , :atype => (atype, "gpu array type to use for selfplays")
  , :player => player
  , :randomize => randomize
  )
end

function program_selfplay_player(
  ; power :: Int = 250
  , temperature :: Float64 = 1.0
  , exploration :: Float64 = 1.41
  , dilution :: Float64 = 0.0
  )

  Dict{Symbol, Any}(
    :power => (power, "power of the mcts player")
  , :temperature => (temperature, "temperature of the mcts player")
  , :exploration => (exploration, "exploration value of the mcts player")
  , :dilution => (dilution, "dilution value of the mcts player")
  )
end

function program_selfplay_randomize(
  ; instance :: Float64 = 0.0
  , branch:: Float64 = 0.0
  , branch_step_min :: Int = 1
  , branch_step_max :: Int = 10
  )

  Dict{Symbol, Any}(
    :instance => (instance, "instance randomization for the selfplays")
  , :branch => (branch, "branch probability during the selfplay")
  , :branch_step_min => (branch_step_min, "minimal stepsize when branching")
  , :branch_step_max => (branch_step_max, "maximal stepsize when branching")
  )
end

function jtac()
  program = Dict(
    :training => program_training()
  , :selfplay => program_selfplay()
  )
  Dict(
    :train => train()
  , :play => play()
  , :play_local => play_local()
  , :program => program
  )
end


# Auxiliary functions

function strip_doc!(dict)
  for key in keys(dict)
    if dict[key] isa Dict
      strip_doc!(dict[key])
    else
      @assert dict[key] isa Tuple{Any, String}
      dict[key] = dict[key][1]
    end
  end
  dict
end

function generate_doc!(dict)
  for key in keys(dict)
    if dict[key] isa Dict
      generate_doc!(dict[key])
    else
      @assert dict[key] isa Tuple{Any, String}
      doc = dict[key][2]
      default = dict[key][1]
      type = typeof(default)
      dict[key] = Printf.@sprintf "%s (%s, default: %s)" doc type default
    end
  end
  dict
end

function docs()
  config = jtac()
  generate_doc!(config)
end

function default()
  config = jtac()
  strip_doc!(config)
end

function symbolize(dict)
  d = Dict{Symbol, Any}()
  for key in keys(dict)
    @assert typeof(key) in [String, Symbol]
    if dict[key] isa Dict
      d[Symbol(key)] = symbolize(dict[key])
    else
      d[Symbol(key)] = dict[key] 
    end
  end
  d
end

function stringify(dict)
  d = Dict{String, Any}()
  for key in keys(dict)
    @assert typeof(key) in [String, Symbol]
    if dict[key] isa Dict
      d[string(key)] = stringify(dict[key])
    else
      d[string(key)] = dict[key] 
    end
  end
  d
end

struct KeyError <: Exception
  key :: String
end

struct ValueError <: Exception
  key :: String
  value :: Any
end

isbetween(a, b) = x -> a <= x <= b

function valid_ip(str)
  try
    #TODO: check if str is a valid ip string
    # Sockets.@ip_str(str)
    true
  catch _
    false
  end
end

const LIMITS = Dict{String, Tuple{Function, String}}(
    "train.host" => (valid_ip, "invalid host ip")
  , "train.port" => (isbetween(1000, 10000), "port number must be between 1000 and 10000")
  , "play.host" => (valid_ip, "invalid host ip")
  , "play.port" => (isbetween(1000, 10000), "port number must be between 1000 and 10000")
  , "play_local.packsize" => (>=(1), "packsize must be positive")
  , "program.selfplay.gpu" => (x -> x >= -2, "gpu selector must be -1 (cpu) or >= 0 (gpu)")
  , "program.selfplay.atype" => (in(["knet", "cuda"]), "array type must be one of 'knet' o, 'cuda'")
  , "program.selfplay.ntasks" => (isbetween(1, 4096), "number of tasks must be between 1 and 4096")
  , "program.selfplay.batchsize" => (isbetween(1, 4096), "batchsize must be between 1 and 4096")
  # TODO: other entries!
)

function check_consistency(config, def, path = "")

  diff = setdiff(keys(config), keys(def))
  if !isempty(diff)
    keypath = path * String(first(diff))
    KeyError(keypath) |> throw
  end

  for key in keys(config)
    if def[key] isa Dict
      if !(config[key] isa Dict)
        keypath = path * String(key)
        ValueError(keypath, config[key]) |> throw
      end
      path = path * String(key) * "."
      check_consistency(config[key], def[key], path)
    else
      if typeof(def[key]) != typeof(config[key])
        keypath = path * String(key)
        ValueError(keypath, config[key]) |> throw
      end
    end
  end
end

function load(file)
  def = default()
  config = TOML.parsefile(file)
  config = symbolize(config)
  check_consistency(config, def)
  combine(cval, dval) = begin
    if cval isa Dict
      mergewith(combine, cval, dval)
    else
      cval
    end
  end
  mergewith(combine, config, def)
end

function save(file, config)
  config = stringify(config)
  open(file, "w") do io
    TOML.print(io, config, sorted = true, by = length)
  end
end

convert_param(:: Type{Int}, v :: String) = parse(Int, v)
convert_param(:: Type{Float64}, v :: String) = parse(Float64, v)
convert_param(:: Type{T}, v) where {T} = convert(T, v)

function set_param!(cfg, param :: String, value)
  path = Symbol.(split(param, "."))
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

