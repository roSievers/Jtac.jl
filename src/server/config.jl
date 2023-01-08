
# config sections

function server(
  ; host :: String = "127.0.0.1"
  , port :: Int = 7238 
  , name :: String = ENV["USER"] * "-" * string(rand(1:1000))
  , snapshot_interval :: Int = 5
  , model_folder :: String = "./model" )

  Dict( :host => host
      , :port => port
      , :name => name
      , :snapshot_interval => snapshot_interval
      , :model_folder => model_folder )
end


function client(
  ; host :: String = "127.0.0.1"
  , port :: Int = 7248
  , isolated :: Bool = false
  , data_folder :: String = "./data"
  , async :: Int = 50
  , name :: String = ENV["USER"] * "-" * string(rand(1:1000))
  , password :: String = ""
  , retry_gap :: Float64 = 30. )

  Dict( :host => host
      , :port => port
      , :isolated => isolated
      , :async => async
      , :name => name
      , :password => password
      , :retry_gap => retry_gap )
end

function training(
  ; model :: String = ""
  , batchsize :: Int = 512
  , stepsize :: Int = 10
  , gensize :: Int = 100
  , optimizer :: String = "momentum"
  , gamma :: Float64 = 0.95
  , lr :: Float64 = 0.05
  , gpu :: Int = -1
  , atype :: String = "cuda" )

  Dict( :model => model
      , :batchsize => batchsize
      , :stepsize => stepsize
      , :gensize => gensize
      , :optimizer => optimizer
      , :gamma => gamma
      , :lr => lr
      , :gpu => gpu
      , :atype => atype )
end

function selfplay(
  ; model :: String = ""
  , power :: Int = 250
  , temperature :: Float64 = 1.
  , exploration :: Float64 = 1.41
  , dilution :: Float64 = 0.0
  , augment :: Bool = false
  , instance_randomization :: Float64 = 0.0
  , branch_probability :: Float64 = 0.0
  , branch_step_min :: Int = 1
  , branch_step_max :: Int = 10
  , gpu :: Int = -1
  , atype :: String = "knet" )

  Dict( :model => model
      , :power => power
      , :temperature => temperature
      , :exploration => exploration
      , :dilution => dilution
      , :augment => augment
      , :instance_randomization => instance_randomization
      , :branch_probability => branch_probability
      , :branch_step_min => branch_step_min
      , :branch_step_max => branch_step_max
      , :gpu => gpu
      , :atype => atype )
end

function pool(
  ; size_min :: Int = 0
  , size_max :: Int = 1_000_000
  , size_min_test :: Int = 1_000
  , size_max_test :: Int = 10_000
  , keep_iterations :: Int = 10
  , keep_generations :: Int = 3 )

  Dict( :size_min => size_min
      , :size_max => size_max
      , :size_min_test => size_min_test
      , :size_max_test => size_max_test
      , :keep_iterations => keep_iterations
      , :keep_generations => keep_generations )
end

# Auxiliary functions

function default()
  Dict( :server => server()
      , :client => client()
      , :training => training()
      , :selfplay => selfplay()
      , :pool => pool() )
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
  cfg = Dict{Symbol, Any}(Symbol(key) => val for (key, val) in cfg)
  diff = setdiff(keys(cfg), keys(def))
  if !isempty(diff)
    collect(diff) |> ConfigKeyError |> throw
  end
  for key in keys(cfg)
    cfg[key] = Dict(Symbol(k) => val for (k, val) in cfg[key])
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
  cfg = Dict{String, Any}(string(key) => val for (key, val) in cfg)
  for key in keys(cfg)
    cfg[key] = Dict(string(k) => val for (k, val) in cfg[key])
  end
  open(file, "w") do io
    TOML.print(io, cfg, sorted = true, by = length)
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

