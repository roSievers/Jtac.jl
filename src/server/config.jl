
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

