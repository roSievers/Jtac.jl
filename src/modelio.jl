
# -------- Auxiliary functions ----------------------------------------------- #

function dict(t :: String; kwargs...)
  args = [String(key) => val for (key, val) in kwargs]
  Dict{Any, Any}("type" => t, args...)
end
compose(d :: Dict{Any, Any}) = compose(Val(Symbol(d["type"])), d)

# -------- Basic Conversions ------------------------------------------------- #

decompose(p :: Bool)    = p
decompose(p :: Int)     = p
decompose(p :: Float32) = p
decompose(p :: Float64) = p
decompose(p :: String)  = p
#decompose(p :: Symbol)  = p

compose(p :: Bool)    = p
compose(p :: Integer) = Int(p)
compose(p :: Float32) = p
compose(p :: Float64) = p
compose(p :: String)  = p
#compose(p :: Symbol)  = p

decompose(p :: Tuple{Int, Int}) = dict("pair", a = p[1], b = p[2])
decompose(p :: Nothing)         = dict("nothing")

compose(:: Val{:pair}, d)    = (Int(d["a"]), Int(d["b"]))
compose(:: Val{:nothing}, d) = nothing

#function decompose(p :: Array{Int})
#  dict("intarray", dims = collect(size(p)), value = reshape(p, :))
#end

decompose(p :: Vector{Int}) = dict("intvector", value = p)

function decompose(p :: Array{Float32})
  dict("floatarray", dims = decompose(collect(size(p))), value = reshape(p, :))
end

#function compose(:: Val{:intarray}, d)
#  dims = Int.(d["dims"])
#  reshape(convert(Array{Int}, d["value"]), dims...)
#end

compose(:: Val{:intvector}, d) = reinterpret(Int, d["value"])

function compose(:: Val{:floatarray}, d)
  dims = compose(d["dims"])
  data = reinterpret(Float32, d["value"])
  reshape(collect(data), dims...)
end

# -------- Function Conversion ----------------------------------------------- #

decompose(:: typeof(identity)) = dict("function", name = "identity")
decompose(:: typeof(softmax))  = dict("function", name = "softmax")
decompose(:: typeof(relu))     = dict("function", name = "relu")
decompose(:: typeof(tanh))     = dict("function", name = "tanh")
decompose(:: typeof(sigm))     = dict("function", name = "sigm")
decompose(:: typeof(elu))      = dict("function", name = "elu")
decompose(:: typeof(zeros))    = dict("function", name = "zeros")
decompose(:: typeof(ones))     = dict("function", name = "ones")

function compose(:: Val{:function}, d)
  if d["name"] in [ "identity", "softmax", "relu"
                 , "tanh", "sigm", "elu", "zeros", "ones" ]
    eval(Symbol(d["name"]))
  else
    error("Cannot compose function $(d["name"])")
  end
end


# -------- Conversion Macros ------------------------------------------------- #

macro decompose(t, instance)
  names = :(Base.fieldnames(typeof($instance)))
  pairs = :([n => decompose(Base.getfield($instance, n)) for n in $names])
  esc(:(dict($t; $pairs...)))
end

macro compose(t, dict)
  names = :(Base.fieldnames($t))
  args = :([compose($dict[String(entry)]) for entry in $names])
  esc(:($t($args...)))
end

# -------- Knet Conversions -------------------------------------------------- #

decompose(p :: Knet.Param) = dict("parameter", value = decompose(Knet.value(p)))
decompose(bm :: Knet.Ops20.BNMoments) = @decompose "bnmoments" bm

compose(:: Val{:parameter}, d) = Knet.Param(compose(d["value"]))
compose(:: Val{:bnmoments}, d) = @compose Knet.Ops20.BNMoments d 

# -------- Layer Conversions ------------------------------------------------- #

decompose(v :: Vector{Layer})   = dict("layers", value = decompose.(v))
compose(:: Val{:layers}, d)   = Layer[compose(l) for l in d["value"]]


# -------- Feature Conversions ----------------------------------------------- #

# TODO: we need to be able to register features as we are to register games
# for safe compositions
decompose(f :: ConstantFeature) = @decompose "constantfeature" f
compose(:: Val{:constantfeature}, d) = @compose ConstantFeature d

decompose(f :: Vector{Feature}) = dict("features", features = decompose.(f))
compose(:: Val{:features}, d) = compose.(d["features"])


# -------- Game Type Conversions --------------------------------------------- #

function decompose(:: Type{G}) where {G <: AbstractGame}
  dict("gametype", name = String(nameof(G)), params = collect(G.parameters))
end

function compose(:: Val{:gametype}, d)
  name   = compose(d["name"])
  params = compose.(d["params"]) |> collect

  @assert isa(name, String) "Cannot compose gametype"
  @assert all(isbits, params) "Cannot compose gametype"

  Game.GAMES[name](d["params"]...)
end

# -------- Layers ------------------------------------------------------------ #

decompose(l :: Pointwise{false}) = @decompose "pointwise" l
decompose(l :: Dense{false})     = @decompose "dense" l
decompose(l :: Conv{false})      = @decompose "conv" l
decompose(l :: Deconv{false})    = @decompose "deconv" l
decompose(l :: Dropout{false})   = @decompose "dropout" l
decompose(l :: Pool{false})      = @decompose "pool" l
decompose(l :: Batchnorm{false}) = @decompose "batchnorm" l
decompose(l :: Chain{false})     = @decompose "chain" l
decompose(l :: Stack{false})     = @decompose "stack" l
decompose(l :: Residual{false})  = @decompose "residual" l

compose(:: Val{:pointwise}, d) = @compose Pointwise{false} d
compose(:: Val{:dense}, d)     = @compose Dense{false} d
compose(:: Val{:conv}, d)      = @compose Conv{false} d
compose(:: Val{:deconv}, d)    = @compose Deconv{false} d
compose(:: Val{:dropout}, d)   = @compose Dropout{false} d
compose(:: Val{:pool}, d)      = @compose Pool{false} d
compose(:: Val{:batchnorm}, d) = @compose Batchnorm{false} d
compose(:: Val{:chain}, d)     = @compose Chain{false} d
compose(:: Val{:stack}, d)     = @compose Stack{false} d
compose(:: Val{:residual}, d)  = @compose Residual{false} d

# -------- Base Models ------------------------------------------------------- #

decompose(m :: DummyModel) = dict("dummy")
decompose(m :: RandomModel) = dict("random")
decompose(m :: RolloutModel) = dict("rollout")

compose(:: Val{:dummy}, d) = DummyModel()
compose(:: Val{:random}, d) = RandomModel()
compose(:: Val{:rollout}, d) = RolloutModel()

# -------- Neural Models ----------------------------------------------------- #

function decompose(m :: NeuralModel{G, false}) where {G <: AbstractGame}
  d = @decompose "model" m
  push!(d, "gametype" => decompose(G))
  d
end

function compose(:: Val{:model}, d)
  G = compose(d["gametype"])
  @compose NeuralModel{G, false} d
end

# -------- Async Models ------------------------------------------------------ #

function decompose(m :: Async{G}) where {G <: AbstractGame}
  @assert !on_gpu(training_model(m)) "Cannot decompose GPU models"
  dict( "async"
      , model = decompose(m.model)
      , max_batchsize = decompose(m.max_batchsize)
      , buffersize = decompose(m.buffersize) )
end

function compose(:: Val{:async}, d)
  model = compose(d["model"])
  max_batchsize = compose(d["max_batchsize"])
  buffersize = compose(d["buffersize"])
  Async(model, max_batchsize = max_batchsize, buffersize = buffersize)
end

# -------- Caching Models ---------------------------------------------------- #

function decompose(m :: Caching{G}) where {G <: AbstractGame}
  @assert !on_gpu(training_model(m)) "Cannot decompose GPU models"
  dict( "caching"
      , model = decompose(m.model)
      , max_cachesize = decompose(m.max_cachesize) )
end

function compose(:: Val{:caching}, d)
  model = compose(d["model"])
  max_cachesize = compose(d["max_cachesize"])
  Caching(model, max_cachesize = max_cachesize)
end

# -------- Saving & Loading -------------------------------------------------- #

"""
    save(name, model)

Save `model` under the filename `name` with automatically appended extension
".jtm". Note that only the basic model `training_model(model) |> to_cpu` is saved,
so gpu, async, or caching information is lost.
"""
function save(fname :: String, model :: AbstractModel{G}; ext = ".jtm") where {G}
  # Decompose the training_model after bringing it to the cpu
  dict = model |> training_model |> to_cpu |> decompose

  # Save the decomposed model
  open(fname * ext, "w") do io
    MsgPack.pack(io, dict)
  end
end

"""
    load(name)

Load a model from file `name`, where the extension ".jtm" is automatically
appended.
"""
load(fname :: String; ext = ".jtm") = open(fname * ext, "r") do io
  MsgPack.unpack(io) |> compose
end

# -------- Serialization ----------------------------------------------------- #

"""
    serialize(io, model)

Serialize `model` to `io`. Note that this function only accepts models that live
on the CPU.
"""
function serialize(io, model :: AbstractModel{G, false}) where {G}
  MsgPack.pack(io, model |> decompose)
end


"""
    deserialize(io)

Deserialize a Jtac model from `io`. Note that this function only returns models
that live on the CPU.
"""
function deserialize(io) :: AbstractModel
  MsgPack.unpack(io) |> compose
end

