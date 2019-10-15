
# -------- Auxiliary functions ----------------------------------------------- #

dict(t :: Symbol; kwargs...) = Dict{Symbol, Any}(:type => t, kwargs...)
compose(d :: Dict{Symbol, Any}) = compose(Val(d[:type]), d)

# -------- Basic Conversions ------------------------------------------------- #

decompose(p :: Bool)    = p
decompose(p :: Int)     = p
decompose(p :: Float64) = p
decompose(p :: String)  = p

compose(p :: Bool)    = p
compose(p :: Int)     = p
compose(p :: Float64) = p
compose(p :: String)  = p

decompose(p :: Array)           = dict(:array, value = p)
decompose(p :: Tuple{Int, Int}) = dict(:pair, a = p[1], b = p[2])
decompose(p :: Nothing)         = dict(:nothing)

compose(:: Val{:array}, d)   = d[:value]
compose(:: Val{:pair}, d)    = (d[:a], d[:b])
compose(:: Val{:nothing}, d) = nothing

# -------- Function Conversion ----------------------------------------------- #

decompose(:: typeof(identity)) = dict(:function, name = :identity)
decompose(:: typeof(softmax))  = dict(:function, name = :softmax)
decompose(:: typeof(relu))     = dict(:function, name = :relu)
decompose(:: typeof(tanh))     = dict(:function, name = :tanh)
decompose(:: typeof(sigm))     = dict(:function, name = :sigm)
decompose(:: typeof(elu))      = dict(:function, name = :elu)
decompose(:: typeof(zeros))    = dict(:function, name = :zeros)
decompose(:: typeof(ones))     = dict(:function, name = :ones)

function compose(:: Val{:function}, d)
  if d[:name] in [ :identity, :softmax, :relu
                 , :tanh, :sigm, :elu, :zeros, :ones]
    eval(d[:name])
  else
    error("Cannot compose function $(d[:name])")
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
  args = :([compose($dict[entry]) for entry in $names])
  esc(:($t($args...)))
end

# -------- Knet Conversions -------------------------------------------------- #

decompose(p :: Knet.Param)      = dict(:parameter, value = Knet.value(p))
decompose(bm :: Knet.BNMoments) = @decompose :bnmoments bm

compose(:: Val{:parameter}, d) = Knet.Param(d[:value])
compose(:: Val{:bnmoments}, d) = @compose Knet.BNMoments d 

# -------- Jtac Conversions -------------------------------------------------- #

decompose(v :: Vector{Layer})   = dict(:layers, value = decompose.(v))
decompose(f :: Vector{Feature}) = dict(:features, value = f)
decompose(:: Type{G}) where {G <: Game} = dict(:gametype, name = string(G))

compose(:: Val{:layers}, d)   = Layer[compose(l) for l in d[:value]]
compose(:: Val{:features}, d) = d[:value]
compose(:: Val{:gametype}, d) = eval(Meta.parse(d[:name])) # TODO: check if type!

# TODO: The :gametype and :features decompositions are not transparent!
# But maybe this is not a problem, as they are not needed for remote
# reconstruction of playing ability?


# -------- Layers ------------------------------------------------------------ #

decompose(l :: Pointwise{false}) = @decompose :pointwise l
decompose(l :: Dense{false})     = @decompose :dense l
decompose(l :: Conv{false})      = @decompose :conv l
decompose(l :: Deconv{false})    = @decompose :deconv l
decompose(l :: Dropout{false})   = @decompose :dropout l
decompose(l :: Pool{false})      = @decompose :pool l
decompose(l :: Batchnorm{false}) = @decompose :batchnorm l
decompose(l :: Chain{false})     = @decompose :chain l
decompose(l :: Stack{false})     = @decompose :stack l

compose(:: Val{:pointwise}, d) = @compose Pointwise{false} d
compose(:: Val{:dense}, d)     = @compose Dense{false} d
compose(:: Val{:conv}, d)      = @compose Conv{false} d
compose(:: Val{:deconv}, d)    = @compose Deconv{false} d
compose(:: Val{:dropout}, d)   = @compose Dropout{false} d
compose(:: Val{:pool}, d)      = @compose Pool{false} d
compose(:: Val{:batchnorm}, d) = @compose Batchnorm{false} d
compose(:: Val{:chain}, d)     = @compose Chain{false} d
compose(:: Val{:stack}, d)     = @compose Stack{false} d

# -------- Base Models ------------------------------------------------------- #

decompose(m :: DummyModel) = dict(:dummy)
decompose(m :: RandomModel) = dict(:random)
decompose(m :: RolloutModel) = dict(:rollout)

compose(:: Val{:dummy}, d) = DummyModel()
compose(:: Val{:random}, d) = RandomModel()
compose(:: Val{:rollout}, d) = RolloutModel()

# -------- Neural Models ----------------------------------------------------- #

function decompose(m :: NeuralModel{G, false}) where {G <: Game}
  d = @decompose :model m
  push!(d, :gametype => decompose(G))
  d
end

function compose(:: Val{:model}, d)
  G = compose(d[:gametype])
  @compose NeuralModel{G, false} d
end

# -------- Async Models ------------------------------------------------------ #

function decompose(m :: Async{G}) where {G <: Game}
  @assert !on_gpu(training_model(m)) "Cannot decompose GPU models"
  dict( :async
      , model = decompose(m.model)
      , max_batchsize = decompose(m.max_batchsize)
      , buffersize = decompose(m.buffersize) )
end

function compose(:: Val{:async}, d)
  model = compose(d[:model])
  max_batchsize = compose(d[:max_batchsize])
  buffersize = compose(d[:buffersize])

  Async(model, max_batchsize = max_batchsize, buffersize = buffersize)
end

# -------- Saving Models ----------------------------------------------------- #

"""
    save_model(name, model)

Save `model` under the filename `name` with automatically appended extension
".jtm". Note that the model is first converted to a saveable format, i.e., it is
moved to the CPU and `training_model(model)` is extracted.
"""
function save_model(fname :: String, model :: Model{G}) where {G}

  # Decompose the training_model after bringing it to the cpu
  dict = model |> training_model |> to_cpu |> decompose

  # Save the decomposed model
  BSON.bson(fname * ".jtm", model = dict)

end

"""
    load_model(name)

Load a model from file `name`, where the extension ".jtm" is automatically
appended.
"""
load_model(fname :: String) = BSON.load(fname * ".jtm")[:model] |> compose

# -------- Saving and Loading Datasets --------------------------------------- #

"""
    save_dataset(name, dataset)

Save `dataset` under filename `name` with automatically appended extension
".jtd". Dataset caches are not saved.
"""
function save_dataset(fname :: String, d :: DataSet)
  
  # Save the file
  BSON.bson(fname * ".jtd", dataset = d) 

end

"""
    load_dataset(name)

Load a dataset from file "name", where the extension ".jtd" is automatically
appended.
"""
load_dataset(fname :: String) = BSON.load(fname * ".jtd")[:dataset]

