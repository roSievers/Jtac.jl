
# -------- Auxiliary functions ----------------------------------------------- #

dict(t :: Symbol; kwargs...) = Dict{Symbol, Any}(:type => t, kwargs...)
compose(d :: Dict{Symbol, Any}) = compose(Val(d[:type]), d)

# -------- Basic Conversions ------------------------------------------------- #

decompose(p :: Float64)         = dict(:float, value = p)
decompose(p :: String)          = dict(:string, value = p)
decompose(p :: Array)           = dict(:array, value = p)
decompose(p :: Tuple{Int, Int}) = dict(:pair, value = p)
decompose(p :: Knet.Param)      = dict(:parameter, value = Knet.value(p))

compose(:: Val{:parameter}, d) = Knet.Param(d[:value])
compose(:: Val{:array}, d) = d[:value]
compose(:: Val{:pair}, d) = d[:value]
compose(:: Val{:string}, d) = d[:value]
compose(:: Val{:float}, d) = d[:value]


# -------- Function Conversion ----------------------------------------------- #

decompose(:: typeof(identity)) = dict(:function, name = :identity)
decompose(:: typeof(softmax))  = dict(:function, name = :softmax)
decompose(:: typeof(relu))     = dict(:function, name = :relu)
decompose(:: typeof(tanh))     = dict(:function, name = :tanh)
decompose(:: typeof(sigm))     = dict(:function, name = :sigm)
decompose(:: typeof(elu))      = dict(:function, name = :elu)

function compose(:: Val{:function}, d :: Dict{Symbol, Any})
  if d[:name] in [:identity, :softmax, :relu, :tanh, :sigm, :elu]
    eval(d[:name])
  else
    error("Cannot compose function $(d[:name])")
  end
end

# -------- Layers ------------------------------------------------------------ #

decompose(v :: Vector{Layer}) = dict(:layers, value = decompose.(v))
compose(:: Val{:layers}, d) = Layer[compose(layer) for layer in d[:value]]

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

decompose(l :: Pointwise{false}) = @decompose :pointwise l
decompose(l :: Dense{false})     = @decompose :dense l
decompose(l :: Conv{false})      = @decompose :conv l
decompose(l :: Deconv{false})    = @decompose :deconv l
decompose(l :: Dropout{false})   = @decompose :dropout l
decompose(l :: Pool{false})      = @decompose :pool l
decompose(l :: Batchnorm{false}) = error("Cannot decompose Batchnorm yet")

compose(:: Val{:pointwise}, d) = @compose Pointwise{false} d
compose(:: Val{:dense}, d)     = @compose Dense{false} d
compose(:: Val{:conv}, d)      = @compose Conv{false} d
compose(:: Val{:deconv}, d)    = @compose Deconv{false} d
compose(:: Val{:dropout}, d)   = @compose Dropout{false} d
compose(:: Val{:pool}, d)      = @compose Pool{false} d
compose(:: Val{:batchnorm}, d) = error("Cannot compose Batchnorm yet")

decompose(l :: Chain{false}) = @decompose :chain l
decompose(l :: Stack{false}) = @decompose :stack l

compose(:: Val{:chain}, d) = @compose Chain{false} d
compose(:: Val{:stack}, d) = @compose Stack{false} d

# -------- Models ------------------------------------------------------------ #

# TODO: These decompositions are not transparent! But maybe this is not
# a problem, as they are not needed for remote reconstruction of playing
# ability.
decompose(:: Type{G}) where {G <: Game} = dict(:gametype, name = G)
decompose(f :: Vector{Feature}) = dict(:features, value = f)

compose(:: Val{:gametype}, d) = d[:name]
compose(:: Val{:features}, d) = d[:value]

function decompose(m :: Model{G, false}) where {G <: Game}
  d = @decompose :model m
  push!(d, :gametype => decompose(G))
  d
end

function compose(:: Val{:model}, d)
  G = compose(d[:gametype])
  @compose NeuralModel{G, false} d
end


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
function load_model(fname :: String)
  BSON.load(fname * ".jtm")[:model] |> compose
end

