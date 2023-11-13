
"""
Loss function that quantifies the discrepancy between target predictions and
their labels.
"""
struct LossFunction
  f :: Function
end

(l :: LossFunction)(x, y) = l.f(x, y)

Base.convert(:: Type{LossFunction}, f :: Function) = LossFunction(f)

"""
Abstract type for model weight regularizations like [`L1Reg`](@ref)
or [`L2Reg`](@ref).
"""
abstract type Regularization end

"""
    isbiasparameter(param)

Check heuristically if `param` is a bias parameter.
"""
isbiasparameter(param) = prod(size(param)) == maximum(size(param))

"""
L1 weight regularization.
"""
struct L1Reg <: Regularization end

(:: L1Reg)(x :: AbstractArray) = sum(abs, x)
(:: L1Reg)(:: AbstractVector{F}) where {F} = F(0)
(r :: L1Reg)(model) = sum(r, Model.parameters(model))

"""
L2 weight regularization.
"""
struct L2Reg <: Regularization end

(:: L2Reg)(x :: AbstractArray) = sum(abs2, x)
(:: L2Reg)(:: AbstractVector{F}) where {F} = F(0)
(r :: L2Reg)(model) = sum(r, Model.parameters(model))

"""
    gettargets(model, dataset; [targets])

Return a named tuple of compatible targets between `model` and `dataset`. If a
list of target names `targets` is provided, the respectively named targets are
returned only.

See also [`Target.compatibletargets`](@ref), on which this function is based.

This function is an internal auxiliary method for [`LossContext`](@ref).
"""
function gettargets(model, dataset; targets = nothing)
  if isnothing(targets)
    targets = Target.compatibletargets(model, dataset)
  else  
    supported = Target.compatibletargets(model, dataset)
    unsupported = setdiff(targets, keys(supported))
    @assert isempty(unsupported) """
    Requested targets $unsupported are not supported by both model and dataset.
    """
    supported[targets]
  end
end

"""
    getlabels(cache, target_names)

Return the label arrays stored in `cache` that correspond to the target names in
`target_names`.

This function is an internal auxiliary method for [`LossContext`](@ref).
"""
function getlabels(cache :: DataCache, target_names)
  labels = []
  for name in target_names
    index = findfirst(isequal(name), cache.target_names)
    @assert !isnothing(index) "Cache object does not support target :$name"
    push!(labels, cache.target_labels[index])
  end
  labels
end

"""
    getlossfunctions(losses, targets)

Return a vector of loss functions derived from the named target tuple
`targets`. The argument `losses` is expected to be a named tuple whose values
resolve to [`LossFunction`](@ref)s. If not specified explicitly in `losses`, the
loss for a target is determined via [`Target.defaultlossfunction`](@ref).

This function is an internal auxiliary method for [`LossContext`](@ref).
"""
function getlossfunctions(losses, targets)
  losses = (;
    map(Target.defaultlossfunction, targets)...,
    losses...
  )
  lossfs = map(l -> resolve(LossFunction, l), losses)
  collect(lossfs[keys(targets)])
end

"""
Structure that stores meta information necessary to evaluate the model loss on a
dataset. It holds as members
- a vector of active targets (`targets`),
- their names (`target_names`),
- their weights (`target_weights`),
- their loss functionals (`target_lossfunctions`),
- a vector of active regularizations (`regs`),
- their names (`reg_names`),
- their weights (`reg_weights`)
"""
struct LossContext
  targets :: Vector{AbstractTarget}
  target_names :: Vector{Symbol}
  target_weights :: Vector{Float32}
  target_lossfunctions :: Vector{LossFunction}

  regs :: Vector{Regularization}
  reg_names :: Vector{Symbol}
  reg_weights :: Vector{Float32}
end

"""
    LossContext(model, dataset; [losses, regs, targets, weights])
    LossContext(model, cache; [losses, regs, targets, weights])

Construct a `LossContext` object.

By default, all targets supported by both `model` and `dataset`/`cache` are
activated. Use the argument `targets` to specify a subset only. The target loss
functions can be customized via the named tuple `losses`. If left unspecified
for a target, the loss function returned by [`Target.defaultlossfunction`](@ref)
is used.

The argument `regs` must be a named tuple of [`Regularization`](@ref)s, whose
names must be distinct from active target names.

The weights for both target and regularization components can be adjusted via
the named tuple `weights`. Each weight defaults to `1f0`.

See also [`losscomponents`](@ref) and [`loss`](@ref).
"""
function LossContext( model :: NeuralModel{G, B}
                    , dataset :: Union{DataSet{G}, DataCache{G, T}} = DataSet(model)
                    ; losses = (;)
                    , reg = (;)
                    , targets = nothing           
                    , weights = (;)
                    ) where {G, F, T <: AbstractArray{F}, B <: Backend{T}}

  # convert arguments to named tuples
  reg = (; reg...)
  weights = (; weights...)
  losses = (; losses...)

  targets = gettargets(model, dataset; targets)
  target_names = collect(keys(targets))
  target_lossfunctions = getlossfunctions(losses, targets)
  target_weights = [F(get(weights, name, 1)) for name in target_names]
  
  reg_names = collect(keys(reg))
  reg_weights = [F(get(weights, name, 1)) for name in reg_names]

  @assert isempty(intersect(target_names, reg_names)) """
  The same name cannot be used for both a prediction target and a regularization
  term.
  """

  LossContext(
    collect(targets),
    target_names,
    target_weights,
    target_lossfunctions,
    collect(reg),
    reg_names,
    reg_weights,
  )
end

"""
    excludetargets(context)

Return a `LossContext` derived from `context` that deactivates all prediction
targets.
"""
function excludetargets(ctx :: LossContext)
  LossContext([], [], [], [], ctx.regs, ctx.reg_names, ctx.reg_weights)
end

"""
    excluderegularizations(context)

Return a `LossContext` derived from `context` that deactivates all
regularization terms.
"""
function excluderegularizations(ctx :: LossContext)
  LossContext(
    ctx.targets,
    ctx.target_names,
    ctx.target_weights,
    ctx.target_lossfunctions,
    [],
    [],
    [],
  )
end

"""
    losscomponents(model, dataset, context; batchsize)
    losscomponents(model, cache, context)
    losscomponents(model, dataset; batchsize, kwargs...)
    losscomponents(model, cache; kwargs...)

Calculate the losses of `model` on `cache` or `dataset` for all prediction
targets activated in `context`. If no `context` of type [`LossContext`](@ref) is
provided explicitly, it is derived from `kwargs`.

The argument `batchsize` can be used to limit the number of game states
evaluated in one batch by `model` if a dataset `dataset` passed. This may
reduce memory requirements.

See also [`loss`](@ref) and [`LossContext`](@ref).
"""
function losscomponents( model :: NeuralModel{G, B}
                       , cache :: DataCache{G, T}
                       , ctx :: LossContext
                       ) where {G, T <: AbstractArray, B <: Backend{T}}

  labels = getlabels(cache, ctx.target_names)
  outputs = model(cache.data; targets = ctx.target_names) 
  lossfs = ctx.target_lossfunctions
  weigths = ctx.target_weights

  tlosses = map((l, x, y, w) -> w * l(x, y), lossfs, outputs, labels, weigths)
  tlosses ./= length(cache)
  rlosses = Float32[w * r(model) for (r, w) in zip(ctx.regs, ctx.reg_weights)]

  (; zip(ctx.target_names, tlosses)..., zip(ctx.reg_names, rlosses)...)
end

function losscomponents(model, cache :: DataCache; kwargs...)
  ctx = LossContext(model, cache; kwargs...)
  losscomponents(model, cache, ctx)
end

function losscomponents( model :: NeuralModel{G, B}
                       , dataset :: DataSet{G}
                       , ctx :: LossContext
                       ; batchsize :: Integer = 1024
                       ) where {G, T <: AbstractArray, B <: Backend{T}}

  lossfs = ctx.target_lossfunctions
  weights = ctx.target_weights
  tlosses = sum(DataBatches(T, dataset, batchsize)) do cache
    labels = getlabels(cache, ctx.target_names)
    outputs = model(cache.data; targets = ctx.target_names)
    map((l, x, y, w) -> w * l(x, y), lossfs, outputs, labels, weights)
  end
  tlosses ./= length(dataset)

  rlosses = Float32[w * r(model) for (r, w) in zip(ctx.regs, ctx.reg_weights)]

  (; zip(ctx.target_names, tlosses)..., zip(ctx.reg_names, rlosses)...)
end

function losscomponents(model, dataset :: DataSet; batchsize = 1024, kwargs...)
  ctx = LossContext(model, dataset; kwargs...)
  losscomponents(model, dataset, ctx; batchsize)
end


"""
    loss(model, dataset, context; batchsize)
    loss(model, cache, context)
    loss(model, dataset; batchsize, kwargs...)
    loss(model, cache; kwargs...)

Calculate the total loss of `model` on `cache` for all prediction targets
activated in `context`. If no `context` of type [`LossContext`](@ref) is
provided, it is derived from `kwargs`.

The argument `batchsize` can be used to limit the number of game states
evaluated in one batch by `model` if a dataset `dataset` passed. This may
reduce memory requirements.

See also [`losscomponents`](@ref) and [`LossContext`](@ref).
"""
function loss(model, cache :: DataCache, ctx :: LossContext)
  labels = getlabels(cache, ctx.target_names)
  outputs = model(cache.data; targets = ctx.target_names)

  tloss = 0f0
  weights = ctx.target_weights
  lossfs = ctx.target_lossfunctions
  foreach(lossfs, outputs, labels, weights) do l, x, y, weight
    tloss += weight * l(x, y)
  end
  
  rloss = 0f0
  foreach(ctx.regs, ctx.reg_weights) do r, weight
    rloss += weight * r(model)
  end

  tloss / length(cache) + rloss
end

function loss( model :: NeuralModel{G, B}
             , cache :: DataCache{G}
             ; kwargs...
             ) where {G, T <: AbstractArray, B <: Backend{T}}

  ctx = LossContext(model, cache; kwargs...)
  loss(model, cache, ctx)
end

function loss( model :: NeuralModel{G, B}
             , dataset :: DataSet{G}
             ; batchsize = 1024
             , kwargs...
             ) where {G, T <: AbstractArray, B <: Backend{T}}

  ctx = LossContext(model, dataset; kwargs...)
  tloss = sum(DataBatches(T, dataset, batchsize)) do cache
    loss(model, cache, excluderegularizations(ctx)) * length(cache)
  end

  rloss = 0f0
  foreach(ctx.regs, ctx.reg_weights) do r, weight
    rloss += weight * r(model)
  end

  tloss / length(dataset) + rloss
end


"""
   _loss(trunk, heads, activations, cache, context) 

Low-level loss function that can be used by autodiff-based training backends.
"""
function _loss(trunk, heads, activations, data, labels, weights, losses)
  tmp = trunk(data)
  sum(1:length(heads)) do index
    h = heads[index]
    a = activations[index]

    w = weights[index]
    l = losses[index]
    y = labels[index]

    x = a(h(tmp))
    w * l(x, y)
  end
end


"""
   _regloss(trunk, heads, activations, cache, context) 

Low-level regularization loss function that can be used by autodiff-based training backends.
"""
function _regloss(params, ctx :: LossContext)
  if !isempty(ctx.regs)
    sum(1:length(ctx.regs)) do index
      r = ctx.regs[index]
      w = ctx.reg_weights[index]
      w * sum(r, params)
    end
  else
    0f0
  end
end

"""
    setup(model, context, opt) 

Set up an optimizer `opt` for model `model` and loss context `context`.

Must be implemented by trainable backends together with [`step!`](@ref).
"""
function setup(model :: NeuralModel, ctx :: LossContext, opt)
  error("Training is not supported for this backend")
end


"""
    step!(model, cache, context, opt_setup)

Train `model` for one training step on the data in `cache` in the loss context
`context` via the optimizer setup `opt_setup`.

Must be implemented by trainable backends together with [`setup`](@ref).
"""
function step!(model, cache, context, opt)
  error("Training is not supported for this backend")
end

"""
    printlossheader(ctx)

Print a header row that announces the loss components in ctx.
"""
function printlossheader(ctx :: LossContext)
  targets = map(ctx.target_names) do name
    str = string(name)[1:min(end, 8)]
    Printf.@sprintf("%8s", str)
  end
  regs = map(ctx.reg_names) do name
    str = string(name)[1:min(end, 8)]
    Printf.@sprintf("%8s", str)
  end
  strs = ["#", "   epoch", targets..., regs..., "     total", "    length"]
  println(join(strs, " "))
end

"""
    printlossvalues(model, data, ctx; [batchsize, color, epoch])

Print the loss values of `model` applied to `data` in the loss context `ctx`.
"""
function printlossvalues( model
                        , data
                        , ctx :: LossContext
                        ; batchsize = 1024
                        , color = :normal
                        , epoch = 0 )

  lc = losscomponents(model, data, ctx; batchsize)
  losses = map(x -> @sprintf("%8.3f", x), values(lc))
  lossstr = join(losses, " ")
  str = @sprintf("%8d %s %8.3f %8d\n", epoch, lossstr, sum(lc), length(data))
  printstyled(str; color)
end

"""
    printranking(ranking)

Print the results of a contest.
"""
function printranking(rk :: Player.Ranking)
  str = "# " * replace(string(rk, true), "\n" => "\n# ") * "\n#"
  println(str)
end


"""
    learn!(model, data, context, opt; kwargs...)     

Train `model` on `data` with loss context `context` and optimizer `opt`.
"""
function learn!( model :: NeuralModel{G, B}
               , ds :: DataSet{G}
               , ctx :: LossContext
               , opt
               ; test = nothing
               , epochs = 10
               , batchsize = 64
               , shuffle = true
               , partial = true
               , callback = _ -> nothing
               , callback_epoch = _ -> nothing
               , verbose = true
               ) where {G, T <: AbstractArray, B <: Backend{T}}

  @assert length(ds) > 0 "Training dataset is empty"

  model = trainingmodel(model)
  if verbose
    printlossheader(ctx)
  end

  opt_setup = setup(model, ctx, opt)

  totalsteps = 0
  for epoch in 1:epochs

    if verbose
      steps = ceil(Int, length(ds) / batchsize)
      step, finish = Util.stepper("# Learning...", steps)
    end

    for cache in DataBatches(T, ds, batchsize; shuffle, partial)
      step!(model, cache, ctx, opt_setup)
      callback(totalsteps += 1)
      if verbose
        step()
      end
    end

    if verbose
      finish()
      if !isnothing(test)
        printlossvalues(model, ds, ctx; epoch, batchsize, color = 245)
        printlossvalues(model, test, ctx; epoch, batchsize)
      else
        printlossvalues(model, ds, ctx; epoch, batchsize, color = 245)
      end
    end

    callback_epoch(epoch)
  end
end

# function learn!()
