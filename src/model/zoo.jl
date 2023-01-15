
# -------- Linear Neural Model ----------------------------------------------- #

function Shallow(G :: Type{<: AbstractGame}; kwargs...)
  NeuralModel(G, Pointwise(); kwargs...)
end


# -------- Multilayer Perceptron --------------------------------------------- #

function MLP(G :: Type{<: AbstractGame}, hidden, f = "relu"; kwargs...)
  widths = [ prod(size(G)), hidden...]
  layers = [ Dense(widths[j], widths[j+1], f) for j in 1:length(widths) - 1 ]

  NeuralModel(G, Chain(layers...); kwargs...)
end


# -------- Shallow Convolutional Network ------------------------------------ #

function ShallowConv( G :: Type{<: AbstractGame}
                    , f = "relu"
                    ; filters = 64
                    , kwargs... )

  NeuralModel(G, Conv(size(G)[3], filters, f); kwargs...)
end


# -------- Architecture used for Alpha Zero ---------------------------------- #

function zero_res_block(G :: Type{<: AbstractGame}, filters :: Int)
  Model.@residual (size(G)[1:2]..., filters) f = "relu" begin
    Conv(filters, window = 3, padding = 1, stride = 1)
    Batchnorm("relu")
    Conv(filters, window = 3, padding = 1, stride = 1)
    Batchnorm()
  end
end

function zero_conv_block(G :: Type{<: AbstractGame}, ci :: Int, co :: Int)
  Model.@chain (size(G)[1:2]..., ci) begin
    Conv(co, window = 3, padding = 1, stride = 1)
    Batchnorm("relu")
  end
end

function zero_vhead(G :: Type{<: AbstractGame}, filters)
  shape = (size(G)[1:2]..., filters)
  Model.@chain shape begin
    Conv(32, window = 1, padding = 0, stride = 1)
    Batchnorm("relu")
    Dense(256, "relu")
    Dense(1)
  end
end

function zero_phead(G :: Type{<: AbstractGame}, filters)
  shape = (size(G)[1:2]..., filters)
  Model.@chain shape begin
    Conv(32, window = 1, padding = 0, stride = 1)
    Batchnorm("relu")
    Dense(Game.policy_length(G))
  end
end

function ZeroConv( G :: Type{<: AbstractGame}
                 ; blocks = 6, filters = 256
                 , vhead = zero_vhead(G, filters)
                 , phead = zero_phead(G, filters) )
  @assert blocks >= 1
  ci = size(G)[3]
  trunk = Model.Chain( zero_conv_block(G, ci, filters)
                     , [zero_conv_block(G, filters, filters) for i in 1:(blocks-1)]...)
  NeuralModel(G, trunk; vhead = vhead, phead = phead)
end

function ZeroRes( G :: Type{<: AbstractGame}
                ; blocks = 6, filters = 256
                , vhead = zero_vhead(G, filters)
                , phead = zero_phead(G, filters) )
  @assert blocks >= 1
  ci = size(G)[3]
  trunk = Model.Chain( zero_conv_block(G, ci, filters)
                     , [zero_res_block(G, filters) for i in 1:(blocks-1)]...)
  NeuralModel(G, trunk; vhead = vhead, phead = phead)
end


function zero_shrink( model :: NeuralModel{G, false}
                    ; blocks = nothing
                    , filters = nothing ) where {G <: AbstractGame}

  orig_blocks = length(model.trunk.layers)
  orig_filters = size(model.trunk.layers[1].layers[1].w.data, 4)

  blocks = isnothing(blocks) ? orig_blocks : blocks
  filters = isnothing(filters) ? orig_filters : filters

  @assert 1 <= blocks <= length(model.trunk.layers)
  @assert 1 <= filters <= size(model.trunk.layers[1].layers[1].w.data, 4)

  m = copy(model)
  resize!(m.trunk.layers, blocks)

  # trunk blocks
  for (i, block) in enumerate(m.trunk.layers)

    for layer in Model.layers(block)
      _shrink_primitive!(layer, orig_filters, filters, i == 1)
    end

  end

  # heads
  for head in m.heads
    outlen = prod(Model.outsize(m.trunk, size(G)))
    _shrink_head!(head, filters, outlen)
  end

  m
end

function _shrink_primitive!(l :: Model.Conv, _, filters, first)
  l.b.data = l.b.data[:,:,1:filters,:]
  if first
    l.w.data = l.w.data[:,:,:,1:filters]
  else
    l.w.data = l.w.data[:,:,1:filters,1:filters]
  end
  nothing
end

function _shrink_primitive!(l :: Model.Batchnorm, orig_filters, filters, _)
  tmp = copy(l.params.data)
  l.params.data = tmp[1:2filters]
  l.params.data[(filters+1):2filters] = tmp[orig_filters+1:orig_filters + filters]
  l.moments.mean = isnothing(l.moments.mean) ? nothing : l.moments.mean[:,:,1:filters,:]
  l.moments.var = isnothing(l.moments.var) ? nothing : l.moments.var[:,:,1:filters,:]
  nothing
end

_shrink_head!(l :: Model.CompositeLayer, args...) = _shrink_head!(Model.layers(l)[1], args...)

function _shrink_head!(l :: Model.Conv, filters, _)
  l.w.data = l.w.data[:,:,1:filters,:]
end

function _shrink_head!(l :: Model.Dense, _, trunk_outlen)
 l.w.data = l.w.data[:,1:trunk_outlen]
end
