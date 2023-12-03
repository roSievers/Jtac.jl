
"""
    Shallow(G; kwargs...)

Create a shallow [`NeuralModel`](@ref) for games of type `G`.

The remaining arguments `kwargs` are passed to the constructor of
[`NeuralModel`](@ref).
"""
function Shallow(G :: Type{<: AbstractGame}; kwargs...)
  NeuralModel(G, Model.Pointwise(); kwargs...)
end


"""
    MLP(G, f; widths = [64], kwargs...)

Create a multilayer-perceptron [`NeuralModel`](@ref) for games of type `G`.
The widths of the hidden layers are specified by the iterable `widths`. The
activation function is determined by `f`.

The remaining arguments `kwargs` are passed to the constructor of
[`NeuralModel`](@ref).
"""
function MLP(G :: Type{<: AbstractGame}, f = "relu"; widths = [64], kwargs...)
  widths = [ prod(size(G)), widths...]
  layers = [ Model.Dense(widths[j], widths[j+1], f) for j in 1:length(widths) - 1 ]

  NeuralModel(G, Model.Chain(layers); kwargs...)
end


"""
    ShallowConv(G, f; filters = 64, kwargs...)

Create a shallow convolutional [`NeuralModel`](@ref) for games of type `G`.
The number of convolutional filters is specified by `filters`. The activation
function is determined by `f`.

The remaining arguments `kwargs` are passed to the constructor of
[`NeuralModel`](@ref).
"""
function ShallowConv( G :: Type{<: AbstractGame}
                    , f = "relu"
                    ; filters = 64
                    , kwargs... )

  NeuralModel(G, Model.Conv(size(G)[3], filters, f); kwargs...)
end


# -------- Architecture used for Alpha Zero ---------------------------------- #

"""
    zero_res_block(G, filters)

Create one residual Alpha Zero block (`conv -> batchnorm -> conv -> batchnorm`)
that suits game type `G` and has `filters` many filters.
"""
function zero_res_block(G :: Type{<: AbstractGame}, filters :: Int)
  Model.@residual (size(G)[1:2]..., filters) :relu begin
    Conv(filters, window = 3, pad = 1, stride = 1)
    Batchnorm(:relu)
    Conv(filters, window = 3, pad = 1, stride = 1)
    Batchnorm()
  end
end

"""
    zero_conv_block(G, filters)

Create one convolutional Alpha Zero block (`conv -> batchnorm`) that suits game
type `G` and has `filters` many filters.
"""
function zero_conv_block(G :: Type{<: AbstractGame}, ci :: Int, co :: Int)
  Model.@chain (size(G)[1:2]..., ci) begin
    Conv(co, window = 3, pad = 1, stride = 1)
    Batchnorm("relu")
  end
end

"""
    zero_vhead(G, filters)

Return the value head of the Alpha Zero architecture for games of type `G` with
`filters` filters.
"""
function zero_vhead(G :: Type{<: AbstractGame}, filters)
  shape = (size(G)[1:2]..., filters)
  Model.@chain shape begin
    Conv(32, window = 1, pad = 0, stride = 1)
    Batchnorm("relu")
    Dense(256, "relu")
    Dense(1)
  end
end

"""
    zero_phead(G, filters)

Return the policy head of the Alpha Zero architecture for games of type `G` with
`filters` filters.
"""
function zero_phead(G :: Type{<: AbstractGame}, filters)
  shape = (size(G)[1:2]..., filters)
  Model.@chain shape begin
    Conv(32, window = 1, pad = 0, stride = 1)
    Batchnorm("relu")
    Dense(Game.policylength(G))
  end
end

"""
    zero_head(G, target, filters)

Return a head that suits a given `target` (see [`Target.AbstractTarget`](@ref)),
modeled modeled after the Alpha Zero policy head for games of type `G` with
`filters` filters.
"""
function zero_head(G :: Type{<: AbstractGame}, target, filters)
  shape = (size(G)[1:2]..., filters)
  Model.@chain shape begin
    Conv(32, window = 1, pad = 0, stride = 1)
    Batchnorm("relu")
    Dense(length(target))
  end
end


"""
    ZeroConv(G; blocks, filters, targets, kwargs...)  

Create a [`NeuralModel`](@ref) for game type `G` in the convolutional Alpha
Zero architecture. The number of blocks and filters is specified by the
respective keyword arguments `blocks` and `filters`.

Additional targets can be assigned via `targets` (see
[`Target.AbstractTarget`](@ref)). The function [`zero_head`](@ref) is used to
determine suitable target heads.
"""
function ZeroConv( G :: Type{<: AbstractGame}
                 ; blocks = 6
                 , filters = 256
                 , targets = (;)
                 , kwargs... )
  @assert blocks >= 1

  heads = (;
    value = zero_vhead(G, filters),
    policy = zero_phead(G, filters),
    map(target -> zero_head(G, target, filters), targets)...,
  )
  ci = size(G)[3]
  layers = [
    zero_conv_block(G, ci, filters);
    [zero_conv_block(G, filters, filters) for _ in 1:(blocks-1)]
  ]
  trunk = Model.Chain(layers)
  NeuralModel(G, trunk; heads, targets, kwargs...)
end


"""
    ZeroRes(G; blocks, filters, targets, kwargs...)  

Create a [`NeuralModel`](@ref) for game type `G` in the residual Alpha Zero
architecture. The number of blocks and filters is specified by the respective
keyword arguments `blocks` and `filters`.

Additional targets can be assigned via `targets` (see
[`Target.AbstractTarget`](@ref)). The function [`zero_head`](@ref) is used to
determine suitable target heads.
"""
function ZeroRes( G :: Type{<: AbstractGame}
                ; blocks = 6
                , filters = 256
                , targets = (;)
                , kwargs... )
  @assert blocks >= 1
  heads = (;
    value = zero_vhead(G, filters),
    policy = zero_phead(G, filters),
    map(target -> zero_head(G, target, filters), targets)...,
  )
  ci = size(G)[3]
  layers = [
    zero_conv_block(G, ci, filters);
    [zero_res_block(G, filters) for _ in 1:(blocks-1)]
  ]
  trunk = Model.Chain(layers)
  NeuralModel(G, trunk; heads, targets, kwargs...)
end

