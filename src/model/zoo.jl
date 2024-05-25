
"""
    MLP(G, f; widths = [64], kwargs...)

Create a multilayer-perceptron [`NeuralModel`](@ref) for games of type `G`.
The widths of the hidden layers are specified by the iterable `widths`. The
activation function is determined by `f`.

The remaining arguments `kwargs` are passed to the constructor of
[`NeuralModel`](@ref).
"""
function MLP( :: Type{G}
            , f = :relu
            ; widths = [64]
            , tensorizor = DefaultTensorizor{G}()
            , kwargs... ) where {G <: AbstractGame}

  sz = size(tensorizor)
  widths = [prod(sz), widths...]
  layers = [Dense(widths[j], widths[j+1], f) for j in 1:length(widths) - 1]
  NeuralModel(G, Chain(layers); tensorizor, kwargs...)
end


"""
    ShallowConv(G, f; filters = 64, kwargs...)

Create a shallow convolutional [`NeuralModel`](@ref) for games of type `G`.
The number of convolutional filters is specified by `filters`. The activation
function is determined by `f`.

The remaining arguments `kwargs` are passed to the constructor of
[`NeuralModel`](@ref).
"""
function ShallowConv( :: Type{G}
                    , f = :relu
                    ; filters = 64
                    , tensorizor = DefaultTensorizor{G}()
                    , kwargs... ) where {G <: AbstractGame}
  sz = size(tensorizor)
  NeuralModel(G, Model.Conv(sz[3], filters, f); tensorizor, kwargs...)
end


# -------- Architecture used for Alpha Zero ---------------------------------- #

"""
    zero_res_block(sz, filters)

Create one residual Alpha Zero block (`conv -> batchnorm -> conv -> batchnorm`)
that suits tensorizors with size `sz` and has `filters` many filters.
"""
function zero_res_block(sz, filters :: Int)
  Model.@residual (sz[1:2]..., filters) :relu begin
    Conv(filters, window = 3, pad = 1, stride = 1)
    Batchnorm(:relu)
    Conv(filters, window = 3, pad = 1, stride = 1)
    Batchnorm()
  end
end

"""
    zero_conv_block(sz, filters)

Create one convolutional Alpha Zero block (`conv -> batchnorm`) that suits 
tensorizors with size `sz` and has `filters` many filters.
"""
function zero_conv_block(sz, ci :: Int, co :: Int)
  Model.@chain (sz[1:2]..., ci) begin
    Conv(co, window = 3, pad = 1, stride = 1)
    Batchnorm(:relu)
  end
end

"""
    zero_vhead(sz, filters)

Return the value head of the Alpha Zero architecture for inputs with size
`sz` and `filters` filters.
"""
function zero_vhead(sz, filters)
  shape = (sz[1:2]..., filters)
  Model.@chain shape begin
    Conv(32, window = 1, pad = 0, stride = 1)
    Batchnorm(:relu)
    Dense(256, :relu)
    Dense(1)
  end
end

"""
    zero_phead(sz, pl, filters)

Return the policy head of the Alpha Zero architecture for inputs with size
`sz` and `filters` filters. The argument `pl` denotes the policy length for the respective game type.
"""
function zero_phead(sz, pl, filters)
  shape = (sz[1:2]..., filters)
  Model.@chain shape begin
    Conv(32, window = 1, pad = 0, stride = 1)
    Batchnorm("relu")
    Dense(pl)
  end
end

"""
    zero_head(sz, target, filters)

Return a head that suits a given `target` (see [`Target.AbstractTarget`](@ref)),
modeled modeled after the Alpha Zero policy head for inputs with size `sz`
and `filters` filters.
"""
function zero_head(sz, target, filters)
  shape = (sz[1:2]..., filters)
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
function ZeroConv( :: Type{G}
                 ; blocks = 6
                 , filters = 256
                 , targets = (;)
                 , tensorizor = DefaultTensorizor{G}()
                 , kwargs... ) where {G <: AbstractGame}
  @assert blocks >= 1

  sz = size(tensorizor)
  pl = policylength(G)

  heads = (;
    value = zero_vhead(sz, filters),
    policy = zero_phead(sz, pl, filters),
    map(target -> zero_head(sz, target, filters), targets)...,
  )
  ci = sz[3]
  layers = [
    zero_conv_block(sz, ci, filters);
    [zero_conv_block(sz, filters, filters) for _ in 1:(blocks-1)]
  ]
  trunk = Model.Chain(layers)
  NeuralModel(G, trunk; heads, targets, tensorizor, kwargs...)
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
                , tensorizor = DefaultTensorizor{G}()
                , kwargs... )
  @assert blocks >= 1

  sz = size(tensorizor)
  pl = policylength(G)
  
  heads = (;
    value = zero_vhead(sz, filters),
    policy = zero_phead(sz, pl, filters),
    map(target -> zero_head(sz, target, filters), targets)...,
  )
  ci = sz[3]
  layers = [
    zero_conv_block(sz, ci, filters);
    [zero_res_block(sz, filters) for _ in 1:(blocks-1)]
  ]
  trunk = Model.Chain(layers)
  NeuralModel(G, trunk; heads, targets, tensorizor, kwargs...)
end

