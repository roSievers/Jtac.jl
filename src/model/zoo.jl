
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

