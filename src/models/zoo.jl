
# -------- Linear Neural Model ----------------------------------------------- #

function Shallow(G :: Type{<: AbstractGame}; kwargs...)
  NeuralModel(G, Pointwise(); kwargs...)
end


# -------- Multilayer Perceptron --------------------------------------------- #

function MLP(G :: Type{<: AbstractGame}, hidden; f = Knet.relu, kwargs...)
  widths = [ prod(size(G)), hidden...]
  layers = [ Dense(widths[j], widths[j+1], f) for j in 1:length(widths) - 1 ]

  NeuralModel(G, Chain(layers...); kwargs...)
end


# -------- Shallow Convolutional Network ------------------------------------ #

function ShallowConv( G :: Type{<: AbstractGame}
                    , channels
                    ; f = Knet.relu
                    , kwargs... )

  NeuralModel(G, Conv(size(G)[3], filters, f); kwargs...)
end


# -------- Architecture used for Alpha Zero ---------------------------------- #

function zero_res_block(channels :: Int)
  Model.@residual (8, 8, channels) f = relu begin
    Conv(channels, f = relu, window = 3, padding = 1, stride = 1)
    Batchnorm()
    Conv(channels, f = relu, window = 3, padding = 1, stride = 1)
    Batchnorm()
  end
end

function zero_conv_block(ci :: Int, co :: Int; window = 3, padding = 1, stride = 1)
  Model.@chain (8, 8, ci) begin
    Conv(co, f = relu, window = 3, padding = 1, stride = 1)
    Batchnorm()
  end
end

function zero_vhead(G :: Type{<: AbstractGame}, channels)
  shape = (size(G)[1:2]..., channels)
  Model.@chain shape begin
    Conv(1, f = relu, window = 1, padding = 0, stride = 1)
    Batchnorm()
    Dense(256, f = relu)
    Dense(1, f = relu)
  end
end

function zero_phead(G :: Type{<: AbstractGame}, channels)
  shape = (size(G)[1:2]..., channels)
  Model.@chain shape begin
    Conv(2, f = relu, window = 1, padding = 0, stride = 1)
    Batchnorm()
    Dense(Game.policy_length(G), f = relu)
  end
end

function ZeroConv( G :: Type{<: AbstractGame}
                 ; blocks = 6, channels = 256
                 , vhead = zero_vhead(G, channels)
                 , phead = zero_phead(G, channels) )
  @assert blocks >= 1
  ci = size(G)[3]
  trunk = Model.Chain( zero_conv_block(ci, channels)
                     , [zero_conv_block(channels, channels) for i in 1:(blocks-1)]...)
  NeuralModel(G, trunk; vhead = vhead, phead = phead)
end

function ZeroRes( G :: Type{<: AbstractGame}
                ; blocks = 6, channels = 256
                , vhead = zero_vhead(G, channels)
                , phead = zero_phead(G, channels) )
  @assert blocks >= 1
  ci = size(G)[3]
  trunk = Model.Chain( zero_conv_block(ci, channels)
                     , [zero_res_block(channels) for i in 1:(blocks-1)]...)
  NeuralModel(G, trunk; vhead = vhead, phead = phead)
end

