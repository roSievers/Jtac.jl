
using Jtac

chain = @chain MetaTac begin
  Conv(128, relu, padding = 1)
  Batchnorm()
#  Conv(128, relu, padding = 1)
#  Batchnorm()
#  Conv(128, relu, padding = 1)
#  Batchnorm()
#  Conv(128, relu, padding = 1)
#  Batchnorm()
#  Conv(128, relu, padding = 1) 
#  Batchnorm()
end

vhead = @chain (9, 9, 128) Conv(1, relu, window = 1) Batchnorm() Dense(64, relu) Dense(1)
phead = @chain (9, 9, 128) Conv(2, relu, window = 1) Batchnorm() Dense(81)

#chain = @chain MetaTac Dense(50) Dense(50)
vhead = nothing
phead = nothing

model = NeuralModel(MetaTac, chain, vhead = vhead, phead = phead)
model = model |> to_gpu |> Async

player = MCTSPlayer(model, power = 50, dilution = 0.075)

function train_meta_conv(player)

  mcts_opponents = [MCTSPlayer(power = p, temperature = 0.5) for p in [100, 250, 500, 750]]
  self_opponents = [MCTSPlayer(model, power = p, name = "current$p") for p in [25, 50]]

  loss = Loss(reg = 1e-4)

  i = 0
  for (lr, epochs) in [(2e-2, 20), (1e-3, 40), (2e-4, 60)]
    i += 1
    @time with_contest( train_self!
                , player
                , loss = loss
                , playings = 516
                , batchsize = 1024
                , branching = (before = 0.5, during = 0.0, steps = 1)
                , iterations = 1
                , replays = 2
                , epochs = epochs
                , augment = true
                , opponents = [mcts_opponents; self_opponents]
                , length = 0
                , cache = 2000
                , interval = 10
                , distributed = true
                , optimizer = Knet.Momentum
                , lr = lr
                , gamma = 0.9
                )
    save_model("meta-conv-$i", training_model(player) |> to_cpu)
  end
end
