
include("train.jl")

stack = @stack MetaTac stack_input=true begin
  Conv(1024, relu, stride = 3, padding = 0)
  Dense(512, relu)
end

model = NeuralModel(MetaTac, stack) |> to_gpu |> Async
player = MCTSPlayer(model, power = 250)

mcts_opponents = [MCTSPlayer(power = p) for p in [10, 50, 100, 200, 500]]
self_opponents = [MCTSPlayer(model, power = p, name = "current$p") for p in [10, 50]]

loss = Loss(policy = 0.5, regularization = 1e-5)

with_contest( train_self!
            , player
            , loss = loss
            , playings = 50
            , batchsize = 100
            , branching = 0.02
            , iterations = 10
            , epochs = 100
            , augment = true
            , opponents = [mcts_opponents; self_opponents]
            , length = 1000
            , interval = 5
            )

