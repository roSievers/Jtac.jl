
using Jtac

chain = @stack MNKGame{5, 5, 4} stack_input=true begin
  Conv(256, relu)
  Dense(128, relu)
end

model = NeuralModel(MNKGame{5, 5, 4}, chain) |> to_gpu |> Async
player = MCTSPlayer(model, power = 25)

loss = Loss(value = 2.5, regularization = 1e-4)

mcts_opponents = [MCTSPlayer(power = p) for p in [10, 50, 100]]#, 200, 500, 1000]]
self_opponents = [MCTSPlayer(model, power = p, name = "current$p") for p in [10, 50]]

with_contest( train_self!
            , player
            , loss = loss
            , playings = 100
            , batchsize = 100
            , iterations = 10
            , branching = 0.1
            , epochs = 100
            , opponents = [mcts_opponents; self_opponents]
            , interval = 5
            , length = 2000 )
