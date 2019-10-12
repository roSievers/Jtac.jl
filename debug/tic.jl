
using Jtac

chain = @stack TicTacToe stack_input=true begin
  Conv(128, relu)
  Dense(200, relu)
end

model = NeuralModel(TicTacToe, chain) |> to_gpu |> Async
player = MCTSPlayer(model, power = 100)

loss = Loss(value = 10, regularization = 1e-5)

opponents = [MCTSPlayer(power = p) for p in [10, 25, 50, 100, 200, 300, 400, 500]]

with_contest( train_self!
            , player
            , loss = loss
            , playings = 200
            , batchsize = 100
            , branching = 0.1
            , epochs = 10
            , opponents = opponents
            , interval = 3
            , length = 5000
            , cache = 5000 )
