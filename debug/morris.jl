
using Jtac

chain = @stack Morris stack_input=true begin
  Conv(128, relu)
  Dense(200, relu)
end

model = NeuralModel(Morris, chain)
player = MCTSPlayer(model, power = 100)

loss = Loss(value = 10, regularization = 1e-5)

opponents = [MCTSPlayer(power = p) for p in [10, 25, 50, 100, 250, 500]]

with_contest( train_self!
            , player
            , loss = loss
            , playings = 500
            , batchsize = 100
            , branching = 0.1
            , epochs = 10
            , opponents = opponents
            , interval = 3
            , length = 1000
            , cache = 2000 )
