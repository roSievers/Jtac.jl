
include("train.jl")

#chain = @stack TicTacToe stack_input=true begin
#  Conv(1024, relu)
#  Dense(512, relu)
#end

chain = @stack TicTacToe stack_input=true begin
  Conv(128, relu)
  Dense(200, relu)
end

model = NeuralModel(TicTacToe, chain) |> to_gpu |> Async

opponents = [MCTSPlayer(power = p) for p in [10, 25, 50, 100, 200, 300, 400, 500]]

train!( model
      , power = 500
      , selfplays = 200
      , batchsize = 100
      , branching_prob = 0.1
      , epochs = 10
      , value_weight = 10.
      , regularization_weight = 1e-5
      , opponents = opponents
      , contest_interval = 3
      , contest_length = 5000
      , contest_cache = 5000 )
