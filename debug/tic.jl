
include("train.jl")

chain = @chain TicTacToe begin
  Conv(512, relu)
  Batchnorm()
end

model = NeuralModel(TicTacToe, chain) |> to_gpu |> Async

train!( 
        model,
        power = 250,
        selfplays = 100,
        batchsize = 100,
        branch_prob = 0.1,
        epochs = 10,
        regularization_weight = 1e-2,
        opponents = [MCTSPlayer(power = p) for p in [10, 25, 50, 100, 200, 300, 400, 500]],
        contest_interval = 1,
        contest_length = 1000
      )
