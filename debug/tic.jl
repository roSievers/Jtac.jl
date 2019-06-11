
include("train.jl")

chain = @stack TicTacToe stack_input=true begin
  Conv(1024, relu)
  Dense(512, relu)
end

model = NeuralModel(TicTacToe, chain) |> to_gpu |> Async

train!( 
        model,
        power = 500,
        selfplays = 100,
        batchsize = 100,
        branch_prob = 0.1,
        epochs = 100,
        regularization_weight = 1e-2,
        opponents = [MCTSPlayer(power = p) for p in [10, 25, 50, 100, 200, 300, 400, 500]],
        contest_interval = 1,
        contest_length = 1000
      )
