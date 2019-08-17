
chain = @stack MNKGame{5, 5, 4} stack_input=true begin
  Conv(1024, relu)
  Dense(512, relu)
end

model = NeuralModel(MNKGame{5, 5, 4}, chain) |> to_gpu |> Async

mcts_opponents = [MCTSPlayer(power = p) for p in [10, 50, 100, 200, 500, 1000]]
self_opponents = [MCTSPlayer(model, power = p, name = "current$p") for p in [10, 50]]

train!( 
        model,
        power = 500,
        selfplays = 100,
        batchsize = 100,
        iterations = 10,
        branching_prob = 0.1,
        epochs = 100,
        regularization_weight = 1e-4,
        opponents = [mcts_opponents; self_opponents],
        contest_interval = 5,
        contest_length = 2000
      )
