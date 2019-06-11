
include("train.jl")

stack = @stack MetaTac stack_input=true begin
  Conv(1024, relu, stride = 3, padding = 0)
  Dense(512, relu)
end

backup_model = NeuralModel(MetaTac, stack) 
model = backup_model |> copy |> to_gpu |> Async

mcts_opponents = [MCTSPlayer(power = p) for p in [10, 50, 100, 200, 500]]
self_opponents = [MCTSPlayer(model, power = p, name = "current$p") for p in [10, 50]]


train!( 
        model,
        power = 500,
        selfplays = 50,
        batchsize = 100,
        branch_prob = 0.02,
        iterations = 10,
        epochs = 100,
        augment = true,
        regularization_weight = 2e-8,
        opponents = [mcts_opponents; self_opponents],
        contest_length = 1000,
        contest_interval = 5
      )

