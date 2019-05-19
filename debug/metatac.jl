
include("train.jl")

stack = @stack MetaTac stack_input=true begin
  Conv(512, relu, stride = 3, padding = 0)
  Conv(1024, relu, padding = 0)
  Dense(512, relu)
end

model = NeuralModel(MetaTac, stack) |> to_gpu |> Async

train!( 
        model,
        power = 500,
        selfplays = 50,
        batchsize = 100,
        branch_prob = 0.02,
        iterations = 10,
        epochs = 10,
        regularization_weight = 1e-3,
        learning_rate = 1e-4,
        opponents = [MCTSPlayer(power = p) for p in [10, 25, 50, 100, 200, 500]],
        contest_interval = 1
      )

