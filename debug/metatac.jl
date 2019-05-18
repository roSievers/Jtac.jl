
include("train.jl")

chain = @chain( 
                MetaTac,
                Conv(512, relu, stride = 3, padding = 0),
                Conv(1024, relu),
                Dense(512, relu),
              )

model = NeuralModel(MetaTac, chain) |> to_gpu |> Async

train!( 
        model,
        power = 100,
        selfplays = 100,
        batchsize = 50,
        branch_prob = 0.02,
        epochs = 100
      )

