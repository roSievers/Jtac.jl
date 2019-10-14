
using Jtac

G = MetaTac

chain = @chain G Conv(1024) Dense(512) Dense(512) Dense(128)
model = NeuralModel(G, chain) |> to_gpu |> Async

player = MCTSPlayer(model, power = 50)
