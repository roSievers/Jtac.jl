

using Jtac, cuDNN, Flux
using JtacPacoSako

import .Model: Zoo
import .Game: MetaTac
import .Player: IntuitionPlayer, MCTSPlayer, MCTSPlayerGumbel
import .Training: learn!, record

model = Zoo.ZeroConv(MetaTac, blocks = 8, filters = 128, backend = :cudaflux, async = 512)

@show Training.L2Reg()(model.model)

player = MCTSPlayerGumbel(model, power = 32, name = "gumbel")

# players = [
#   IntuitionPlayer(player, temperature = 0.25, name = "intu"),
#   MCTSPlayer(player, temperature = 0.25),
#   [MCTSPlayer(MetaTac, power = power, temperature = 0.25) for power in [64, 128, 256, 512, 1024]]...
# ]
# Player.rank(players, 1000, verbose = true, threads = true)

# weights = (value = 1.0, policy = 1.0)

opt = Flux.Momentum(5e-5)
learn!(model, generations = 1, epochs = 5, opt = opt, batchsize = 512, weights = weights) do gen
  record(player, 128, branch = Game.branch(prob = 0.5, steps = 1:10), threads = true)
end

# opt = Flux.OptimiserChain(Flux.WeightDecay(2f-5), Flux.Momentum(2e-3))
# learn!(model, generations = 25, epochs = 3, opt = opt, batchsize = 512, weights = weights) do gen
#   record(player, 64, branch = Game.branch(prob = 0.5, steps = 1:10), threads = true, anneal = n -> n <= 10)
# end

opt = Flux.OptimiserChain(Flux.WeightDecay(2f-5), Flux.Momentum(1e-3))
learn!(model, generations = 50, epochs = 5, opt = opt, batchsize = 512, weights = weights) do gen
  record(player, 64, branch = Game.branch(prob = 0.5, steps = 1:10), threads = true, anneal = n -> n <= 10)
end

# opt = Flux.OptimiserChain(Flux.WeightDecay(2f-5), Flux.Momentum(5e-4))
# learn!(model, generations = 25, epochs = 3, opt = opt, batchsize = 512, weights = weights) do gen
#   record(player, 64, branch = Game.branch(prob = 0.5, steps = 1:10), threads = true, anneal = n -> n <= 10)
# end

opt = Flux.OptimiserChain(Flux.WeightDecay(2f-5), Flux.Momentum(1e-4))
learn!(model, generations = 100, epochs = 3, opt = opt, batchsize = 512, weights = weights) do gen
  record(player, 64, branch = Game.branch(prob = 0.5, steps = 1:10), threads = true, anneal = n -> n <= 10)
end
