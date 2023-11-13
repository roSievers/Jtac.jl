
using Revise
using Jtac, Flux, CUDA

G = Game.TicTacToe
model = Model.Zoo.ZeroConv(G, blocks = 2, filters = 64, backend = :cuflux)
player = Player.MCTSPlayerGumbel(Model.AsyncModel(model), power = 500)

ctx = Training.LossContext(model)
setup = Training.setup(model, ctx, Adam(0.001))

for epoch in 1:10
  @show Model.apply(model, G())
  ds = Training.record(player, 50, branch = Game.branch(prob = 0.25))
  Training.learn!(model, ds, ctx, setup, epochs = 1)
end

