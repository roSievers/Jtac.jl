using Revise
using Jtac

G = MetaTac

#chain = @chain G Conv(256) Pool() Deconv(512) Conv(512) Conv(512) Dense(82)
#chain = @chain G Dense(1000) Dense(1000) Dense(1000) Dense(1000) Dense(Jtac.policy_length(G)+1)


model = ShallowConv(G, 200)
#model = NeuralModel(G, chain)
amodel = Async(model, max_batchsize = 100)

games = map(1:10000) do _
  game = G()
  for i in 1:rand(1:20)
    if !Jtac.is_over(game)
      Jtac.random_turn!(game)
    end
  end
  game
end

println("CPU performance for 10000 games")
@time model(games);
@time map(model, games);
@time asyncmap(amodel, games, ntasks = Jtac.ntasks(amodel));

gmodel = model |> to_gpu
gamodel = Async(gmodel, max_batchsize = 100)

println()
println("GPU performance for 10000 games")
@time gmodel(games)
@time map(gmodel, games)
@time asyncmap(gamodel, games, ntasks = Jtac.ntasks(gamodel)) 
