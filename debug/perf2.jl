using Revise
using Jtac

G = MetaTac
bp = 0.02

chain = @chain G Conv(256) Pool() Dropout(0.2) Deconv(512) Conv(512) Dropout(0.2) Conv(256) Dense(82)
#chain = @chain G Dense(1000) Dense(1000) Dense(1000) Dense(1000) Dense(Jtac.policy_length(G)+1)


#model = ShallowConv(G, 500)
model = BaseModel(G, chain)

amodel = Async(model, max_batchsize = 100)

games = [G() for i in 1:100]

println("CPU performance for 1000 games")
@time model(games);
@time map(model, games);
@time asyncmap(amodel, games);

@time record_selfplay(model, branch_prob = bp, power = 100)
#@time record_selfplay(amodel, 20, branch_prob = bp)

gmodel = model |>  to_gpu
gamodel = Async(gmodel, max_batchsize = 100)

games = [MetaTac() for i in 1:1000]
#games = map(1:100) do _
#  game = G()
#  for i in 1:rand(1:20)
#    if !Jtac.is_over(game)
#      Jtac.random_turn!(game)
#    end
#  end
#  game
#end

println()
println("GPU performance for 1000 games")
@time gmodel(games)
@time map(gmodel, games)
@time asyncmap(gamodel, games) 

@time record_selfplay(gmodel, branch_prob = bp, power = 100)
@time record_selfplay(gamodel, 20, branch_prob = bp, power = 100)

