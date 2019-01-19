
using Jtac

function train(; model = ShallowConv(TicTacToe, 200),
                 power = 15,
                 branch_prob = 0.2,
                 temperature = 1
              )

  p1 = MCTPlayer(model, power = power)
  p2 = MCTPlayer(copy(model), power = power)
  p3 = MCTPlayer(RolloutModel(), power = 500, temperature = 0.5)
  #p4 = MCTPlayer(RolloutModel(), power = 1000, temperature = 0.5)
  #p5 = MCTPlayer(RolloutModel(), power = 1000, temperature = 0.1)

  println("Before training of p1 (from perspective of p1):")

  println("p1 vs. p2: $(sum(pvp(p1, p2) for i in 1:500))")
  println("p2 vs. p1: $(-sum(pvp(p2, p1) for i in 1:500))")

  println("p1 vs. p3: $(sum(pvp(p1, p3) for i in 1:500))")
  println("p3 vs. p1: $(-sum(pvp(p3, p1) for i in 1:500))")

  println()

  set_optimizer!(model, Knet.Adam, lr = 5e-3)

  l = 0

  for i in 1:1999

    dataset = record_selfplay(model, 5, power = power, 
                              branch_prob = branch_prob, 
                              augment = true, temperature = temperature)

    #bts = batches(dataset, 25)

    l += train_step!(model, dataset)

    if i % 10 == 0
      println("i: $i, loss: $(l/100*2)")
      l = 0
    end

    if i % 250 == 0
      println("After $i training steps:")

      println("p1 vs. p2: $(sum(pvp(p1, p2) for i in 1:500))")
      println("p2 vs. p1: $(-sum(pvp(p2, p1) for i in 1:500))")

      println("p1 vs. p3: $(sum(pvp(p1, p3) for i in 1:500))")
      println("p3 vs. p1: $(-sum(pvp(p3, p1) for i in 1:500))")
    end
  end

  println("After full training:")

  println("p1 vs. p2: $(sum(pvp(p1, p2) for i in 1:500))")
  println("p2 vs. p1: $(-sum(pvp(p2, p1) for i in 1:500))")

  println("p1 vs. p3: $(sum(pvp(p1, p3) for i in 1:500))")
  println("p3 vs. p1: $(-sum(pvp(p3, p1) for i in 1:500))")

  model

end

# OBSERVATIONS:
# * The lower the power, the better the trained LinearModel becomes against Rollout
# * When the power during training is different to the power during playout, we suck
#
