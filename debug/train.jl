
using Jtac

function train!( model;
                 power = 50,
                 branch_prob = 0.05,
                 temperature = 1,
                 augment = true,   # whether to use symmetry augmentation on the recorded games
                 epochs = 10,      # number of times a new dataset is produced (with constant network)
                 batchsize = 200,  # number of states per gd-step
                 selfplays = 50    # number of selfplays to record the dataset
                                   # note that branching and augmentation may lead to a higher number of recorded games
              )

  println("Begin training")

  @show power
  @show epochs
  @show selfplays
  @show batchsize
  @show branch_prob
  @show temperature
  @show augment

  p1 = MCTPlayer(model, power = power)
  p2 = MCTPlayer(copy(model), power = power)
  p3 = MCTPlayer(RolloutModel(), power = 500, temperature = 0.5)

  println("Before training of p1 (from perspective of p1):")

  println("p1 vs. p2: $(asyncmap(_ -> pvp(p1, p2), 1:50) |> sum)")
  println("p2 vs. p1: $(asyncmap(_ -> pvp(p2, p1), 1:50) |> sum)")

  println("p1 vs. p3: $(asyncmap(_ -> pvp(p1, p3), 1:50) |> sum)")
  println("p3 vs. p1: $(asyncmap(_ -> pvp(p3, p1), 1:50) |> sum)")
  

  println()

  set_optimizer!(model, Knet.Adam, lr = 5e-3)


  for i in 1:epochs

    l = 0

    dataset = record_selfplay(model, selfplays, power = power, 
                              branch_prob = branch_prob, 
                              augment = augment, temperature = temperature)

#    @info "Dataset generation complete"
    batches = minibatch(dataset, batchsize, shuffle = true, partial = false)

    for batch in batches
      l += train_step!(training_model(model), batch)
#      @info "Single training step done"
    end
#    @info "All training steps done"


    println("i: $i, loss: $(l/batchsize/length(batches))")

    if i % 10 == 0 && i != epochs
      println("After $i training steps:")

      println("p1 vs. p2: $(asyncmap(_ -> pvp(p1, p2), 1:50) |> sum)")
      println("p2 vs. p1: $(asyncmap(_ -> pvp(p2, p1), 1:50) |> sum)")

      println("p1 vs. p3: $(asyncmap(_ -> pvp(p1, p3), 1:50) |> sum)")
      println("p3 vs. p1: $(asyncmap(_ -> pvp(p3, p1), 1:50) |> sum)")
    end
  end

  println("After full training:")

  println("p1 vs. p2: $(asyncmap(_ -> pvp(p1, p2), 1:50) |> sum)")
  println("p2 vs. p1: $(asyncmap(_ -> pvp(p2, p1), 1:50) |> sum)")

  println("p1 vs. p3: $(asyncmap(_ -> pvp(p1, p3), 1:50) |> sum)")
  println("p3 vs. p1: $(asyncmap(_ -> pvp(p3, p1), 1:50) |> sum)")

  model

end

# OBSERVATIONS:
# * The lower the power, the better the trained LinearModel becomes against Rollout
# * When the power during training is different to the power during playout, we suck
#
