
using Jtac

function train!( model;
                 power = 100,
                 branch_prob = 0.,
                 selfplay_temperature = 1.,
                 contest_temperature = 1.,
                 contest_length :: Int = 250,
                 contest_interval :: Int = 10,
                 optimizer = Knet.Adam,
                 learning_rate = 1e-3,
                 iterations = 1,
                 regularization_weight = 0.,
                 opponents = [],
                 augment = true,   # whether to use symmetry augmentation on the recorded games
                 epochs = 10,      # number of times a new dataset is produced (with constant network)
                 batchsize = 200,  # number of states per gd-step
                 selfplays = 50,   # number of selfplays to record the dataset
                                   # note that branching and augmentation may lead to a higher number of recorded games
              )

  println("Training options:")
  println()
  @show epochs
  @show selfplays
  @show batchsize
  @show branch_prob
  @show power
  @show augment
  @show optimizer
  @show learning_rate
  @show selfplay_temperature
  @show contest_length
  @show contest_temperature
  @show contest_interval
  println("opponents = ", name.(opponents))

  async = isa(model, Async)

  if contest_length > 0
    players = [
      IntuitionPlayer(model, temperature = contest_temperature, name = "current");
      IntuitionPlayer(copy(model), temperature = contest_temperature, name = "initial");
      opponents
    ]

    println()
    println("First contest begins...")
    print_ranking(players, contest_length, async = async)
  end

  println()
  println("Training begins...")

  set_optimizer!(model, optimizer, lr = learning_rate)


  for i in 1:epochs

    l = 0

    dataset = record_selfplay(model, selfplays, power = power, 
                              branch_prob = branch_prob, augment = augment, 
                              temperature = selfplay_temperature)

    for j in 1:iterations

      batches = minibatch(dataset, batchsize, shuffle = true, partial = false)

      for batch in batches
        l += train_step!(training_model(model), batch, 
                         regularization_weight = regularization_weight)
      end

    end

    println("i: $i, loss: $(l/length(dataset)/iterations)")

    if i % contest_interval == 0 && i != epochs && contest_length > 0
      println()
      println("Intermediate contest begins...")
      print_ranking(players, contest_length, async = async)
      println()
    end
  end

  if contest_length > 0
    println()
    println("Final contest begins...")
    print_ranking(players, contest_length, async = async)
  end

  model

end

