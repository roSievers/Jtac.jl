
using Jtac
using Printf

function train!( model;
                 power = 100,
                 branch_prob = 0.,
                 temperature = 1.,
                 opponents = [],
                 contest_temperature = 1.,
                 contest_length :: Int = 250,
                 contest_interval :: Int = 10,
                 optimizer = Knet.Adam,
                 value_weight = 1.,
                 policy_weight = 1.,
                 regularization_weight = 0.,
                 iterations = 1,   # how often we cycle through the generated training set
                 augment = true,   # whether to use symmetry augmentation on the recorded games
                 epochs = 10,      # number of times a new dataset is produced (with constant network)
                 batchsize = 200,  # number of states per gd-step
                 selfplays = 50,   # number of selfplays to record the dataset
                 kwargs...         # arguments for the optimizer
              )

  println("Training options:")
  println()
  @show epochs
  @show selfplays
  @show batchsize
  @show branch_prob
  @show iterations
  @show power
  @show augment
  @show optimizer
  @show value_weight
  @show policy_weight
  @show regularization_weight
  @show temperature
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

  set_optimizer!(model, optimizer; kwargs...)


  for i in 1:epochs

    dataset = record_selfplay(model, selfplays, power = power, 
                              branch_prob = branch_prob, augment = augment, 
                              temperature = temperature)

    for j in 1:iterations

      batches = minibatch(dataset, batchsize, shuffle = true, partial = false)

      for batch in batches
        train_step!( training_model(model)
                   , batch
                   , value_weight = value_weight
                   , policy_weight = policy_weight
                   , regularization_weight = regularization_weight
                   )
      end

    end

    # Calculate loss for this epoch
    l = loss_components(model, dataset)
    loss = value_weight * l.value + 
           policy_weight * l.policy + 
           regularization_weight * l.regularization


    @printf( "%d: %6.3f %6.3f %6.3f %6.3f %d\n"
           , i
           , loss
           , l.value * value_weight
           , l.policy * policy_weight
           , l.regularization * regularization_weight
           , length(dataset)
           )


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

