
using Jtac
import Random, Printf
import ProgressMeter
import Crayons


#
# TODO:
#   * Progress meter async-friendly
#   * pre-play contest games that do not depend on trainmodel(model)
#   * modularization and documentation
#   * progress bar for the contest
#

# Auxiliary functions

const gray_cr = Crayons.Crayon(foreground = :dark_gray)

function progressmeter( n, desc
                      ; dt = 0.5
                      , kwargs... )

  glyphs = ProgressMeter.BarGlyphs("[=>⋅]")
  ProgressMeter.Progress( n
                        , dt = dt
                        , desc = desc
                        , barglyphs = glyphs
                        , kwargs... )

end

progress! = ProgressMeter.next!

clear_output!(p) = ProgressMeter.printover(p.output, "")

function print_loss( epoch
                   , loss
                   , value_loss
                   , policy_loss
                   , regularization_loss
                   , setsize
                   , crayon = "" )

  str = Printf.@sprintf( "%d %6.3f %6.3f %6.3f %6.3f %d"
                       , epoch
                       , loss
                       , value_loss
                       , policy_loss
                       , regularization_loss
                       , setsize )

  println(crayon, str)

end

format_option(s, v) = Printf.@sprintf "# %-22s %s\n" string(s, ":") v

function print_contest_results(players, contest_length, async)
    println(gray_cr, "\n# CONTEST")
    print_ranking(players, contest_length, prepend = "#", async = async)
    println()
end

# Main training function

function train!( model
               ; power = 100
               , branch_prob = 0.
               , temperature = 1.
               , opponents = []
               , no_contests = false
               , contest_temperature = 1.
               , contest_length :: Int = 250
               , contest_interval :: Int = 10
               , regularization_weight = 0.
               , testset_fraction = 0.1
               , policy_weight = 1.
               , value_weight = 1.
               , iterations = 1   # how often we cycle through the generated training set
               , augment = true   # whether to use symmetry augmentation on the recorded games
               , epochs = 10      # number of times a new dataset is produced (with constant network)
               , batchsize = 200  # number of states per gd-step
               , selfplays = 50   # number of selfplays to record the dataset
               , optimizer = Knet.Adam
               , kwargs...        # arguments for the optimizer
               )

  print( gray_cr, "\n"
       , "# OPTIONS\n"
       , format_option(:epochs, epochs)
       , format_option(:selfplays, selfplays)
       , format_option(:batchsize, batchsize)
       , format_option(:iterations, iterations)
       , format_option(:power, power)
       , format_option(:augment, augment)
       , format_option(:optimizer, optimizer)
       , format_option(:value_weight, value_weight)
       , format_option(:policy_weight, policy_weight)
       , format_option(:regularization_weight, regularization_weight)
       , format_option(:temperature, temperature)
       , format_option(:testset_fraction, testset_fraction)
       , format_option(:no_contests, no_contests)
       , format_option(:contest_length, contest_length)
       , format_option(:contest_temperature, contest_temperature)
       , format_option(:contest_interval, contest_interval) 
       , format_option(:opponents, join(name.(opponents), " "))
       , "\n" )

  async = isa(model, Async)

  no_contests |= contest_length <= 0

  if !no_contests
    players = [
      IntuitionPlayer(model, temperature = contest_temperature, name = "current");
      IntuitionPlayer(copy(model), temperature = contest_temperature, name = "initial");
      opponents
    ]

    print_contest_results(players, contest_length, async)
  end

  set_optimizer!(model, optimizer; kwargs...)

  println("# TRAINING")

  for i in 1:epochs

    # Data generation via selfplays

    p = progressmeter( selfplays + 1, "# Selfplays...")

    dataset = record_selfplay( model, selfplays
                             , power = power
                             , branch_prob = branch_prob
                             , augment = augment
                             , temperature = temperature
                             , callback = () -> progress!(p) )

    clear_output!(p)

    testlength = round(Int, testset_fraction * length(dataset))
    testset, trainset = split(dataset, testlength, shuffle = true)

    steps = iterations * div(length(trainset), batchsize)

    p = progressmeter( steps + 1, "# Training...")

    for j in 1:iterations

      batches = minibatch(trainset, batchsize, shuffle = true, partial = false)

      for batch in batches
        train_step!( training_model(model)
                   , batch
                   , value_weight = value_weight
                   , policy_weight = policy_weight
                   , regularization_weight = regularization_weight
                   )
        progress!(p)
      end

    end

    clear_output!(p)

    # Calculate loss for this epoch
    for (set, crayon)  in [(testset, ""), (trainset, gray_cr)]
      l = loss_components(model, set)
      loss = value_weight * l.value + 
             policy_weight * l.policy + 
             regularization_weight * l.regularization

      print_loss( i
                , loss
                , l.value * value_weight
                , l.policy * policy_weight
                , l.regularization * regularization_weight
                , length(set)
                , crayon )

    end

    if (i % contest_interval == 0 || i == epochs) && !no_contests
      print_contest_results(players, contest_length, async)
    end
  end

  model

end

