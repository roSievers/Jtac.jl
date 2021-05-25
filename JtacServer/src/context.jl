
"""
Context in which the training of a player takes place. Contains information like
player parameters (power, temperature, ...) or hyperparameters of the training
process (learning rate, batchsize, ...).
"""
mutable struct Context

  id :: Int

  # Player options for dataset generation
  name :: String
  power :: Int
  temperature :: Float32
  exploration :: Float32
  dilution :: Float32

  # Other options for dataset generation
  init_steps :: Tuple{Int, Int}
  branch :: Float64 
  branch_steps :: Tuple{Int, Int}
  augment :: Bool

  # Minimal or maximal playings required from data packages sent by jtac serve
  # instances
  min_playings :: Int
  max_playings :: Int

  # Selecting training data from the DataPool
  # Epoch: subset of the pool used for one training iteration
  epoch_size :: Int
  iterations :: Int
  test_frac :: Float64 # fraction of data to go to test set 
  max_age :: Int
  max_use :: Int

  # Data is only used when the quality of the datapool is above this threshold
  min_quality :: Float64 

  # Number between 0 and 1 to weight age vs. usage to assess the quality of
  # a dataset (convex combination)
  age_weight :: Float64

  # Maximal number of game states stored in the test and trainpools
  capacity :: Int

  # Era: games used for training until the reference model gets its next update
  era_size :: Int

  # Options for training after an epoch has been selected from the DataPool
  batch_size :: Int
  learning_rate :: Float32
  momentum :: Float32
  loss_weights :: Vector{Float32}

  # number of backups
  backups :: Int

  # Additional meta-information for purposes of documentation
  msg :: String

end

function Context( id :: Int
                ; name = "default"
                , power = 50
                , temperature = 1.
                , exploration = 1.41
                , dilution = 0.
                , init_steps = (0, 0)
                , branch = 0.
                , branch_steps = (0, 0)
                , augment = true
                , min_playings = 1
                , max_playings = 1000
                , epoch_size = 5000
                , iterations = 1
                , test_frac = 0.1
                , max_age = 3
                , max_use = 3
                , min_quality = 0.
                , age_weight = 0.5
                , capacity = 10^6
                , era_size = 20000
                , batch_size = 512
                , learning_rate = 1e-2
                , momentum = 0.9
                , loss_weights = [1., 1., 0.]
                , backups = 2
                , msg = "" )

  Context( id, name, power, temperature, exploration, dilution,
           init_steps, branch, branch_steps, augment,
           min_playings, max_playings,
           epoch_size, iterations, test_frac, max_age, max_use,
           min_quality, age_weight,
           capacity, era_size, batch_size, learning_rate, momentum, loss_weights,
           backups, msg )
end

