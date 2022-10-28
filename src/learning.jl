
# -------- Auxiliary Functions ----------------------------------------------- #

# Set an optimizer for all parameters of a model
function set_optimizer!(model, opt = Knet.SGD; kwargs...)

  for param in Knet.params(model)

    # Feature heads that are not used have length 0. They do not get an
    # optimizer.
    if length(param) > 0

      # If opt is given, overwrite every optimizer in param.opt
      if !isnothing(opt)

        param.opt = opt(; kwargs...)

      # If opt is not given, overwrite only non-existing param.opt with SGD
      elseif isnothing(opt) && isnothing(param.opt)

        param.opt = Knet.SGD(; kwargs...)

      end
    end
  end
end

# A single training step, the loss is returned
function train_step!(l :: Loss, model, cache :: Cache)

  tape = Knet.@diff sum(loss(l, model, cache))

  for param in Knet.params(model)
    Knet.update!(Knet.value(param), Knet.grad(tape, param), param.opt)
  end

  Knet.value(tape)
end

# -------- Logging Losses while Training ------------------------------------- #

function print_loss_header(loss, use_features = true)
  names = use_features ? loss_names(loss) : loss_names(loss)[1:3]
  components = map(names) do c
    Printf.@sprintf("%10s", string(c)[1:min(end, 10)])
  end
  println(join(["#"; "   epoch"; components; "     total"; "    length"], " "))
end

function print_loss(l, p, epoch, train, test = nothing)
  if isnothing(test)
    ds = [(train, :normal)]
  else
    ds = [(train, 245), (test, :normal)]
  end
  for (set, col) in ds
    # Compute the losses and get them as strings
    ls = loss(l, training_model(p), set)
    losses = map(x -> @sprintf("%10.3f", x), ls)

    # Print everything in grey (for train) and white (for test)
    str = @sprintf( "%10d %s %10.3f %10d\n"
                  , epoch , join(losses, " "), sum(ls), length(set) )
    printstyled(str, color = col)
  end
end

function print_ranking(rk)

  # Log that a contest comes next
  printstyled("#\n# Contest with $(length(rk.players)) players:\n#", color = 245)
  println()

  # Get the summary of the contest and print it
  str = "# " * replace(string(rk, true), "\n" => "\n# ") * "\n#"
  printstyled(str, color = 245)
  println()
end


# -------- Training on Datasets ---------------------------------------------- #

"""
    train!(model/player, dataset; <keyword arguments>)

Train `model`, or the training model of `player`, on `dataset`.

# Arguments
- `loss = Loss()`: Loss used for training.
- `epochs = 10`: Number of iterations through `dataset`.
- `batchsize = 50`: Batchsize for the update steps.
- `callback_epoch`: Function called after each epoch.
- `callback_step`: Function called after each update step.
- `optimizer = nothing`: Optimizer initialized for each weight in `model`
- `kwargs...`: Keyword arguments for `optimizer`

# Examples
```julia
G = Game.TicTacToe
dataset = Data.record(Player.MCTSPlayer(), 10, game = G) 
model = Model.NeuralModel(G, Model.@chain G Dense(50, "relu"))
loss = Training.Loss(value = 1., policy = 0.15, reg = 1e-4)
Training.train!(model, dataset, loss = loss, epochs = 15)
```
"""
function train!( player  :: Union{AbstractPlayer, AbstractModel}
               , trainset :: DataSet
               ; loss = Loss()
               , epochs = 10
               , batchsize = 50
               , callback_step = (_) -> nothing
               , callback_epoch = (_) -> nothing
               , optimizer = nothing
               , quiet = false
               , kwargs... )


  # Get basic info about the player's model
  model = training_model(player)
  gpu = on_gpu(model)

  # Check if features can be enabled
  use_features = feature_compatibility(loss, model, trainset)

  # Set/overwrite optimizers
  set_optimizer!(model, optimizer; kwargs...)

  # Print loss header if not quiet
  !quiet && print_loss_header(loss, use_features)


  # Generate the batch iterator (each iteration is shuffled independently)
  batches = Batches( trainset
                   , batchsize
                   , shuffle = true
                   , partial = true
                   , gpu = gpu )

  # Iterate through the trainset epochs times
  for j in 1:epochs

    if !quiet
      step, finish = stepper("# Learning...", length(batches))
    end

    for (i, cache) in enumerate(batches)

      train_step!(loss, model, cache)
      callback_step(i)
      !quiet && step()

    end

    if !quiet
      finish()
      print_loss(loss, player, j, trainset)
    end

    callback_epoch(j)
  end
end


# -------- Training by Playing ----------------------------------------------- #


function _train!( player :: AbstractPlayer{G}
                , gen_data :: Function
                ; loss = Loss()
                , epochs = 10
                , playings = 20
                , iterations = 10
                , batchsize = 50
                , testfrac = 0.1
                , optimizer = Knet.SGD
                , replays = 0
                , quiet = false
                , callback_epoch = (_) -> nothing
                , callback_iter = (_) -> nothing
                , kwargs...
                ) where {G <: AbstractGame}

  @assert playings > 0 "Number of playings for training must be positive"

  # If we do not print results, it is not necessary to test
  quiet || testfrac < 0 && (testfrac = 0.)

  # Get number of playings used for training
  train_playings = ceil(Int, playings * (1-testfrac))

  # Print the loss header if not quiet
  !quiet && print_loss_header(loss, feature_compatibility(loss, player))

  # Set the optimizer for the player's model
  set_optimizer!(training_model(player), optimizer; kwargs...)

  # Save the last `replay` generated datasets and use them for training
  replay_buffer = []

  for i in 1:epochs

    # Generate callback functions for the progress meter
    step, finish = stepper("# Playing...", playings)
    cb = quiet ? () -> nothing : step

    # Generate new datasets for this generation
    datasets = gen_data(cb, playings)

    # Create the training set by merging with the last `playing` generations of
    # datasets that are stored in the replay_buffer
    trainset = merge(datasets[1:train_playings]...)
    for set in replay_buffer
      trainset = merge(trainset, set...)
    end

    # Create the testing set by collecting the remaining datasets generated
    # in this epoche
    if testfrac > 0
      testset = merge(datasets[train_playings+1:playings]...)
    else
      testset = DataSet{G}() 
    end

    # Clear the progress bar (if it was printed)
    finish()

    # Prepare the next progress bar
    steps = iterations * ceil(Int, length(trainset) / batchsize)
    step, finish = stepper("# Learning...", steps)
    cb = (_) -> quiet ? nothing : step()

    # Train the player with the generated dataset
    train!( player
          , trainset
          , loss = loss
          , epochs = iterations
          , batchsize = batchsize
          , quiet = true
          , callback_step = cb
          , callback_epoch = callback_iter )

    # Clear the progress bar
    finish()

    # Calculate and print loss for this epoch if not quiet
    !quiet && print_loss(loss, player, i, trainset, testset)

    # Bring the full dataset from this epoche in the replay buffer
    push!(replay_buffer, datasets)

    # Remove the oldest item from the replay buffer if it is to large
    length(replay_buffer) >= replays+1 && deleteat!(replay_buffer, 1)

    # Call callback function
    callback_epoch(i)

  end 

end


"""
    train!(player; <keyword arguments>)

Train an MCTS `player` via playing against itself.

The training model of `player` learns to predict the MCTS policy, the value of
game states, and possible features (if `player` supports the same features as
the keyword argument `loss`). Note that only players with NeuralModel-based
training models can be trained currently.

# Arguments
- `loss = Loss()`: Loss used for training.
- `epochs = 10`: Number of epochs.
- `playings = 20`: Games played per `epoch` for training-set generation.
- `iterations = 10`: Number of training epochs per training-set.
- `batchsize = 50`: Batchsize during training from the training-set.
- `testfrac = 0.1`: Fraction of `playings` used to create test-sets.
- `augment = true`: Whether to use augmentation on the created data sets.
- `replays = 0`: Add datasets from last `replay` epochs to the trainset.
- `quiet = false`: Whether to suppress logging of training progress.
- `branch`: Random branching function. See `Data.record`.
- `prepare`: Preparation function. See `Data.record`.
- `callback_epoch`: Function called after every epoch.
- `callback_iter`: Function called after every iteration.
- `distributed = false`: Shall workers be used for self-playings?
- `tickets = nothing`: Number of tickets if distributed.
- `optimizer = Knet.SGD`: Optimizer for each weight in the training model.
- `kwargs...`: Keyword arguments for `optimizer`

# Examples
```julia
# Train a neural network model to learn TicTacToe by playing against itself
G = Game.TicTacToe
model = Model.NeuralModel(G, Model.@chain G Conv(64, "relu") Dense(32, "relu"))
player = Player.MCTSPlayer(model, power = 50, temperature = 0.75, exploration = 2.)
Training.train!(player, epochs = 5, playings = 100)
```
"""
function train!( player :: MCTSPlayer
               ; loss = Loss()
               , prepare = prepare(steps = 0)
               , branch = branch(prob = 0.)
               , augment = true
               , distributed = false
               , tickets = nothing
               , kwargs... )

  # Only use player-enabled features for recording if they are compatible with
  # the loss
  features = feature_compatibility(loss, player) ? Jtac.features(player) : Feature[]

  # Function to generate datasets through selfplays
  gen_data = (cb, n) -> record( player
                              , n
                              , prepare = prepare
                              , branch = branch
                              , augment = augment
                              , features = features
                              , merge = false
                              , callback = cb 
                              , distributed = distributed
                              , tickets = tickets )


  _train!(player, gen_data; loss = loss, kwargs...)

end


"""
    train_against!(player, enemy; <keyword arguments>)

Train an MCTS `player` under `loss` through playing against `enemy`.

The training model of `player` learns to predict the MCTS policy, the value of
game states, and possible features (if `player` supports the same features as
the keyword argument `loss`) when playing against `enemy`. If the enemy is too
good, this may turn out to be a bad learning mode: a player that loses all the
time will produce very pessimistic value predictions for each single game state,
which will harm the MCTS algorithm for improved policies. Note that only players
with NeuralModel-based training models can be trained.

# Arguments
- `loss = Loss()`: Loss used for training.
- `start`: Function that (randomly) yields -1 or 1 to fix the starting player.
- `epochs = 10`: Number of epochs.
- `playings = 20`: Games played per `epoch` for training-set generation.
- `iterations = 10`: Number of training epochs per training-set.
- `batchsize = 50`: Batchsize during training from the training-set.
- `testfrac = 0.1`: Fraction of `playings` used to create test-sets.
- `augment = true`: Whether to use augmentation on the created data sets.
- `replays = 0`: Add datasets from last `replay` epochs to the trainset.
- `quiet = false`: Whether to suppress logging of training progress.
- `branch`: Random branching function. See `Data.record_against`.
- `prepare`: Preparation function. See `Data.record_against`.
- `callback_epoch`: Function called after every epoch.
- `callback_iter`: Function called after every iteration.
- `distributed = false`: Shall workers be used for self-playings?
- `tickets = nothing`: Number of tickets if distributed.
- `optimizer = Knet.SGD`: Optimizer for each weight in the training model.
- `kwargs...`: Keyword arguments for `optimizer`

# Examples
```julia
# Train a neural network model by playing against an MCTS player
G = Game.TicTacToe
model = Model.NeuralModel(G, Model.@chain G Conv(64, "relu") Dense(32, "relu"))
player = Player.MCTSPlayer(model, power = 50, temperature = 0.75, exploration = 2.)
enemy = Player.MCTSPlayer(power = 250)
Training.train_against!(player, enemy, epochs = 5, playings = 100)
```
"""
function train_against!( player :: MCTSPlayer
                       , enemy
                       ; loss = Loss()
                       , start :: Function = () -> rand([-1, 1])
                       , branch = branch(prob = 0.)
                       , prepare = prepare(steps = 0)
                       , augment = true
                       , distributed = false
                       , tickets = nothing
                       , kwargs... )

  features = feature_compatibility(loss, player) ? features(player) : Feature[]

  gen_data = (cb, n) -> record_against( player
                                      , enemy
                                      , n
                                      , start = start
                                      , augment = augment
                                      , branch = branch
                                      , prepare = prepare
                                      , features = features
                                      , merge = false
                                      , callback = cb
                                      , distributed = distributed
                                      , tickets = tickets )

  _train!(player, gen_data; loss = loss, kwargs...)

end


"""
    train_from_model!(pupil, teacher [, players]; <keyword arguments>)

Train a `pupil` by letting it approximate the predictions of
`teacher`'s model on game states created by `players`.

In each epoch, two random `players` are chosen (by default, these are the pupil
and the teacher) to generate a set of game states by playing the game. These
games are used to create a training set by applying the teachers training model.
Note that only players with NeuralModel-based training models can be trained
currently.

# Arguments
- `loss = Loss()`: Loss used for training.
- `epochs = 10`: Number of epochs.
- `playings = 20`: Games played per `epoch` for training-set generation.
- `iterations = 10`: Number of training epochs per training-set.
- `batchsize = 50`: Batchsize during training from the training-set.
- `testfrac = 0.1`: Fraction of `playings` used to create test-sets.
- `branch`: Random branching function.
- `prepare`: Preparation function.
- `augment = true`: Whether to use augmentation on the created data sets.
- `replays = 0`: Add datasets from last `replay` epochs to the trainset.
- `quiet = false`: Whether to suppress logging of training progress.
- `callback_epoch`: Function called after every epoch.
- `callback_iter`: Function called after every iteration.
- `optimizer = Knet.SGD`: Optimizer for each weight in the training model.
- `kwargs...`: Keyword arguments for `optimizer`
"""
function train_from_model!( pupil :: AbstractPlayer{G}
                          , teacher :: Union{AbstractModel{H}, AbstractPlayer{H}}
                          , players = [pupil]
                          ; loss = Loss()
                          , branch = branch(prob = 0.)
                          , prepare = prepare(steps = 0)
                          , augment = true
                          , kwargs... 
                          ) where {H <: AbstractGame, G <: H}

  # Check if pupil and teacher are compatible featurewise
  use_features = features(pupil) == features(teacher) != Feature[]

  # Function that generates datasets
  gen_data = (cb, n) -> begin

    # TODO: Async?

    # Construct a list of games by letting two players compete
    datasets = asyncmap(1:n) do _

      # Get two players at random
      p1, p2 = rand(players, 2)

      # Watch them playing
      games = pvp_games(p1, p2, game = prepare(G()))[1:end-1]

      # Force the players to 'make errors' sometimes
      branchgames = branch.(games)
      branches = map(g -> pvp_games(p1, p2, g), branchgames)

      # Let the teacher model look at all games to generate a dataset
      ds = record_model( training_model(teacher)
                       , vcat(games, branches...)
                       , use_features = use_features
                       , augment = augment )

      # Give the signal that one playing is complete
      cb()

      ds

    end

    datasets

  end

  _train!(pupil, gen_data; loss = loss, kwargs...)

end


# -------- Training With Contests -------------------------------------------- #

"""
    with_contest(train_function, player, <arguments>; <keyword arguments>)

Train `player` according to `train_function` while conducting regular contests
during the training process.

The argument `train_function` must be one of `train!`, `train_against!`, or
`train_from!`. The function `with_contest` will always at least print the
results of a contest before training and after training. 

# Arguments
The non-keyword arguments supported are the non-keyword arguments that are also
accepted / needed by `train_function`. As keyword arguments, all keyword
arguments of `train_function` are supported. Additionally, the following
arguments can be used to adapt the behavior of `with_contest`:

- `interval = 10`: After how many epochs do contests take place.
- `length = 250`: Number of playings during one contest. 
- `opponents`: List of opponents for the competition. To this list the
   two players `current` and `initial` are added, which are `IntuitionPlayer`s
   with the model (or a copy of the initial model) of `player`.
- `temperature`: Temperature of the intuition players `current` and `initial`.
- `cache = 0`: Number of games to be cached. If `cache > 0`, a number
   of `cache` games between all players that are independent of `player`'s model
   are conducted and cached for all competitions that follow.

# Examples
```julia

G = Game.TicTacToe
loss = Training.Loss(policy = 0.25)
opponents = [Player.MCTSPlayer(power = 50), Player.MCTSPlayer(power = 500)]

model = Model.NeuralModel(G, Model.@chain G Conv(100, "relu") Dense(32, "relu"))
player = Player.MCTSPlayer(model, power = 25)

Training.with_contest( Training.train!
                     , player
                     , loss = loss
                     , cache = 1000
                     , length = 500
                     , opponents = opponents
                     , interval = 5
                     , epochs = 20
                     , playings = 150 )
```

"""
function with_contest( trainf!     # the training function
                     , player :: MCTSPlayer
                     , args...
                     ; loss = Loss()
                     , opponents = AbstractPlayer[]
                     , length :: Int = 250
                     , cache :: Int = 0
                     , interval :: Int = 10
                     , temperature = player.temperature
                     , distributed = false
                     , epochs = 10
                     , kwargs... )

  # One can set length == 0 to access trainf! without contests
  if length <= 0

    @info "Contests are disabled."
    return trainf!( player
                  , args...
                  ; loss = loss
                  , epochs = epochs
                  , distributed = distributed
                  , kwargs... )

  end

  # Rename the length keyword argument
  len = length
  length = Base.length

  # List of all players that will compete in the contest
  players = [

    IntuitionPlayer( player.model
                   , temperature = temperature
                   , name = "current" );
    IntuitionPlayer( copy(player.model)
                   , temperature = temperature
                   , name = "initial" );
    opponents

  ]

  # Creating a cache of contest games for all players that do not depend on the
  # model that will be trained
  if cache > 0

    # Get the training model
    model = training_model(player)

    # Reorder players such that the active (i.e., learning) players come last
    # and passive ones first
    active = filter(p -> training_model(p) == model, players)
    passive = filter(p -> training_model(p) != model, players)
    players = [passive; active]

    # Get the respective numbers of players
    n  = length(players)
    np = length(passive)
    na = length(active)

    # Get the indices of all active models
    aidx = collect(1:na) .+ np

    # Get the progress-meter going
    step, finish = stepper("# Caching...", cache)

    # Create the cache for games between passive players
    cache_results = zeros(Int, n, n, 3)
    cache_results[1:np,1:np,:] = compete( passive
                                        , cache
                                        , distributed = distributed
                                        , callback = step )

    # Remove the progress bar
    finish()

    #  Leave a message that confirms caching.
    printstyled("# Cached $cache matches by $n players", color = 245)
    println()

  else

    # If caching is not activated, regard each player as active
    aidx = 1:length(players)
    cache_results = zeros(Int, length(players), length(players), 3)

  end

  # Create the callback function
  cb = epoch -> begin
    if epoch % interval == 0

      step, finish = stepper("# Contest...", len)

      results = compete( players, len, aidx
                       , distributed = distributed , callback = step )
      finish()

      # special printing, since we want to prepend '# ' to each line
      print_ranking(Ranking(players, results .+ cache_results))

      if 0 < epoch < epochs
        print_loss_header(loss, feature_compatibility(loss, player))
      end
    end
  end

  # Run contest before training
  cb(0)

  # Train with a contest every interval epochs
  trainf!( player
         , args...
         ; loss = loss
         , callback_epoch = cb
         , epochs = epochs
         , distributed = distributed
         , kwargs... )

  # Run contest after training, if it was not run in trainf! already
  epochs % interval == 0 ? nothing : cb(epochs)

end

