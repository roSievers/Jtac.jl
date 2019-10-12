
# -------- Auxiliary Functions ----------------------------------------------- #

# Set an optimizer for all parameters of a model
function set_optimizer!(model, opt = Knet.Adam; kwargs...)

  for param in Knet.params(model)
    
    # Feature heads that are not used have length 0. They do not get an
    # optimizer.
    if length(param) > 0
      param.opt = opt(; kwargs...)
    end
  end

end

# A single training step, the loss is returned
function train_step!(l :: Loss, model, dataset :: DataSet)

  tape = Knet.@diff sum(loss(l, model, dataset))

  for param in Knet.params(model)
    Knet.update!(Knet.value(param), Knet.grad(tape, param), param.opt)
  end

  Knet.value(tape)

end


# -------- Training on Datasets ---------------------------------------------- #

"""
    train!(model/player, dataset; loss, <keyword arguments>)

Train `model`, or the training model of `player`, on `dataset` to optimize
`loss`.

# Arguments
- `epochs = 10`: Number of iterations through `dataset`.
- `batchsize = 50`: Batchsize for the update steps.
- `callback_epoch`: Function called after each epoch.
- `callback_step`: Function called after each update step.
- `optimizer = nothing`: Optimizer initialized for each weight in `model`
- `kwargs...`: Keyword arguments for `optimizer`

# Examples
```julia
set = record_self(MCTSPlayer(), 10, game = TicTacToe()) 
model = NeuralModel(TicTacToe, @chain TicTacToe Dense(50))
loss = Loss(policy = 0.15, regularization=1e-4)
train!(model, set, loss = loss, epochs = 15)
```

"""
function train!( player  :: Union{Player, Model}
               , trainset :: DataSet
               ; loss
               , epochs = 10
               , batchsize = 50
               , callback_step = (_) -> nothing
               , callback_epoch = (_) -> nothing
               , optimizer = nothing
               , kwargs... )

  model = training_model(player)

  !isnothing(optimizer) && set_optimizer(model, optimizer; kwargs...)

  for j in 1:epochs

    batches = minibatch(trainset, batchsize, shuffle = true, partial = false)

    for (i, batch) in enumerate(batches)

      train_step!(loss, model, batch)
      callback_step(i)

    end

    callback_epoch(j)

  end

end


# -------- Training by Playing ----------------------------------------------- #

function _train!( player
                , gen_data :: Function
                ; loss
                , epochs = 10
                , playings = 20
                , iterations = 10
                , batchsize = 50
                , testfrac = 0.1
                , optimizer = Knet.Adam
                , quiet = false
                , callback_epoch = (_) -> nothing
                , callback_iter = (_) -> nothing
                , kwargs... )

  # How many playings do we do for the purpose of testing?
  test_playings = ceil(Int, playings * testfrac)
  total_playings = test_playings + playings

  # Print the loss header if not quiet
  !quiet && print_loss_header(loss, check_features(loss, player))

  # Set the optimizer for the player's model
  set_optimizer!(training_model(player), optimizer; kwargs...)

  for i in 1:epochs

    # Print progress-meter if not quiet
    !quiet && (pm = progressmeter( total_playings + 1, "# Playing..."))
    cb = () -> quiet ? nothing : progress!(pm)

    # Generate train and testsets
    trainset = gen_data(cb, playings)
    !quiet && (testset = gen_data(cb, test_playings))

    # Print next progress-meter if not quiet
    update_steps = iterations * div(length(trainset), batchsize)
    !quiet && clear_output!(pm)
    !quiet && (pm = progressmeter(update_steps + 1, "# Learning..."))
    cb = (_) -> quiet ? nothing : progress!(pm)

    # Train the player with the generated dataset
    train!( player
          , trainset
          , loss = loss
          , epochs = iterations
          , batchsize = batchsize
          , callback_step = cb
          , callback_epoch = callback_iter )

    # Calculate and print loss for this epoch if not quiet
    !quiet && clear_output!(pm)
    !quiet && print_loss(loss, player, i, trainset, testset)

    # Call callback function
    callback_epoch(i) 

  end 

end


"""
    train_self!(player; loss, <keyword arguments>)

Train an MCTS `player` under `loss` via playing against itself.

The training model of `player` learns to predict the MCTS policy, the value of
game states, and possible features (if `player` supports the same features as
`loss`). Note that only players with NeuralModel-based training models can
be trained currently.

# Arguments
- `epochs = 10`: Number of epochs.
- `playings = 20`: Games played per `epoch` for training-set generation.
- `iterations = 10`: Number of training epochs per training-set.
- `batchsize = 50`: Batchsize during training from the training-set.
- `testfrac = 0.1`: Fraction of `playings` used to create test-sets.
- `branching = 0.`: Random branching probability.
- `augment = true`: Whether to use augmentation on the created data sets.
- `quiet = false`: Whether to suppress logging of training progress.
- `callback_epoch`: Function called after every epoch.
- `callback_iter`: Function called after every iteration.
- `optimizer = Adam`: Optimizer for each weight in the training model.
- `kwargs...`: Keyword arguments for `optimizer`

# Examples
```julia
# Self-train a simple neural network model for 5 epochs
model = NeuralModel(TicTacToe, @chain TicTacToe Conv(64) Dense(32))
player = MCTSPlayer(model, power = 50, temperature = 0.75, exploration = 2.)
train_self!(player, epochs = 5, playings = 100)
```
"""
function train_self!( player :: MCTSPlayer
                    ; loss
                    , branching = 0.
                    , augment = true
                    , kwargs... )

  # Only use player-enabled features for recording if they are compatible with
  # the loss
  features = check_features(loss, player) ? Jtac.features(player) : Feature[]

  # Function to generate datasets through selfplays
  gen_data = (cb, n) -> record_self( player
                                   , n
                                   , augment = augment
                                   , branching = branching
                                   , features = features
                                   , callback = cb )

  _train!(player, gen_data; loss = loss, kwargs...)

end


"""
    train_against!(player, enemy; loss, <keyword arguments>)

Train an MCTS `player` under `loss` through playing against `enemy`.

The training model of `player` learns to predict the MCTS policy, the value of
game states, and possible features (if `player` supports the same features as
`loss`) when playing against `enemy`. If the enemy is too good, this may turn
out to be a bad learning mode: a player that loses all the time will produce
very pessimistic value predictions for each single game state, which will
harm the MCTS algorithm for improved policies. Note that only players with
NeuralModel-based training models can be trained currently.

# Arguments
- `start`: Function that (randomly) yields -1 or 1 to fix the starting player.
- `epochs = 10`: Number of epochs.
- `playings = 20`: Games played per `epoch` for training-set generation.
- `iterations = 10`: Number of training epochs per training-set.
- `batchsize = 50`: Batchsize during training from the training-set.
- `testfrac = 0.1`: Fraction of `playings` used to create test-sets.
- `branching = 0.`: Random branching probability.
- `augment = true`: Whether to use augmentation on the created data sets.
- `quiet = false`: Whether to suppress logging of training progress.
- `callback_epoch`: Function called after every epoch.
- `callback_iter`: Function called after every iteration.
- `optimizer = Adam`: Optimizer for each weight in the training model.
- `kwargs...`: Keyword arguments for `optimizer`

# Examples
```julia
# Train a simple neural network model against an MCTS player for 5 epochs
model = NeuralModel(TicTacToe, @chain TicTacToe Conv(64) Dense(32))
player = MCTSPlayer(model, power = 50, temperature = 0.75, exploration = 2.)
enemy = MCTSPlayer(power = 250)
train_against!(player, enemy, epochs = 5, playings = 100)
```
"""
function train_against!( player :: MCTSPlayer
                       , enemy
                       ; loss
                       , start :: Function = () -> rand([-1, 1])
                       , branching = 0.
                       , augment = true
                       , kwargs... )

  features = check_features(loss, player) ? features(player) : Feature[]

  gen_data = (cb, n) -> record_against( player
                                      , enemy
                                      , n
                                      , start = start
                                      , augment = augment
                                      , branching = branching
                                      , features = features
                                      , callback = cb ) 

  _train!(player, gen_data; loss = loss, kwargs...)

end


"""
    train_from!(pupil, teacher [, players]; loss, <keyword arguments>)

Train a `pupil` under `loss` by letting it approximate the predictions of
`teacher` on game states created by `players`.

In each epoch, two random `players` are chosen (by default, these are the pupil
and the teacher) to generate a set of game states by playing the game. These
games are used to create a training set by applying the teachers training model.
Note that only players with NeuralModel-based training models can be trained
currently.

# Arguments
- `epochs = 10`: Number of epochs.
- `playings = 20`: Games played per `epoch` for training-set generation.
- `iterations = 10`: Number of training epochs per training-set.
- `batchsize = 50`: Batchsize during training from the training-set.
- `testfrac = 0.1`: Fraction of `playings` used to create test-sets.
- `branching = 0.`: Random branching probability.
- `augment = true`: Whether to use augmentation on the created data sets.
- `quiet = false`: Whether to suppress logging of training progress.
- `callback_epoch`: Function called after every epoch.
- `callback_iter`: Function called after every iteration.
- `optimizer = Adam`: Optimizer for each weight in the training model.
- `kwargs...`: Keyword arguments for `optimizer`
"""
function train_from!( pupil :: Player{G}
                    , teacher
                    , players = [pupil, teacher]
                    ; loss
                    , branching = 0.
                    , augment = true
                    , kwargs... 
                    ) where {G <: Game}

  # Check if pupil and teacher are compatible featurewise
  use_features = features(pupil) == features(teacher)

  # Function that generates datasets
  gen_data = (cb, n) -> begin

    # TODO: Async?

    # Construct a list of games by letting two players compete
    datasets = map(1:n) do _

      # Get two players at random
      p1, p2 = rand(players, 2)

      # Watch them playing
      games = pvp_games(p1, p2, G())

      # Force the to 'make errors' sometimes
      branchpoints = randsubseq(games, branching)
      branches = map(g -> pvp_games(p1, p2, g), games)

      # Let the teacher model look at all games to generate a dataset
      ds = record_model( training_model(teacher)
                       , vcat(games, branches...)
                       , use_features = use_features
                       , augment = augment )

      # Give the signal that one playing is complete
      cb()

      ds

    end

    merge(datasets...)

  end

  _train!(pupil, gen_data; loss = loss, kwargs...)

end


# -------- Training With Contests -------------------------------------------- #

"""
    with_contest(train_function, player, <arguments>; <keyword arguments>)

Train `player` according to `train_function` while conducting regular contests
during the training process.

The argument `train_function` must be one of `train_self!`, `train_against!`, or
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
   two players `current` and `initial` are added, which are IntuitionPlayers
   with the model (or a copy of the initial model) of `player`.
- `temperature`: Temperature of the intuition players `current` and `initial`.
- `cache = 0`: Number of games to be cached. If `cache > 0`, a number
   of `cache` games between all players that are independent of `player`'s model
   are conducted and cached for all competitions that follow.

# Examples
```julia

G = TicTacToe
loss = Loss(policy = 0.25)
opponents = [MCTSPlayer(power = 50), MCTSPlayer(power = 500)]

model = NeuralModel(G, @chain G Conv(100, relu) Dense(32, relu))
player = MCTSPlayer(model, power = 25)

with_contest( train_self!
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
                     ; opponents = Player[]
                     , length :: Int = 250
                     , cache :: Int = 0
                     , interval :: Int = 10
                     , temperature = player.temperature
                     , epochs = 10
                     , kwargs... )

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
    aplayers = filter(p -> training_model(p) == model, players)
    pplayers = filter(p -> training_model(p) != model, players)
    players = [pplayers; aplayers]

    # Get the indices of all active models
    active = collect(1:length(aplayers)) .+ length(pplayers)

    # Get the progress-meter going
    n = length(pplayers) * (length(pplayers) - 1)
    p = progressmeter(n + 1, "# Caching...")

    # Create the cache
    cache = playouts( pplayers, cache, callback = () -> progress!(p) )

    # Remove the progress bar and leave a message that confirms caching.
    clear_output!(p)
    println( gray_crayon
           , "# Cached $(length(cache)) matches by $(length(players)) players" )

  else

    # If caching is not activated, regard each player as active
    active = 1:length(players)
    cache = []

  end

  # Check if the player's model is async
  async = isasync(playing_model(player))

  # Create the callback function
  cb = epoch -> begin
    if epoch % interval == 0
      print_contest(players, len, async, active, cache)
    end
  end

  # Run contest before training
  cb(0)

  # Train with a contest every interval epochs
  trainf!(player, args...; callback_epoch = cb, epochs = epochs, kwargs...)

  # Run contest after training, if it was not run in trainf! already
  epochs % interval == 0 ? nothing : cb(0)

end

