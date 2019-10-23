
# -------- Auxiliary Functions ----------------------------------------------- #

# Set an optimizer for all parameters of a model
function set_optimizer!(model, opt = Knet.Adam; kwargs...)

  for param in Knet.params(model)
    
    # Feature heads that are not used have length 0. They do not get an
    # optimizer.
    if length(param) > 0

      # If opt is given, overwrite every optimizer in param.opt
      if !isnothing(opt)

        param.opt = opt(; kwargs...)

      # If opt is not given, overwrite only non-existing param.opt with Adam
      elseif isnothing(opt) && isnothing(param.opt)

        param.opt = Knet.Adam(; kwargs...)

      end

    end

  end

end

# A single training step, the loss is returned
function train_step!(l :: Loss, model, cache :: DataCache)

  tape = Knet.@diff sum(loss(l, model, cache))

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


  # Get basic info about the player's model
  model = training_model(player)
  gpu = on_gpu(model)

  # Check if features can be enabled
  use_features = check_features(loss, model, trainset)

  # Set/overwrite optimizers
  set_optimizer!(model, optimizer; kwargs...)

  # Generate the batch iterator
  batches = Batches( trainset
                   , batchsize
                   , shuffle = true
                   , partial = false
                   , gpu = gpu )

  # Iterate through the trainset
  for j in 1:epochs

    for (i, cache) in enumerate(batches)

      train_step!(loss, model, cache)
      callback_step(i)

    end

    callback_epoch(j)

  end

end


# -------- Training by Playing ----------------------------------------------- #

function _train!( player :: Player{G}
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
                , kwargs... 
                ) where {G <: Game}

  @assert playings > 0 "Number of playings for training must be positive"

  # If we do not print results, it is not necessary to test
  quiet || testfrac < 0 && (testfrac = 0.)

  # Get number of playings used for training
  train_playings = ceil(Int, playings * (1-testfrac))

  # Print the loss header if not quiet
  !quiet && print_loss_header(loss, check_features(loss, player))

  # Set the optimizer for the player's model
  set_optimizer!(training_model(player), optimizer; kwargs...)

  for i in 1:epochs

    # Generate callback functions for the progress meter
    step, finish = stepper("# Playing...", playings)
    cb = quiet ? () -> nothing : step

    # Generate train and testsets
    datasets = gen_data(cb, playings)
    trainset = merge(datasets[1:train_playings]...)

    if testfrac > 0
      testset = merge(datasets[train_playings+1:playings]...)
    else
      testset = DataSet{G}() 
    end

    # Clear the progress bar (if it was printed)
    finish()

    # Prepare the next progress bar
    steps = iterations * div(length(trainset), batchsize)
    step, finish = stepper("# Learning...", steps)
    cb = (_) -> quiet ? nothing : step()

    # The Knet allocator works better for training but worse for playing
    switch_knet_allocator()

    # Train the player with the generated dataset
    train!( player
          , trainset
          , loss = loss
          , epochs = iterations
          , batchsize = batchsize
          , callback_step = cb
          , callback_epoch = callback_iter )

    # Clear the progress bar
    finish()

    # Calculate and print loss for this epoch if not quiet
    !quiet && print_loss(loss, player, i, trainset, testset)

    # Undo the changes in the Knet allocator
    switch_knet_allocator()

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
- `distributed = false`: Shall workers be used for self-playings?
- `tickets = nothing`: Number of tickets if distributed.
- `optimizer = Adam`: Optimizer for each weight in the training model.
- `kwargs...`: Keyword arguments for `optimizer`

# Examples
```julia
# Self-train a simple neural network model for 5 epochs
model = NeuralModel(TicTacToe, @chain TicTacToe Conv(64, relu) Dense(32, relu))
player = MCTSPlayer(model, power = 50, temperature = 0.75, exploration = 2.)
train_self!(player, epochs = 5, playings = 100)
```
"""
function train_self!( player :: MCTSPlayer
                    ; loss
                    , branching = 0.
                    , augment = true
                    , distributed = false
                    , tickets = nothing
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
                                   , merge = false
                                   , callback = cb 
                                   , distributed = distributed
                                   , tickets = tickets )


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
- `distributed = false`: Shall workers be used for self-playings?
- `tickets = nothing`: Number of tickets if distributed.
- `optimizer = Adam`: Optimizer for each weight in the training model.
- `kwargs...`: Keyword arguments for `optimizer`

# Examples
```julia
# Train a simple neural network model against an MCTS player for 5 epochs
model = NeuralModel(TicTacToe, @chain TicTacToe Conv(64, relu) Dense(32, relu))
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
                       , distributed = false
                       , tickets = nothing
                       , kwargs... )

  features = check_features(loss, player) ? features(player) : Feature[]

  gen_data = (cb, n) -> record_against( player
                                      , enemy
                                      , n
                                      , start = start
                                      , augment = augment
                                      , branching = branching
                                      , features = features
                                      , merge = false
                                      , callback = cb
                                      , distributed = distributed
                                      , tickets = tickets )

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

  # Complement the branching options (before, during, steps)
  branching = branch_options(branching)

  # Check if pupil and teacher are compatible featurewise
  use_features = features(pupil) == features(teacher)

  # Function that generates datasets
  gen_data = (cb, n) -> begin

    # TODO: Async?

    # Construct a list of games by letting two players compete
    datasets = asyncmap(1:n) do _

      # Get two players at random
      p1, p2 = rand(players, 2)

      # Watch them playing
      games = pvp_games(p1, p2, branch_root(G(), branching))[1:end-1]

      # Force the players to 'make errors' sometimes
      branchgames = branch(games, branching)
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
                     ; loss
                     , opponents = Player[]
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
    print(gray_crayon)
    println("# Cached $cache matches by $n players")

  else

    # If caching is not activated, regard each player as active
    aidx = 1:length(players)
    cache_results = zeros(Int, length(players), length(players), 3)

  end

  # Create the callback function
  cb = epoch -> begin
    if epoch % interval == 0

      step, finish = stepper("# Contest...", len)

      results = compete( players
                       , len
                       , aidx
                       , distributed = distributed
                       , callback = step )

      finish()

      print_ranking(Ranking(players, results + cache_results))

      0 < epoch < epochs && print_loss_header(loss, check_features(loss, player))

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

