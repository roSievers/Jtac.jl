
"""
    _loss(model, cache; weights, reg_targets)

Internally used loss function for training. The prediction targets of `model`
and `cache` must be consistent.
"""
function _loss( model :: NeuralModel{G, GPU}
              , cache :: Cache{G, GPU}
              ; weights = (;)
              , reg_targets = []
              ) where {G, GPU}

  output = model(cache.data, true)
  pred_targets = Target.targets(model)
  pred = map(pred_targets, cache.labels, output) do t, l, v
    weight = Float32(get(weights, Target.name(t), 1.0))
    weight * Target.loss(t, l, v) / length(cache)
  end
  reg = map(reg_targets) do t
    weight = Float32(get(weights, Target.name(t), 1.0))
    weight * Target.loss(t, model)
  end
  [pred; reg]
end

"""
    loss(model, dataset; maxbatch = 1024, weights, reg_targets)
    loss(model, cache; weights, reg_targets)

Calculate loss components for the prediction targets of `model` as well as the
regularization targets `reg_targets`, weighted by `weights`. The prediction
targets of `model` and `cache` must be consistent.
"""
function loss( model :: NeuralModel{G, GPU}
             , cache :: Cache{G, GPU}
             ; reg_targets
             , weights = (;)
             ) where {G, GPU}
  losses = _loss(model, cache; reg_targets, weights)
  targets = [Target.targets(model); reg_targets]
  names = Target.name.(targets)
  (; zip(names, losses)...)
end

function loss( model :: NeuralModel{G, GPU}
             , dataset :: DataSet{G}
             ; maxbatch = 1024
             , reg_targets = []
             , weights = (;)
             ) where {G, GPU}

  pred_targets = Target.targets(model)
  ds = Target.adapt(dataset, pred_targets)

  batches = Batches(ds, maxbatch, gpu = GPU)
  losses = sum(batches) do cache
    _loss(model, cache; reg_targets, weights) .* length(cache)
  end ./ length(ds)
  targets = [pred_targets; reg_targets]
  names = Target.name.(targets)
  (; zip(names, losses)...)
end

# -------- Auxiliary Functions ----------------------------------------------- #

# Set an optimizer for all parameters of a model
function set_optimizer!(model, opt = Knet.SGD; kwargs...)
  for param in Knet.params(model)
    # If opt is given, overwrite every optimizer in param.opt
    if !isnothing(opt)
      param.opt = opt(; kwargs...)
    # If opt is not given, overwrite only non-existing param.opt with SGD
    elseif isnothing(opt) && isnothing(param.opt)
      param.opt = Knet.SGD(; kwargs...)
    end
  end
end

# A single training step, the loss is returned
function train_step!(model, cache :: Cache; kwargs...)

  tape = Knet.@diff sum(_loss(model, cache; kwargs...))

  for param in Knet.params(model)
    Knet.update!(Knet.value(param), Knet.grad(tape, param), param.opt)
  end

  Knet.value(tape)
end

# -------- Logging Losses while Training ------------------------------------- #

function info_loss_header(targets)
  strings = map(targets) do t
    str = string(Target.name(t))[1:min(end, 10)]
    Printf.@sprintf("%10s", str)
  end
  join(["#"; "   epoch"; strings; "     total"; "    length"], " ") |> println
end

function info_loss_values(model, epoch, train, test = nothing; kwargs...)
  if isnothing(test)
    dsets = [(train, :normal)]
  else
    dsets = [(train, 245), (test, :normal)]
  end
  for (ds, col) in dsets
    # Compute the losses and get them as strings
    l = loss(model, ds; kwargs...)
    losses = map(x -> @sprintf("%10.3f", x), values(l))

    # Print everything in grey (for train) and white (for test)
    str = @sprintf( "%10d %s %10.3f %10d\n"
                  , epoch , join(losses, " "), sum(l), length(ds) )
    printstyled(str, color = col)
  end
end

function info_ranking(rk)

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
    train!(model/player, trainset[s], testset; <keyword arguments>)

Train `model`, or the training model of `player`, on `trainset`.
Evaluate the loss on `testset` after each epoch.

The second argument can be a vector of trainsets. In this case, not all game
states are brought into array representation at the same time, reducing the
memory requirements.

# Arguments
- `epochs = 10`: Number of iterations through `dataset`.
- `batchsize = 50`: Batchsize for the update steps.
- `reg_targets`: Regularization targets.
- `weights`: Target weights.
- `callback_epoch`: Function called after each epoch.
- `callback_step`: Function called after each update step.
- `quiet`: Whether to print training progress information.
- `optimizer`: Optimizer to be initialized for each weight in `model`.
- `kwargs...`: Keyword arguments for `optimizer`

# Examples
```julia
G = Game.TicTacToe
dataset = Player.record(Player.MCTSPlayer(), 10, instance = G) 
model = Model.NeuralModel(G, Model.@chain G Dense(50, "relu"))

reg_targets = [Target.L2Reg()]
weights = (value = 1., policy = 0.5, l2reg = 1e-3)
Training.train!(model, dataset; epochs = 15, reg_targets, weights)
```
"""
function train!( player  :: Union{AbstractPlayer, AbstractModel}
               , trainsets :: Vector{<: DataSet}
               , testset = nothing
               ; epochs = 10
               , batchsize = 50
               , reg_targets = []
               , weights = (;)
               , callback_step = (_) -> nothing
               , callback_epoch = (_) -> nothing
               , optimizer = nothing
               , quiet = false
               , store_on_gpu = false
               , kwargs... )

  @assert !isempty(trainsets) "Trainset vector is empty"

  model = training_model(player)
  set_optimizer!(model, optimizer; kwargs...)
  gpu = on_gpu(model)

  trainsets = [Target.adapt(ts, model) for ts in trainsets]

  targets = [Target.targets(model); reg_targets]
  !quiet && info_loss_header(targets)


  # Iterate through the trainset epochs times
  for j in 1:epochs

    if !quiet
      steps = ceil(Int, sum(length, trainsets) / batchsize)
      step, finish = Util.stepper("# Learning...", steps)
    end

    for ts in trainsets
      # Generate the batch iterator (each iteration is shuffled independently)
      batches = Batches( ts
                       , batchsize
                       ; shuffle = true
                       , partial = true
                       , gpu
                       , store_on_gpu )

      for (i, cache) in enumerate(batches)

        train_step!(model, cache; reg_targets, weights)
        callback_step(i)
        !quiet && step()

      end
    end

    if !quiet
      finish()
      info_loss_values(model, j, trainsets[1], testset; reg_targets, weights)
    end

    callback_epoch(j)
  end
end

function train!( player :: Union{AbstractPlayer, AbstractModel}
               , trainset :: DataSet
               , args...
               ; kwargs... )
  train!(player, [trainset], args...; kwargs...)
end


# -------- Training by Playing ----------------------------------------------- #


function _train!( player :: AbstractPlayer{G}
                , gen_data :: Function
                ; epochs = 10
                , matches = 20
                , iterations = 10
                , batchsize = 50
                , testfrac = 0.1
                , reg_targets = []
                , weights = (;)
                , optimizer = Knet.Momentum
                , replays = 0
                , quiet = false
                , callback_epoch = (_) -> nothing
                , callback_iter = (_) -> nothing
                , kwargs...
                ) where {G <: AbstractGame}

  @assert matches > 0 "Number of matches for training must be positive"

  model = training_model(player)
  set_optimizer!(model, optimizer; kwargs...)
  targets = [Target.targets(model); reg_targets]

  # If we do not print results it is not necessary to test
  quiet || testfrac < 0 && (testfrac = 0.)

  # Get number of matches used for training
  train_matches = ceil(Int, matches * (1-testfrac))

  # Print the loss header if not quiet
  !quiet && info_loss_header(targets)

  # Save the last `replay` generated datasets and use them for training
  replay_buffer = []

  for i in 1:epochs

    # Generate callback functions for the progress meter
    step, finish = Util.stepper("# Playing...", matches)
    cb = quiet ? () -> nothing : step

    # Generate new datasets for this generation and make sure that
    # they fit the model
    datasets = gen_data(cb, matches)
    datasets = [Target.adapt(ds, model) for ds in datasets]

    # Create the training set by merging with the last `matches` generations of
    # datasets that are stored in the replay_buffer
    trainset = merge(datasets[1:train_matches])
    for set in replay_buffer
      trainset = merge([trainset, set...])
    end

    # Create the testing set by collecting the remaining datasets generated
    # in this epoch
    if testfrac > 0
      testset = merge(datasets[train_matches+1:matches])
    else
      testset = nothing
    end

    # Clear the progress bar (if it was printed)
    finish()

    # Prepare the next progress bar
    steps = iterations * ceil(Int, length(trainset) / batchsize)
    step, finish = Util.stepper("# Learning...", steps)
    cb = (_) -> quiet ? nothing : step()

    # Train the player with the generated dataset
    train!( player
          , trainset
          ; epochs = iterations
          , batchsize
          , reg_targets
          , weights
          , quiet = true
          , callback_step = cb
          , callback_epoch = callback_iter )

    # Clear the progress bar
    finish()

    # Calculate and print loss for this epoch if not quiet
    !quiet && info_loss_values(model, i, trainset, testset; reg_targets, weights)

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

Train the `NeuralModel` of an MCTS `player` via playing against itself.

The training model of `player` learns to predict the improved MCTS policy, the
expected value of game states, and other optional prediction targets.

# Arguments
- `epochs = 10`: Number of epochs.
- `matches = 20`: Games played per `epoch` for training-set generation.
- `iterations = 10`: Number of training epochs per training-set.
- `batchsize = 50`: Batchsize during training from the training-set.
- `reg_targets = []`: Regularization targets.
- `weights = (;)`: Target weights.
- `testfrac = 0.1`: Fraction of `matches` used to create test-sets.
- `augment = true`: Whether to use augmentation on the created data sets.
- `replays = 0`: Add datasets from last `replay` epochs to the trainset.
- `quiet = false`: Whether to suppress logging of training progress.
- `branch`: Random branching function. See `Player.record`.
- `prepare`: Preparation function. See `Player.record`.
- `callback_epoch`: Function called after every epoch.
- `callback_iter`: Function called after every iteration.
- `optimizer = Knet.SGD`: Optimizer for each weight in the training model.
- `kwargs...`: Keyword arguments for `optimizer`

# Examples
```julia
# Train a neural network model to learn TicTacToe by playing against itself
G = Game.TicTacToe
model = Model.NeuralModel(G, Model.@chain G Conv(64, "relu") Dense(32, "relu"))
player = Player.MCTSPlayer(model, power = 50, temperature = 0.75, exploration = 2.)
Training.train!(player, epochs = 5, matches = 100)
```
"""
function train!( player :: MCTSPlayer{G}
               ; instance = () -> Game.instance(G)
               , branch = Game.branch(prob = 0., steps = 1)
               , augment = true
               , kwargs... ) where {G}

  gen_data = (cb, n) -> record( player
                              , n
                              ; instance 
                              , branch
                              , augment
                              , merge = false
                              , callback = cb )


  _train!(player, gen_data; kwargs...)

end


# -------- Training With Contests -------------------------------------------- #

"""
    train_contest!(player; <keyword arguments>)

Train `player` by selfplay while conducting contests during the training
process.

# Arguments
Supported keyword arguments include the one of `train!`. Additionally, the following
arguments can be used to adapt the contests.

- `interval = 10`: After how many epochs do contests take place.
- `pairings = 250`: Number of matches during one contest. 
- `opponents`: List of opponents for the competition. The
   two players `current` and `initial` are added to this list, which are
   `IntuitionPlayer`s with the base model (or a copy of the initial base model)
   of `player`.
- `temperature`: Temperature of the intuition players `current` and `initial`.
- `cache = 0`: Number of games to be cached. If `cache > 0`, a number
   of `cache` games between all players that are independent of `player`'s model
   are conducted and cached for all competitions that follow.

# Examples
```julia

G = Game.TicTacToe
opponents = [Player.MCTSPlayer(power = 50), Player.MCTSPlayer(power = 500)]

model = Model.NeuralModel(G, Model.@chain G Conv(100, "relu", padding = 1) Dense(32, "relu"))
player = Player.MCTSPlayer(model, power = 50)

Training.train_contest!( player
                       , cache = 1000
                       , pairings = 500
                       , opponents = opponents
                       , interval = 5
                       , epochs = 20
                       , matches = 150 )
```

"""
function train_contest!( player :: MCTSPlayer
                       , args...
                       ; reg_targets = []
                       , weights = (;)
                       , opponents = AbstractPlayer[]
                       , pairings :: Int = 250
                       , cache :: Int = 0
                       , interval :: Int = 10
                       , temperature = player.temperature
                       , epochs = 10
                       , kwargs... )

  if pairings <= 0

    @info "Contests disabled."
    return train!( player
                 , args...
                 ; reg_targets
                 , weights
                 , epochs
                 , kwargs... )

  end

  model = training_model(player)
  targets = [Target.targets(model); reg_targets]

  # List of all players that will compete in the contest
  players = [

    IntuitionPlayer( model
                   , temperature = temperature
                   , name = "current" );

    IntuitionPlayer( copy(model)
                   , temperature = temperature
                   , name = "initial" );
    opponents

  ]

  # Creating a cache of contest games for all players that do not depend on the
  # model that will be trained
  if cache > 0

    # Reorder players such that the active (i.e., learning) players come last
    # and passive ones first
    active = filter(p -> training_model(p) == model, players)
    passive = filter(p -> training_model(p) != model, players)
    players = [passive; active]

    # Get the respective numbers of players
    n  = length(players)
    np = length(passive)
    na = length(active)

    if np <= 1

      @warn "Cannot cache pairings with less than 2 passive players"
      aidx = 1:length(players)
      cache_results = zeros(Int, length(players), length(players), 3)

    else

      # Get the indices of all active models
      aidx = collect(1:na) .+ np

      # Get the progress-meter going
      step, finish = Util.stepper("# Caching...", cache)

      # Create the cache for games between passive players
      cache_results = zeros(Int, n, n, 3)
      cache_results[1:np,1:np,:] = compete( passive
                                          , cache
                                          , callback = step )

      # Remove the progress bar
      finish()

      #  Leave a message that confirms caching.
      printstyled("# Cached $cache matches by $n players", color = 245)
      println()

    end

  else

    # If caching is not activated, regard each player as active
    aidx = 1:length(players)
    cache_results = zeros(Int, length(players), length(players), 3)

  end

  # Create the callback function
  cb = epoch -> begin
    if epoch % interval == 0

      step, finish = Util.stepper("# Contest...", pairings)

      results = compete( players, pairings, aidx
                       ; callback = step )
      finish()

      # special printing, since we want to prepend '# ' to each line
      info_ranking(Ranking(players, results .+ cache_results))

      if 0 < epoch < epochs
        info_loss_header(targets)
      end
    end
  end

  # Run contest before training
  cb(0)

  # Train with a contest every interval epochs
  train!( player
        , args...
        ; reg_targets
        , weights
        , epochs
        , callback_epoch = cb
        , kwargs... )

  # Run contest after training, if it was not run in trainf! already
  epochs % interval == 0 ? nothing : cb(epochs)

end

