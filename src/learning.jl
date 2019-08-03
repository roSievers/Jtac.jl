# This module collects functions which help us train models.


# -------- Datasets ---------------------------------------------------------- #

# Methods: length, merge, split, augment, 
#          minibatch, save_dataset, load_dataset

# Stores the experience from one or more selfplays in an unstructured form.
# All labels starts with a number in {-1, 0, 1} to indicate the game result and
# label[2:end] contains the improved policy found by Monte Carlo tree search.
mutable struct DataSet{G <: Game}
  data :: Vector{G}
  label :: Vector{Vector{Float32}}
end

# TODO: make sure that data and label must always have the same length!
function DataSet{G}() where G <: Game
  DataSet(Vector{G}(), Vector{Vector{Float32}}())
end

Base.length(d :: DataSet) = length(d.data)

function Base.merge(d :: DataSet{G}, ds...) where G <: Game
  dataset = DataSet{G}()
  dataset.data = vcat([d.data, (x.data for x in ds)...]...)
  dataset.label = vcat([d.label, (x.label for x in ds)...]...)
  dataset
end

function Base.split(d :: DataSet{G}, size :: Int; shuffle = true) where {G}
  n = length(d)
  @assert size <= n
  idx = shuffle ? randperm(n) : 1:n
  idx1, idx2 = idx[1:size], idx[size+1:end]
  d1 = DataSet{G}(d.data[idx1], d.label[idx1])
  d2 = DataSet{G}(d.data[idx2], d.label[idx2])
  d1, d2
end

function augment(d :: DataSet{G}) :: DataSet{G} where G <: Game
  aux(data, label) = DataSet(augment(data, label)...)
  merge(aux.(d.data, d.label)...)
end

function Knet.minibatch( d :: DataSet{G}
                       , batchsize
                       ; shuffle = false
                       , partial = true
                       ) where {G}
  l = length(d)
  indices = shuffle ? Random.shuffle(1:l) : collect(1:l)
  batches = []
  i, j = 1, batchsize
  while max(i, j) <= l
    sel = indices[i:j]
    ds  = DataSet{G}(d.data[sel], d.label[sel])
    push!(batches, ds)
    i += batchsize
    j += partial ? min(batchsize, l - j) : batchsize
  end
  batches
end

function save_dataset(fname :: String, d :: DataSet)
  BSON.bson(fname * ".jtd", dataset = dataset) 
end

function load_dataset(fname :: String)
  BSON.load(fname * ".jtd")[:dataset]
end


# -------- Generating Datasets ----------------------------------------------- #

# Executes a selfplay and returns the Replay as a Dataset
function record_selfplay( model :: Model{G, GPU}
                        , n :: Int = 1
                        ; game :: T = G()
                        , power :: Int = 100
                        , temperature = 1.
                        , exploration = 1.41
                        , branching_prob = 0.  # Probability for random branching
                        , augment = true
                        , ntasks = 100
                        , callback :: Function = () -> nothing 
                        ) :: DataSet{T} where {G, T, GPU}

  @assert (T <: G) "Provided game does not fit model"

  rootgame = game

  function play()

    local game = copy(rootgame)
    branched_games = []
    dataset = DataSet{T}()

    while !is_over(game)

      # Random branching
      # With a certain probability we introduce a branching point with a random
      # move. This should help the network explore suboptimal situations better.
      if rand() <= branching_prob
        push!(branched_games, random_turn!(copy(game)))
      end

      # Record the current game state and do one mctree_turn
      push!(dataset.data, copy(game))
      actions = legal_actions(game)
      node = mctree_turn!( model
                         , game
                         , power = power
                         , temperature = temperature
                         , exploration = exploration )

      # The visit counters are stored in a dense array where each entry
      # corresponds to a legal move. We need the policy over all moves
      # including zeros for illegal moves. Here we do the transformation.

      # We also add a leading zero which correspond to the outcome prediction.
      posterior_distribution = node.visit_counter / sum(node.visit_counter)
      improved_policy = zeros(Float32, 1 + policy_length(game))
      improved_policy[actions .+ 1] .= posterior_distribution
      push!(dataset.label, improved_policy)
    end
    game_result = status(game)
    # We left the first entry for each label empty for the game result
    for i = 1:length(dataset.data)
      dataset.label[i][1] = current_player(dataset.data[i]) * game_result
    end

    # We now play all games which were created through random branching
    branch_datasets = map(branched_games) do branched_game
      record_selfplay( model
                     , 1
                     , game = branched_game
                     , power = power
                     , augment = false
                     , temperature = temperature
                     , exploration = exploration
                     , branching_prob = branching_prob )
    end
    callback()
    merge(dataset, branch_datasets...)
  end

  if !isa(model, Async) || n == 1
    sets = map(_ -> play(), 1:n)
  else
    sets = asyncmap(_ -> play(), 1:n, ntasks = ntasks)
  end

  if augment
    merge(sets...) |> Jtac.augment
  else
    merge(sets...)
  end
end


# -------- Loss -------------------------------------------------------------- #

function loss_components( model :: Model{G, GPU}
                        , dataset :: DataSet{G}
                        ) where {G, GPU}

  # Dataset size
  n = length(dataset)

  # Push the label matrix to the gpu if the model lives there
  at = atype(GPU)
  label = convert(at, hcat(dataset.label...))

  # Apply the model
  output = model(dataset.data)

  # Calculate the different loss components
  
  # Squared error loss for the value prediction
  value_loss = sum(abs2, output[1, :] .- label[1, :])

  # Cross entropy loss for the policy prediction
  policy_loss = -sum(label[2:end, :] .* log.(output[2:end, :]))

  # L2 regularization (weight decay)
  regularization_loss = sum(Knet.params(model)) do param
    s = size(param)
    maximum(s) < prod(s) ? sum(abs2, param) : 0f0
  end

  ( value = value_loss / n
  , policy = policy_loss / n
  , regularization = regularization_loss 
  )

end

function loss( model :: Model{G, GPU}, dataset :: DataSet{G};
               value_weight = 1f0, 
               policy_weight = 1f0, 
               regularization_weight = 0f0 ) where {G, GPU}

  # Convert all weights to Float32
  value_weight = convert(Float32, value_weight)
  policy_weight = convert(Float32, policy_weight)
  regularization_weight = convert(Float32, regularization_weight)

  l = loss_components(model, dataset)

  # Return the total loss
  value_weight * l.value +
  policy_weight * l.policy +
  regularization_weight * l.regularization
end

loss(model, data, label) = loss(model, DataSet([data], [label]))


# -------- Auxiliary Functions for Training ---------------------------------- #

# Set an optimizer for all parameters of a model
function set_optimizer!(model, opt = Knet.Adam; kwargs...)
  for param in Knet.params(model)
    param.opt = opt(; kwargs...)
  end
end

# A single training step, the loss is returned
function train_step!(model, dataset :: DataSet; kwargs...)
  tape = Knet.@diff loss(model, dataset; kwargs...)
  for param in Knet.params(model)
    Knet.update!(Knet.value(param), Knet.grad(tape, param), param.opt)
  end
  Knet.value(tape)
end


# -------- Printing Functions for Monitoring the Training -------------------- #

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

function print_contest_results(players, contest_length, async, active, cache)
    println(gray_crayon, "#\n# # --------- Contest -------------------------------- #\n#")

    r = length(active)
    k = length(players) - r
    n = (r * (r-1) + 2k*r)

    p = progressmeter(n + 1, "# Contest...")

    rk = ranking( players
                , contest_length
                , async = async
                , active = active
                , cache = cache
                , callback = () -> progress!(p) )

    clear_output!(p)

    print(gray_crayon)
    print_ranking(players, rk, prepend = "#")

    println("#")
end


# -------- Training ---------------------------------------------------------- #

"""
        train!(model; <keyword arguments>)

Train the neural network model `model` of type `NeuralModel{G}` via selfplay for
the game `G`. 

# Arguments

- `power :: Int = 100`: MCTS power used during selfplays.
- `epochs :: Int = 10`: Number of epochs used for training.
- `batchsize :: Int = 200`: Batchsize for iterating through the training data.
- `selfplays :: Int = 20`: Number of selfplays for dataset creation per epoch.
- `iterations :: Int = 1`: Number of iterations through the training data per epoch.
- `temperature :: Real = 1`: MCTS temperature used during selfplays.
- `exploration :: Real = 1.41`: MCTS exploration parameter.
- `branching_prob :: Real = 0`: Probability for random branching in selfplays.
- `augment :: Bool = true`: Augment the dataset after its generation by selfplays.
- `test_fraction :: Real = 0.1`: Fraction of the dataset used for testing.
- `policy_weight :: Real = 1.`: Weight for the policy-based loss term.
- `value_weight :: Real = 1.`: Weight for the value-based loss term.
- `regularization_weight :: Real = 1.`: Weight for the regularization-based loss term.
- `opponents = []`: List of players that the model has to face in contests.
- `contest_temperature :: Real = temperature`: Temperature used for contests.
- `contest_length :: Int = 250`: Max. number of games played during a contest.
- `contest_interval :: Int = 10`: Number of epochs between contests.
- `contest_cache :: Int = 0`: Number of cached games between `model`-independent
  players. If `contest_cache > 0`, the `model` will only play against 
  `model`-dependent opponents during contests and use results from the cache 
  for the total ranking.
- `no_contest :: Bool = false`: Deactivate contests during the training.
- `optimizer = Knet.Adam`: Optimization algorithm to use for update steps.

Other keyword arguments may be provided. They are fed to the `optimizer`.

# Examples

```julia
chain = @chain TicTacToe Conv(128, relu) Dense(100, relu)
model = NeuralModel(TicTacToe, chain)

opponents = [ MCTSPlayer(power = p) for p in [10, 50, 100, 500] ]

train!( model
      , power = 250
      , epochs = 10
      , batchsize = 50
      , selfplays = 200
      , iterations = 2
      , branching_prob = 0.1
      , value_weight = 10.
      , regularization_weight = 1e-5
      , contest_interval = 5
      , contest_length = 1000
      , contest_cache = 5000
      , opponents = opponents )
```

"""
function train!( model
               ; power = 100
               , epochs = 10
               , batchsize = 200
               , selfplays = 50
               , iterations = 1
               , temperature = 1.
               , exploration = 1.41
               , branching_prob = 0.
               , augment = true
               , test_fraction = 0.1
               , policy_weight = 1.
               , value_weight = 1.
               , regularization_weight = 0.
               , opponents = []
               , contest_temperature = temperature
               , contest_length :: Int = 250
               , contest_cache :: Int = 0
               , contest_interval :: Int = 10
               , no_contest :: Bool = false
               , optimizer = Knet.Adam
               , kwargs...
               )

  print( gray_crayon, "\n"
       , "# # --------- Options -------------------------------- #\n#\n"
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
       , format_option(:test_fraction, test_fraction)
       , format_option(:no_contest, no_contest)
       , format_option(:contest_length, contest_length)
       , format_option(:contest_cache, contest_cache)
       , format_option(:contest_temperature, contest_temperature)
       , format_option(:contest_interval, contest_interval) 
       , format_option(:opponents, join(name.(opponents), " "))
       , "#\n" )

  async = isa(model, Async)

  no_contest |= contest_length <= 0

  if !no_contest

    players = [
      IntuitionPlayer( model
                     , temperature = contest_temperature
                     , name = "current" );
      IntuitionPlayer( copy(model)
                     , temperature = contest_temperature
                     , name = "initial" );
      opponents
    ]

    if contest_cache > 0

      modelplayers = filter(p -> training_model(p) == training_model(model), players)
      otherplayers = filter(p -> training_model(p) != training_model(model), players)
      players = [otherplayers; modelplayers]

      active = collect(1:length(modelplayers)) .+ length(otherplayers)

      n = length(otherplayers) * (length(otherplayers) - 1)
      p = progressmeter(n + 1, "# Caching...")
      cache = playouts( otherplayers, contest_cache, callback = () -> progress!(p))
      clear_output!(p)
      println( gray_crayon
             , "# Cached $(length(cache)) matches from $(length(players)) players")

    else

      active = 1:length(players)
      cache = []

    end

    print_contest_results(players, contest_length, async, active, cache)
  end

  set_optimizer!(model, optimizer; kwargs...)

  println("# # --------- Training ------------------------------- #\n#")

  for i in 1:epochs

    # Data generation via selfplays

    p = progressmeter( selfplays + 1, "# Selfplays...")

    dataset = record_selfplay( model, selfplays
                             , power = power
                             , branching_prob = branching_prob
                             , augment = augment
                             , temperature = temperature
                             , exploration = exploration
                             , callback = () -> progress!(p) )

    clear_output!(p)

    testlength = round(Int, test_fraction * length(dataset))
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
    for (set, crayon)  in [(testset, ""), (trainset, gray_crayon)]
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

    if (i % contest_interval == 0 || i == epochs) && !no_contest
      print_contest_results(players, contest_length, async, active, cache)
    end
  end

  model

end

