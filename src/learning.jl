# This module collects functions which help us train models.

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

function Base.merge(d :: DataSet{G}, ds...) where G <: Game
  dataset = DataSet{G}()
  dataset.data = vcat([d.data, (x.data for x in ds)...]...)
  dataset.label = vcat([d.label, (x.label for x in ds)...]...)
  dataset
end

Base.length(d :: DataSet) = length(d.data)

function augment2(data, label)
  DataSet(augment(data, label)...)
end

function augment(d :: DataSet{G}) :: DataSet{G} where G <: Game
  merge(augment2.(d.data, d.label)...)
end

# Calculates the loss function for a single data point.
function loss(model :: Model, data :: Game, label :: Vector{Float32})
  output = model(data)
  value_loss = (output[1] - label[1])^2
  cross_entropy_loss = -sum(label[2:end] .* log.(output[2:end]))

  value_loss + cross_entropy_loss
end

# Calculates the loss function for a whole data set.
function loss(model :: Model, dataset :: DataSet)
  sum = 0
  for i = 1:length(dataset.data)
    sum += loss(model, dataset.data[i], dataset.label[i])
  end
  sum
end

# Executes a selfplay and returns the Replay as a Dataset
function record_selfplay(startgame :: G, n = 1; power = 100, model = RolloutModel(startgame)) :: DataSet{G} where G <: Game
  sets = map(1:n) do _
    game = copy(startgame)
    dataset = DataSet{G}()
    while !is_over(game)
      push!(dataset.data, copy(game))
      actions = legal_actions(game)
      node = mctree_turn!(game, power = power, model = model)

      # The visit counters are stored in a dense array where each entry
      # corresponds to a legal move. We need the policy over all moves
      # including zeros for illegal moves. Here we do the transformation.

      # We also add a leading zero which correspond to the outcome prediction.
      posterior_distribution = node.visit_counter / sum(node.visit_counter)
      improved_policy = zeros(1 + policy_length(game))
      improved_policy[actions .+ 1] = posterior_distribution
      push!(dataset.label, improved_policy)
    end
    game_result = status(game)
    # We left the first entry for each label empty for the game result
    for i = 1:length(dataset.data)
      dataset.label[i][1] = current_player(dataset.data[i]) * game_result
    end
    dataset
  end
  merge(sets...)
end
