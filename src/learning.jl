# This module collects functions which help us train models.

# Stores the experience from one or more selfplays in an unstructured form.
# All labels starts with a number in {-1, 0, 1} to indicate the game result and
# label[2:end] contains the improved policy found by Monte Carlo tree search.
mutable struct DataSet{G <: Game}
  data :: Vector{G}
  label :: Vector{Vector{Float32}}
end

function DataSet{G}() where G <: Game
  DataSet(Vector{G}(), Vector{Vector{Float32}}())
end

# Calculates the loss function for a single data point.
function loss(data :: Game, label :: Vector{Float32}, model :: Model)
  output = model(data)
  value_loss = (output[1] - label[1])^2
  cross_entropy_loss = -sum(label[2:end] .* log.(output[2:end]))

  value_loss + cross_entropy_loss
end

# Calculates the loss function for a whole data set.
function loss(dataSet :: DataSet, model :: Model)
  sum :: Float32 = 0
  for i = 1:length(dataSet.data)
    sum += loss(dataSet.data[i], dataSet.label[i], model)
  end
  sum
end

# Executes a selfplay and returns the Replay as a Dataset
function record_selfplay(game :: G; power = 100, model = RolloutModel(game)) :: DataSet{G} where G <: Game
  game = copy(game)
  dataSet = DataSet{G}()
  while !is_over(game)
    push!(dataSet.data, copy(game))
    actions = legal_actions(game)
    node = mctree_turn!(game, power = power, model = model)

    # The visit counters are stored in a dense array where each entry
    # corresponds to a legal move. We need the policy over all moves
    # including zeros for illegal moves. Here we do the transformation.
    # We also add a leading zero which correspond to the outcome prediction.
    posterior_distribution = node.visit_counter / sum(node.visit_counter)
    improved_policy = zeros(1 + policy_length(game))
    improved_policy[actions .+ 1] = posterior_distribution
    push!(dataSet.label, improved_policy)
  end
  game_result = status(game)
  # We left the first entry for each label empty for the game result
  for i = 1:length(dataSet.data)
    dataSet.label[i][1] = current_player(dataSet.data[i]) * game_result
  end
  dataSet
end
