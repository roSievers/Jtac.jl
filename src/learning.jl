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


function loss( model :: Model{G, GPU}, dataset :: DataSet{G};
               value_weight = 1f0, policy_weight = 1f0, 
               regularization_weight = 0f0 ) where {G, GPU}

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
  regularization_loss = 0f0
  if regularization_weight >= 0f0
    for param in Knet.params(model)
      regularization_loss += sum(abs2, param)
    end
  end

  # Return the total loss
  value_weight * value_loss +
  policy_weight * policy_loss +
  regularization_weight * regularization_loss
end

loss(model, data, label) = loss(model, DataSet([data], [label]))

# Executes a selfplay and returns the Replay as a Dataset
function record_selfplay( model :: Model{G, GPU}, n = 1; 
                          game :: T = G(),
                          power = 100, 
                          temperature = 1.,
                          branch_prob = 0.,  # Probability for random branching
                          augment = true ) :: DataSet{T} where {G, T, GPU}

  @assert (T <: G) "Provided game does not fit model"

  rootgame = game
  sets = map(1:n) do _
    branched_games = []
    game = copy(rootgame)
    dataset = DataSet{T}()
    while !is_over(game)

      # Random branching
      # With a certain probability we introduce a branching point with a random
      # move. This should help the network explore suboptimal situations better.
      if rand() <= branch_prob
        push!(branched_games, random_turn!(copy(game)))
      end

      # Record the current game state and do one mctree_turn
      push!(dataset.data, copy(game))
      actions = legal_actions(game)
      node = mctree_turn!(model, game, power = power, temperature = temperature)

      # The visit counters are stored in a dense array where each entry
      # corresponds to a legal move. We need the policy over all moves
      # including zeros for illegal moves. Here we do the transformation.

      # We also add a leading zero which correspond to the outcome prediction.
      posterior_distribution = node.visit_counter / sum(node.visit_counter)
      improved_policy = zeros(Float32, 1 + policy_length(game))
      improved_policy[actions .+ 1] = posterior_distribution
      push!(dataset.label, improved_policy)
    end
    game_result = status(game)
    # We left the first entry for each label empty for the game result
    for i = 1:length(dataset.data)
      dataset.label[i][1] = current_player(dataset.data[i]) * game_result
    end
    # We now play all games which were created through random branching
    branch_datasets = map(branched_games) do branched_game
      record_selfplay(model, 1, game = branched_game, power = power, augment = false, 
                      temperature = temperature, branch_prob = branch_prob)
    end
    merge(dataset, branch_datasets...)
  end
  if augment
    merge(sets...) |> Jtac.augment
  else
    merge(sets...)
  end
end

# Set an optimizer for all parameters of a model
function set_optimizer!(model, opt = Knet.Adam; kwargs...)
  for param in Knet.params(model)
    param.opt = opt(; kwargs...)
  end
end

# A single training step, the loss is returned
function train_step!(model, dataset :: DataSet)
  tape = Knet.@diff loss(model, dataset)
  for param in Knet.params(model)
    Knet.update!(Knet.value(param), Knet.grad(tape, param), param.opt)
  end
  Knet.value(tape)
end


