# This module collects functions which help us train models.

# Stores a replay of a selfplay which we may use to train the model
mutable struct SelfplayRecord{G <: Game}
  game_states :: Vector{G}
  # A list of normalized node.visit_counter
  posterior_distributions :: Vector{Vector{Float64}}
  # Possible values are -1, 0, 1.
  game_result :: Float64
end

# Calculates the loss function for a single game moment from a selfplay.
function loss(game_state :: Game, posterior :: Vector{Float64}, result :: Float64, model :: Model) :: Float64
  value, policy = apply(model, game)
  value_loss = (value - result * game_state.current_player)^2
  cross_entropy_loss = - posterior * log.(policy)

  value_loss + cross_entropy_loss
end

# Calculates the loss function for a whole selfplay
function loss(replay :: SelfplayReplay, model :: Model)
  sum :: Float64 = 0
  for i = 1:size(replay.game_states)
    sum += loss(replay.game_states[i], replay.posterior_distributions[i], replay.game_result, model)
  end
  sum
end

# Executes a selfplay and returns the Replay
function record_selfplay(game; power = 100, model = RolloutModel(game)) :: SelfplayReplay
  game = copy(game)
  replay = SelfplayReplay()
  while !is_over(game)
    push!(replay.game_states, copy(game))
    node = mctree_turn!(game, power = power, model = model)
    posterior_distribution = node.visit_counter / sum(node.visit_counter)
    push!(replay.posterior_distributions, posterior_distribution)
  end
  replay.game_result = status(game)
  replay
end