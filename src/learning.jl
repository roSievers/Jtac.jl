# This module collects functions which help us train models.

# Stores a replay of a selfplay which we may use to train the model
mutable struct SelfplayReplay{G <: Game}
  games :: Vector{G}
  # A list of normalized node.visit_counter
  posterior_distributions :: Vector{Vector{Float32}}
  # Possible values are -1, 0, 1.
  game_result :: Float32
end

function SelfplayReplay{G}() where G <: Game
  SelfplayReplay(Vector{G}(), Vector{Vector{Float32}}(), Float32(0))
end

# Calculates the loss function for a single game moment from a selfplay.
function loss(game :: Game, posterior :: Vector{Float32}, result :: Float32, model :: Model) :: Float32
  value, policy = apply(model, game)
  value_loss = (value - result * game.current_player)^2
  cross_entropy_loss = - sum(posterior .* log.(policy))

  value_loss + cross_entropy_loss
end

# Calculates the loss function for a whole selfplay
function loss(replay :: SelfplayReplay, model :: Model)
  sum :: Float32 = 0
  for i = 1:length(replay.game)
    sum += loss(replay.game[i], replay.posterior_distributions[i], replay.game_result, model)
  end
  sum
end

# Executes a selfplay and returns the Replay
function record_selfplay(game :: G; power = 100, model = RolloutModel(game)) :: SelfplayReplay where G <: Game
  game = copy(game)
  replay = SelfplayReplay{G}()
  while !is_over(game)
    push!(replay.games, copy(game))
    actions = legal_actions(game)
    node = mctree_turn!(game, power = power, model = model)

    posterior_distribution = node.visit_counter / sum(node.visit_counter)
    improved_policy = zeros(81)
    improved_policy[actions] = posterior_distribution
    push!(replay.posterior_distributions, improved_policy)
  end
  replay.game_result = status(game)
  replay
end
