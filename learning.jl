
# This function is intend to remove all children from the root node of the tree.
# This way the garbage collector can free most memory and we keep all the
function collapse!(root :: Node)
    root.children = []
end

struct PreparedLossFunction
    improved_policy :: Vector{Float32},
    game_result :: Float32,
    legal_actions :: Vector[Int8]
end

const VectorGame = Vector{Any}

# Translates the game into a low level representation used by the model.
function VectorGame(state :: GameState) :: VectorGame
    board = reshape(game.board, (9, 9))
    mask = reshape(game.legal_actions_mask, (9, 9))
    Any[ game.current_player * board, mask ]
end

function VectorGame(states :: Vector{GameState}) :: VectorGame
  vgames = VectorGame.(states)
  Any[ cat(3, first.(vgames)...), cat(3, second.(vgames)...) ]
end

# Translates from board coordinates to visual coordinates.
# Visual coordinates are as follows, where a = 10, b = 11, ...
# 1 2 3 4 5 6 7 8 9
# a b c d e f g h i
# j k l m n o p q r
# s t u v w x y z ...
function transfigure_board_shape(original_shape :: Vector{Float32}) :: Vector{Float32}
    network_shape = zeros(81)
    for x_o = 1:3, x_i = 1:3, y_o = 1:3, y_i = 1:3
        original_coord = combined_index(y_o, x_o, y_i, x_i)
        network_coord = x_i + 3 * (x_o - 1) + 9 * (y_i - 1) + 27 * (y_o - 1)
        network_shape[network_coord] = original_shape[original_coord]
    end
end

# TODO: Komplett neu schreiben
function loss(weights, state, prepared_loss_function :: PreparedLossFunction)
    value_loss = (prepared_loss_function.game_result - ???) ^ 2
    model_policy = ???[prepared_loss_function.legal_actions]
    cross_entropy_loss = - prepared_loss_function.improved_policy' * log.(model_policy)

    value_loss + cross_entropy_loss
end

# Returns (s_t, \pi_t, z_t), where
#  - s_t is the current game state
#  - \pi_t is the improved estimate of the policy from state s_t
#  - z_t is the result of the selfplay
function convert_lists_to_masks(instant :: Tuple{GameState, Node}, game_result :: Float32)
    actions = legal_actions(instant[1])
    improved_policy = zeros(81)
    improved_policy[actions] = instant[2].visit_counter / sum(instant[2].visit_counter)

    (instant[1].board * instant[1].current_player, improved_policy, game_result * instant[1].current_player)
end

function prepare_loss_function(replay :: Vector{Tuple{GameState, Node}}) :: PreparedLossFunction
    masks = convert_lists_to_masks.(replay)

    legal_actions_mask = hcat(get_legal_action_mask.(first.(replay))...)
    # Sperre

    improved_policy = second.(replay)
end

# The loss function for a single move of the game.
function loss_at_node(node :: Node, game :: GameState, game_result :: Float32, value_policy :: Float32, move_policy :: Array{Float32}}) :: Float64
    # Comparing the value prediction of the model to the actual game result.
    value_loss = (value_policy - game_result) ^ 2
    # Comparing the prediction of the policy network to the improved policy
    # found through Monte Carlo tree search.
    actions = legal_actions(game)
    action_policy = policy[2][actions] / sum(policy[2][actions])
    improved_policy = node.visit_counter / sum(node.visit_counter)
    cross_entropy_loss = - improved_policy' * log.(action_policy)

    value_loss + cross_entropy_loss
end

# Argument wrangling
function loss_at_node(model, replay :: Tuple{GameState, Node}, game_result :: Float64) :: Float64
    loss_at_node(replay[2], replay[1], game_result, apply(model, replay[1]))
end

# The loss function for one selfplay
function loss(model, replay :: Vector{Tuple{GameState, Node}}, game_result :: Float64) :: Float64
    sum(loss_at_node.(model, replay, game_result))
end

function usage_example()
    l = record_selfplay()
    loss(l, 1.0, RolloutModel())
end
