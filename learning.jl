
# This function is intend to remove all children from the root node of the tree.
# This way the garbage collector can free most memory and we keep all the
# information we need for the learning step.
# The function is called collapse, because it is the opposite of expand.
function collapse!(root :: Node)
    root.children = []
end

# The loss function for a single move of the game. 
function loss_at_node(node :: Node, game :: GameState, game_result :: Float64, policy :: Tuple{Float64, Array{Float64}}) :: Float64
    # Comparing the value prediction of the model to the actual game result.
    value_loss = (policy[1] - game_result) ^ 2
    # Comparing the prediction of the policy network to the improved policy
    # found through Monte Carlo tree search.
    actions = legal_actions(game)
    action_policy = policy[2][actions] / sum(policy[2][actions])
    improved_policy = node.visit_counter / sum(node.visit_counter)
    cross_entropy_loss = - improved_policy' * log.(action_policy)
    
    value_loss + cross_entropy_loss
end

# Argument wrangling
function loss_at_node(replay :: Tuple{GameState, Node}, game_result :: Float64, model) :: Float64
    loss_at_node(replay[2], replay[1], game_result, apply(model, replay[1]))
end

# The loss function for one selfplay
function loss(replay :: Vector{Tuple{GameState, Node}}, game_result :: Float64, model) :: Float64
    sum(loss_at_node.(replay, game_result, model))
end

function usage_example()
    l = record_selfplay()
    loss(l, 1.0, RolloutModel())
end