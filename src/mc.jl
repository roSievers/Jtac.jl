# Monte Carlo implementation

mutable struct Node
    action :: ActionIndex             # How did we get here?
    parent :: Union{Node, Nothing}    # Where did we get here from?
    children :: Array{Node}           # Where can we walk?
    visit_counter :: Array{Float64}   # How often were children visited?
    expected_reward :: Array{Float64} 
    model_policy :: Array{Float64}    
end

Broadcast.broadcastable(node :: Node) = Ref(node)

function Node(action = 0, parent = nothing ) :: Node
    Node(action, parent, [], [], [], [])
end

# Find all children for a node and assesses them through the model.
# The state value predicted by the model is returned.
function expand!(node :: Node, game :: Game, model :: Model) :: Float64
    actions = legal_actions(game)
    policy = apply(model, game)

    if isempty(actions)
        return policy[1]
    end
    node.children = Node.(actions, node)
    node.visit_counter = zeros(UInt, length(node.children))
    node.expected_reward = zeros(UInt, length(node.children))
    # Filtern und renormieren
    node.model_policy = policy[2:end][actions] / sum(policy[2:end][actions])
    policy[1]
end

# Helper function whis is true, when a Node is a leaf
function is_leaf(node :: Node) :: Bool
    isempty(node.children)
end

# Traverse the tree from top to bottom and return a root leaf
function descend_to_leaf!(game :: Game, node :: Node) :: Node
    while length(node.children) != 0
        best_i = findmax(confidence(node))[2]
    
        best_child = node.children[best_i]
        apply_action!(game, best_child.action)
    
        node = best_child
    end
    return node
end

function confidence(node) :: Array{Float64}
    exploration_weight :: Float64 = 1.41 :: Float64 * sqrt(sum(node.visit_counter) :: Float64)
    result = zeros(Float64, length(node.children))
    for i = 1:length(node.children)
        result[i] = node.expected_reward[i] + exploration_weight * node.model_policy[i] / (1 + node.visit_counter[i])
    end
    result
end

function expand_tree_by_one!(node, game, model)
    new_game = copy(game)
    new_node = descend_to_leaf!(new_game, node)
    state_value = expand!(new_node, new_game, model)
    # We backpropagate the negative value, as the parent calculates its expected reward from it.
    backpropagate!(new_node, -state_value)
end

function backpropagate!(node, value) :: Nothing
    if node.parent == nothing
        return
    end
    parent = node.parent
    i = findfirst(x -> x === node, parent.children)
    parent.expected_reward[i] = (parent.expected_reward[i] * parent.visit_counter[i] + value) / (parent.visit_counter[i] + 1)
    parent.visit_counter[i] += 1
    backpropagate!(parent, -value)
end

function mctree_turn!(game :: Game, power = 1000) :: Node
    model = RolloutModel(game)
    root = Node()
    for i = 1:power
        expand_tree_by_one!(root, game, model)
    end
    
    # TODO: The paper states, that during self play we pick a move from the
    # improved stochastic policy root.visit_counter at random.
    # Note that visit_counter is generally prefered over expected_reward
    # when choosing the best move in a match, as it is less susceptible to
    # random fluctuations.
    best_i = findmax(root.visit_counter)[2]
    best_child = root.children[best_i]
    apply_action!(game, best_child.action)
    root
end

function record_selfplay(game, power = 100)
    node_list = Vector{Tuple{Game, Node}}()
    while !is_over(game)
        game_copy = copy(game)
        current_root = mctree_turn!(game, power)
        # Remove all children to save memory
        current_root.children = []
        push!(node_list, (game_copy, current_root))
    end
    node_list
end

function mctree_vs_random(game, power = 100)
  game = copy(game)
  while !is_over(game)
    if game.current_player == 1
      mctree_turn!(game, power)
    else
      random_turn!(game)
    end
  end
  status(game)
end
