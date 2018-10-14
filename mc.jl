# Monte Carlo implementation

# Knoten
#     StateValueEstimate
#     Array{Unterknoten}

# Unterknoten
#     Knoten, vielleicht
#     ExpectedReward Q
#     Besuchszahl N
#     ActionValueEstimate P = 

mutable struct Node
    action :: Int8 # How we got here
    parent :: Union{Node, Nothing}
    children :: Array{Node}
    visit_counter :: Array{Float64} # Wie oft wurden die Kinder besucht?
    expected_reward :: Array{Float64}
    model_policy :: Array{Float64}
end

Broadcast.broadcastable(node :: Node) = Ref(node)

function Node(action = 0, parent = nothing ) :: Node
    Node(action, parent, [], [], [], [])
end

# Findet für einen Node alle zugelassenen Unterknoten und bewertet sie mit
# dem übergebenen Model. The state value from the Model ist returned.
function expand!(node :: Node, game :: GameState, model :: Model) :: Float64
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
function descend_to_leaf!(game :: GameState, node :: Node) :: Node
    while length(node.children) != 0
        best_i = findmax(confidence(node))[2]
    
        best_child = node.children[best_i]
        place!(game, best_child.action)
    
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

function ai_turn!(game :: GameState, power = 1000) :: Node
    model = RolloutModel()
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
    place!(game, best_child.action)
    root
end

function record_selfplay(power = 100)
    game = new_game()
    node_list = Vector{Tuple{GameState, Node}}()
    while !game_result(game)[1]
        game_copy = copy(game)
        current_root = ai_turn!(game, power)
        # Remove all children to save memory
        current_root.children = []
        push!(node_list, (game_copy, current_root))
    end
    node_list
end