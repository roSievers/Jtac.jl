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
    parent :: Nullable{Node}
    children :: Array{Node}
    visit_counter :: Array{UInt} # Wie oft wurden die Kinder besucht?
    expected_reward :: Array{Float64}
    model_policy :: Array{Float64}
end

function Node(action = 0, parent = nothing ) :: Node
    if parent == nothing
        Node(action, Nullable{Node}(), [], [], [], [])
    else
        Node(action, Nullable(parent), [], [], [], [])
    end
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
    node.model_policy = policy[2][actions] / sum(policy[2][actions])
    policy[1]
end

function is_leaf(node :: Node)
    isempty(node.children)
end

function descend_to_leaf!(game :: GameState, node :: Node) :: Node
    if is_leaf(node)
        return node
    end

    best_i = indmax(confidence(node))

    best_child = node.children[best_i]
    place!(game, best_child.action)

    descend_to_leaf!(game, best_child)
end

function confidence(node) :: Array{Float64}
    visit_total :: Float64 = sum(node.visit_counter)
    node.expected_reward .+ 1.41 * node.model_policy .* sqrt( visit_total ) ./ (1 + node.visit_counter)
end

function expand_tree_by_one!(node, game, model)
    new_game = copy(game)
    new_node = descend_to_leaf!(new_game, node)
    state_value = expand!(new_node, new_game, model)
    backpropagate!(new_node, state_value)
end

function backpropagate!(node, value) :: Void
    if isnull(node.parent)
        return
    end
    parent = get(node.parent)
    i = findfirst(x -> x === node, parent.children)
    parent.expected_reward[i] = (parent.expected_reward[i] * parent.visit_counter[i] + value) / (parent.visit_counter[i] + 1)
    parent.visit_counter[i] += 1
    backpropagate!(parent, -value)
end