# Monte Carlo implementation

mutable struct Node
    action :: ActionIndex             # How did we get here?
    parent :: Union{Node, Nothing}    # Where did we get here from?
    children :: Vector{Node}           # Where can we walk?
    visit_counter :: Vector{Float64}   # How often were children visited?
    expected_reward :: Vector{Float64} 
    model_policy :: Vector{Float64}    
end

Broadcast.broadcastable(node :: Node) = Ref(node)

function Node(action = 0, parent = nothing ) :: Node
    Node(action, parent, [], [], [], [])
end

# Find all children for a node and assesses them through the model.
# The state value predicted by the model is returned.
function expand!(node :: Node, game :: Game, model :: Model) :: Float64
  # We need to first check if the game is still active
  # and only evaluate the model on those games.
  if is_over(game)
    return status(game)*current_player(game) 
  end

  actions = legal_actions(game)
  value, policy = apply(model, game)

  # Initialize the vectors that will be filled with info about the children 
  node.children = Node.(actions, node)
  node.visit_counter = zeros(UInt, length(node.children))
  node.expected_reward = zeros(UInt, length(node.children))

  # Filter and normalize the policy vector returned by the network
  node.model_policy = policy[actions] / sum(policy[actions])
  value
end

# Helper function that returns true when a Node is a leaf
is_leaf(node :: Node) :: Bool = isempty(node.children)

# Traverse the tree from top to bottom and return a leaf
function descend_to_leaf!(game :: Game, node :: Node) :: Node
  while !is_leaf(node)
    best_i = findmax(confidence(node))[2]
  
    best_child = node.children[best_i]
    apply_action!(game, best_child.action)
  
    node = best_child
  end
  node
end

# The confidence in a node
# TODO: 1.41 should not be hardcoded
function confidence(node) :: Array{Float64}
  exploration_weight = 1.41 * sqrt(sum(node.visit_counter))
  result = zeros(Float64, length(node.children))
  for i = 1:length(node.children)
    exploration = exploration_weight * node.model_policy[i] / (1 + node.visit_counter[i])
    result[i] = node.expected_reward[i] + exploration
  end
  result
end

function expand_tree_by_one!(node, game, model)
  new_game = copy(game)
  new_node = descend_to_leaf!(new_game, node)
  value = expand!(new_node, new_game, model)
  # Backpropagate the negative value, since the parent calculates its expected
  # reward from it.
  backpropagate!(new_node, -value)
end

function backpropagate!(node, value) :: Nothing
  if node.parent != nothing
    # Since the parent keeps all child-information, we have to access it
    # indirectly
    parent = node.parent
    i = findfirst(x -> x === node, parent.children)

    # Update the reward and visit_counter
    counter = parent.visit_counter[i]
    reward = (parent.expected_reward[i] * counter + value) / (counter + 1)
    parent.expected_reward[i] = reward
    parent.visit_counter[i] += 1

    # Continue the backpropagation
    backpropagate!(parent, -value)
  end
end

function mctree_turn!(game :: Game; 
                      power = 100,
                      model = RolloutModel(game)) :: Node

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

function record_selfplay(game; power = 100, model = RolloutModel(game))
  node_list = Vector{Tuple{Game, Node}}()
  while !is_over(game)
    game_copy = copy(game)
    current_root = mctree_turn!(game, power = power, model = model)
    # Remove all children to save memory
    current_root.children = []
    push!(node_list, (game_copy, current_root))
  end
  node_list
end

function mctree_vs_random(game; power = 100, model = RolloutModel(game), tree_player = 1)
  @assert tree_player == 1 || tree_player == -1 "tree_player must be 1 or -1"
  game = copy(game)
  while !is_over(game)
    if current_player(game) == tree_player
      mctree_turn!(game, power = power, model = model)
    else
      random_turn!(game)
    end
  end
  status(game)
end
