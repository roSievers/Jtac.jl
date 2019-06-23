# Monte Carlo implementation

mutable struct Node
    action :: ActionIndex             # How did we get here?
    parent :: Union{Node, Nothing}    # Where did we get here from?
    children :: Vector{Node}           # Where can we walk?
    visit_counter :: Vector{Float32}   # How often were children visited?
    expected_reward :: Vector{Float32} 
    model_policy :: Vector{Float32}    
end

Broadcast.broadcastable(node :: Node) = Ref(node)

function Node(action = 0, parent = nothing ) :: Node
    Node(action, parent, [], [], [], [])
end

# Find all children for a node and assesses them through the model.
# The state value predicted by the model is returned.
function expand!(node :: Node, game :: Game, model :: Model) :: Float32
  # We need to first check if the game is still active
  # and only evaluate the model on those games.
  if is_over(game)
    return status(game)*current_player(game) 
  end

  actions = legal_actions(game)
  value, policy = apply(model, game)

  # Initialize the vectors that will be filled with info about the children 
  node.children = Node.(actions, node)
  node.visit_counter = zeros(Float32, length(node.children))
  node.expected_reward = zeros(Float32, length(node.children))

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
function confidence(node) :: Array{Float32}
  exploration_weight = 1.41 * sqrt(sum(node.visit_counter))
  result = zeros(Float32, length(node.children))
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

# Runs the mcts algorithm, expanding the given root node.
# This function can be used directy instead of mctree_action if you need
# detailed information about the mcts decision.
function run_mcts(model, game :: Game;
                          root = Node(), # To track the expansion
                          power = 100,
                          temperature = 1.)
  
  # Expand the root node
  for i = 1:power
    expand_tree_by_one!(root, game, model)
  end
end

function mctree_action(model, game :: Game;
                       root = Node(), # To track the expansion
                       power = 100,
                       temperature = 1.) :: ActionIndex
  run_mcts(
    model,
    game,
    root = root,
    power = power,
    temperature = temperature
    )
  
  # The paper states, that during self play we pick a move from the
  # improved stochastic policy root.visit_counter at random.
  # Note that visit_counter is generally prefered over expected_reward
  # when choosing the best move in a match, as it is less susceptible to
  # random fluctuations.
  # TODO: We now also draw from root.visit_counter in real playthroughts,
  # not only during learning. Think about this!
  if temperature == 0
    chosen_i = findmax(root.visit_counter)[2]
  else
    weighted_counter = root.visit_counter.^(1/temperature)
    probs = weighted_counter / sum(weighted_counter)
    chosen_i = choose_index(probs)
  end
  
  root.children[chosen_i].action
end

function mctree_turn!(model, game :: Game; 
                      power = 100,
                      temperature = 1.,) :: Node

  root = Node()
  action = mctree_action(model, game, power = power, 
                         temperature = temperature, root = root)
  apply_action!(game, action)
  root
end

