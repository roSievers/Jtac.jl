
# -------- MCTS Nodes -------------------------------------------------------- #

mutable struct Node
    action :: ActionIndex              # How did we get here?
    current_player :: Int              # Who is allowed an action in this situation?
    parent :: Union{Node, Nothing}     # Where did we get here from?
    children :: Vector{Node}           # Where can we walk?
    visit_counter :: Vector{Float32}   # How often were children visited?
    expected_reward :: Vector{Float32} 
    model_policy :: Vector{Float32}    
end

Broadcast.broadcastable(node :: Node) = Ref(node)

Node(action = 0, parent = nothing ) = Node(action, 0, parent, [], [], [], [])

is_leaf(node :: Node) :: Bool = isempty(node.children)

"""
Find all children for a node and assesses them through the model.
The state value predicted by the model is returned.
The value is from the perspective of the first player
"""
function expand!(node :: Node, game :: Game, model :: Model) :: Float32
  node.current_player = current_player(game)

  # We need to first check if the game is still active
  # and only evaluate the model on those games.
  if is_over(game)
    return status(game)
  end

  actions = legal_actions(game)
  # This value variable is from the perspective of current_player(game)
  value, policy = apply(model, game)

  # Initialize the vectors that will be filled with info about the children 
  node.children = Node.(actions, node)
  node.visit_counter = zeros(Float32, length(node.children))
  node.expected_reward = zeros(Float32, length(node.children))

  # Filter and normalize the policy vector returned by the network
  node.model_policy = policy[actions] / sum(policy[actions])
  value * current_player(game)
end


# -------- MCTS Algorithm --------------------------------------------------- #

# The confidence in a node
function confidence(node; exploration = 1.41) :: Array{Float32}

  result = zeros(Float32, length(node.children))
  weight = exploration * sqrt(sum(node.visit_counter))

  for i = 1:length(node.children)
    exploration = weight * node.model_policy[i] / (1 + node.visit_counter[i])
    result[i] = node.expected_reward[i] + exploration
  end

  result

end

"""
Traverse the tree from top to bottom and return a leaf.
While this is done, the game is mutated.
"""
function descend_to_leaf!( game :: Game
                         , node :: Node
                         ; exploration = 1.41
                         ) :: Node

  while !is_leaf(node)
    best_i = findmax(confidence(node, exploration = exploration))[2]
  
    best_child = node.children[best_i]
    apply_action!(game, best_child.action)
  
    node = best_child
  end

  node

end

function backpropagate!(node, p1_value) :: Nothing

  if node.parent != nothing
    # Since the parent keeps all child-information, we have to access it
    # indirectly
    parent = node.parent
    i = findfirst(x -> x === node, parent.children)

    # Update the reward and visit_counter
    counter = parent.visit_counter[i]
    parent_value = p1_value * parent.current_player
    reward = (parent.expected_reward[i] * counter + parent_value) / (counter + 1)
    parent.expected_reward[i] = reward
    parent.visit_counter[i] += 1

    # Continue the backpropagation
    backpropagate!(parent, p1_value)
  end

end

function expand_tree_by_one!(node, game, model; exploration = 1.41)
  new_game = copy(game)
  # Decending to a leaf also changes the new_game value. This value then
  # corresponds to the game state at the leaf.
  new_node = descend_to_leaf!(new_game, node, exploration = exploration)
  p1_value = expand!(new_node, new_game, model)

  # Backpropagate the negative value, since the parent calculates its expected
  # reward from it.
  backpropagate!(new_node, p1_value)
end

# Runs the mcts algorithm, expanding the given root node.
function run_mcts( model
                 , game :: Game
                 ; root = Node()      # To track the expansion
                 , power = 100
                 , temperature = 1.
                 , exploration = 1.41 )
  root.current_player = current_player(game)
  
  for i = 1:power
    expand_tree_by_one!(root, game, model, exploration = exploration)
  end

end

function mctree_policy( model, game :: Game
                      ; root = Node()
                      , power = 100
                      , temperature = 1.
                      , exploration = 1.41 ) :: Vector{Float32}

  run_mcts( model
          , game
          , root = root
          , power = power
          , temperature = temperature
          , exploration = exploration )

  
  # The paper states, that during self play we pick a move from the
  # improved stochastic policy root.visit_counter at random.
  # Note that visit_counter is generally prefered over expected_reward
  # when choosing the best move in a match, as it is less susceptible to
  # random fluctuations.
  # TODO: We now also draw from root.visit_counter in real playthroughts,
  # not only during learning. Think about this!

  if temperature == 0
    probs = zeros(Float32, length(root.visit_counter))
    probs[findmax(root.visit_counter)[2]] = 1
  else
    weighted_counter = (root.visit_counter/power).^(1/temperature)
    probs = weighted_counter / sum(weighted_counter)
  end

  probs

end

function mctree_action( model
                      , game :: Game
                      ; root = Node()
                      , power = 100
                      , temperature = 1.
                      , exploration = 1.41 
                      ) :: ActionIndex

  probs = mctree_policy( model
                       , game
                       , root = root
                       , power = power
                       , temperature = temperature
                       , exploration = exploration )
  
  chosen_i = choose_index(probs)
  root.children[chosen_i].action

end

function mctree_turn!( model
                     , game :: Game
                     ; power = 100
                     , temperature = 1.
                     , exploration = 1.41
                     ) :: Node

  root = Node()
  action = mctree_action( model
                        , game
                        , root = root
                        , power = power
                        , temperature = temperature
                        , exploration = exploration )

  apply_action!(game, action)
  root

end

