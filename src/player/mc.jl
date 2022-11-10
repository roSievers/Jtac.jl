
# -------- MCTS Nodes -------------------------------------------------------- #

mutable struct Node
    action :: ActionIndex              # How did we get here?
    current_player :: Int              # Who is allowed an action in this situation?
    parent :: Union{Node, Nothing}     # Where did we get here from?
    children :: Vector{Node}           # Where can we walk?
    visit_counter :: Vector{Float32}   # How often were children visited?
    expected_reward :: Vector{Float32} # The expected result of the various children
    model_policy :: Vector{Float32}    # Policy prior of the model
end

Node(action = 0, parent = nothing ) = Node(action, 0, parent, [], [], [], [])
Broadcast.broadcastable(node :: Node) = Ref(node)

is_leaf(node :: Node) :: Bool = isempty(node.children)

# -------- MCTS Algorithm --------------------------------------------------- #


# Small perturbation constant for breaking symmerties in the MCTS algorithm by
# injecting a little dose of randomness
const mcts_eps = 1f-6

# Logit normal distribution used to dilute the prior policy distribution
# at the root node
function logit_normal(n)
  ey = exp.(randn(Float32, n))
  ey / sum(ey)
end

# Find the child index with maximal confidence, based on a policy informed upper confidence bound (puct)
function max_puct_index(node, exploration :: Float32) :: Int

  max_i = 1
  max_puct = -Inf32

  # Precalculate the exploration weight. The constant mcts_eps is added to make
  # sure that the model_policy matters for finding the maximum even if
  # visit_counter only contains zeros.
  weight = exploration * sqrt(sum(node.visit_counter)) + mcts_eps

  for i in 1:length(node.children)
    explore = weight * node.model_policy[i] / (1 + node.visit_counter[i])
    puct = node.expected_reward[i] + explore
    if puct > max_puct
      max_i = i
      max_puct = puct
    end
  end

  max_i
end

# Traverse the tree from top to bottom and return a leaf.
# While this is done, the game is mutated.
function descend_to_leaf!( game :: AbstractGame
                         , node :: Node
                         ; exploration ) :: Node

  while !is_leaf(node)
    best_i = max_puct_index(node, exploration)

    best_child = node.children[best_i]
    apply_action!(game, best_child.action)

    node = best_child
  end

  node
end

function backpropagate!(node, p1_value) :: Nothing

  if !isnothing(node.parent)
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

"""
Find all children for a node and assesses them through the model.
The state value predicted by the model is returned.
The value is from the perspective of the first player
"""
function expand!(node :: Node, game :: AbstractGame, model :: AbstractModel; dilution :: Float32) :: Float32
  node.current_player = current_player(game)

  # We need to first check if the game is still active
  # and only evaluate the model on those games.
  is_over(game) && return status(game)

  actions = legal_actions(game)
  # This value variable is from the perspective of current_player(game)
  value, policy = apply(model, game)

  # Initialize the vectors that will be filled with info about the children 
  node.children = Node.(actions, node)
  node.visit_counter = zeros(Float32, length(node.children))
  node.expected_reward = zeros(Float32, length(node.children))

  # Filter and normalize the policy vector returned by the network. Addition of
  # mcts_eps prevents divisions by zero in pathological cases (note that
  # model_policy does not strictly have to be a probability distribution for the
  # MCTS steps, since it is only used for calculation of the puct value).
  node.model_policy = policy[actions] / (sum(policy[actions]) + mcts_eps) 

  # At the root node, we add noise to the policy prior to facilitate finding
  # unexpected but possibly good moves
  # Note: In the original alpha zero implementation (and its imitations),
  # dirichlet noise (instead of logit normal) is used.
  if isnothing(node.parent) && dilution > 0
    noise = logit_normal(length(node.model_policy))
    node.model_policy .= (1-dilution) * node.model_policy + dilution * noise

  # Even if we do not want to dilute the policy, we add a tiny bit of noise
  # for symmetry breaking.
  # Reasoning: if the model is a rollout model, model_policy is uniformly
  # distributed. This would cause situations in which the maximal puct is with
  # non-negligible probability not unique. Adding some noise makes it almost
  # surely unique.
  else
    noise = rand(Float32, length(node.model_policy))
    node.model_policy .= node.model_policy .+ mcts_eps * noise
  end

  value * current_player(game)
end

function expand_tree_by_one!(node, game, model; exploration, dilution)
  new_game = copy(game)
  # Descending to a leaf also changes the new_game value. This value then
  # corresponds to the game state at the leaf.
  new_node = descend_to_leaf!(new_game, node; exploration)
  p1_value = expand!(new_node, new_game, model; dilution)

  # Backpropagate the negative value, since the parent calculates its expected
  # reward from it.
  backpropagate!(new_node, p1_value)
end

"""
    run_mcts(model, game; [root, power = 100, exploration = 1.41, dilution = 0.0])

Run the MCTS algorithm for `game` using `model` for policy priors and value
evaluations. Returns the expanded `root` node.
"""
function run_mcts( model
                 , game :: AbstractGame
                 ; root :: Node = Node()      # To track the expansion
                 , power = 100
                 , exploration = 1.41
                 , dilution = 0.0
                 ) :: Node

  root.current_player = current_player(game)

  exploration = Float32(exploration)
  dilution = Float32(dilution)

  for i = 1:power
    expand_tree_by_one!(root, game, model; exploration, dilution)
  end

  root
end


# -------- MCTS Value and Policy --------------------------------------------- #

function one_hot(n, k)
  r = zeros(Float32, n)
  r[k] = 1f0
  r
end

# Auxiliary function to sharpen / smoothen a probability distribution
function apply_temperature(values :: Vector{Float32}, temp :: Float32)
  if temp == 0f0
    one_hot(length(values), findmax(values)[2])
  elseif temp == 1f0
    values / sum(values)
  else
    weights = values.^(1/temp)
    weights / sum(weights)
  end
end

function mcts_policy( model
                    , game :: AbstractGame
                    ; temperature = 1.f0
                    , kwargs...
                    ) :: Vector{Float32}

  root = run_mcts(model, game; kwargs... )

  # During self play, we pick a move from the improved stochastic policy
  # root.visit_counter at random.
  # visit_counter seems to be prefered over expected_reward when choosing
  # the best move in a match, as it is less susceptible to random fluctuations.
  apply_temperature(root.visit_counter, temperature)
end

function mcts_value_policy( model
                            , game :: AbstractGame
                            ; temperature = 1.f0
                            , kwargs...
                            ) :: Tuple{Float32, Vector{Float32}}

  root = Node()
  policy = mcts_policy(model, game; root, temperature, kwargs...)
  value = sum(policy .* root.expected_reward)
  value, policy
end

