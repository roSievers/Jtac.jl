"""
Node type that stores information related to MCTS simulations.
"""
mutable struct Node
  action :: ActionIndex           # How did we get here?
  player :: Int                   # Who is allowed to play an action?

  index :: Int                    # What is my index in parent's children vector?
  parent :: Union{Node, Nothing}  # Where did we get here from?
  children :: Vector{Node}        # Where can we go to?

  value :: Float32                # Value prior generated by the model
  policy :: Vector{Float32}       # Policy prior generated by the model

  qvalues :: Vector{Float32}      # Empirical estimate of the expected reward (q-value) of the various children
  visits :: Vector{Float32}       # How often were the children visited?
end

"""
    Node(parent, index, action)

Create a child node of `parent` that sits at child index `index` and was reached
via `action`.
"""
Node(parent, index, action) = Node(
  action,
  0,
  index,
  parent,
  Node[],
  0f0,
  Float32[],
  Float32[],
  Float32[],
)

"""
    isroot(node)

Whether the given node is a root node (i.e., has no parent).

See also [`isleaf`](@ref).
"""
isroot(node :: Node) :: Bool = isnothing(node.parent)

"""
    isleaf(node)

Whether the given node is a leaf node (i.e., has not been expanded).

See also [`isroot`](@ref).
"""
isleaf(node :: Node) :: Bool = isempty(node.children)

"""
    resize!(node, n)

Resize the internal buffers of `node` to accept exactly `n` children.
"""
function Base.resize!(node :: Node, n :: Int)
  resize!(node.children, n)
  resize!(node.policy, n)
  resize!(node.qvalues, n)
  resize!(node.visits, n)

  node.policy .= 0
  node.qvalues .= 0
  node.visits .= 0

  nothing
end

"""
    rootnode()

Return an unexpanded root node. Root nodes have no `parent`, no meaningful
`index`, and no meaningful `action`.
"""
rootnode() = Node(nothing, 0, 0)


"""
    expandnode!(node, actions)

Expand the node `node` given action indices `actions`.

This method resizes `node` to `length(actions)` and initializes as many
children. Previously stored children are lost.
"""
function expandnode!(node :: Node, actions)
  n = length(actions)
  resize!(node, n)
  @inbounds for index in 1:n
    child = Node(node, index, actions[index])
    node.children[index] = child
  end
  nothing
end


"""
Structure with the capacity to cache a fixed number of nodes.

`NodeCache`s can be used in successive calls to `mcts` in order to reduce the
number of node allocations. Since they do not grow dynamically, they must be
constructed with a sufficiently large capacity. A pessimistic upper bound
for a game of type `G` is `policylength(G) * power`, where `power` is the
MCTS power.
"""
struct NodeCache{G <: AbstractGame} 
  free :: Vector{Node}
  used :: Vector{Node}
end

"""
    NodeCache(G, capacity)

Create a `NodeCache` for game type `G` that stores `capacity` nodes.
"""
function NodeCache(G, capacity)
  n = Game.policylength(G)
  free = map(1:capacity) do id
    children = Vector{Node}()
    data = [
      Vector{Float32}(),
      Vector{Float32}(),
      Vector{Float32}(),
    ]
    sizehint!(children, n)
    sizehint!.(data, n)
    Node(0, 0, 0, nothing, children, 0f0, data...)
  end
  used = Node[]
  sizehint!(used, capacity)
  
  NodeCache{G}(free, used)
end

"""
    reset!(nodecache)

Reset a nodecache.
"""
function reset!(nc :: NodeCache)
  foreach(nc.used) do node
    node.player = 0
    node.action = 0
    node.index = 0
    node.parent = nothing
    node.value = 0
    resize!(node, 0)
  end
  append!(nc.free, nc.used)
  empty!(nc.used)
end

"""
    rootnode(nodecache)

Draw a root node from `nodecache`.
"""
function rootnode(nc :: NodeCache)
  @assert !isempty(nc.free) "NodeCache capacity exceeded"
  node = pop!(nc.free)
  push!(nc.used, node)
  node
end


"""
    expandnode!(node, actions, nodecache)

Expand `node` via `actions`. All children are drawn from `nodecache`.
"""
function expandnode!(node :: Node, actions, nc :: NodeCache)
  n = length(actions)
  @assert length(nc.free) >= n "NodeCache capacity exceeded"
  resize!(node, n)
  @inbounds for index in 1:n
    child = pop!(nc.free)
    child.index = index
    child.parent = node
    child.action = actions[index]
    node.children[index] = child
    push!(nc.used, child)
  end
  nothing
end

expandnode!(node :: Node, actions, :: Nothing) = expandnode!(node, actions)


"""
Abstract type that embodies a mapping from `Node`s to policy vectors.
"""
abstract type MCTSPolicy end

Pack.@typed MCTSPolicy

"""
    getpolicy!(buffer, policy, node)

Let `policy` determine an MCTS policy from `node` and write it into `buffer`.

See also [`getpolicy`](@ref).
"""
function getpolicy!( buffer :: Vector{Float32}
                   , policy :: MCTSPolicy
                   , node :: Node )
  error("To be implemented")
end


"""
    getpolicy(policy, node)

Let `policy` determine an MCTS policy from `node`.

See also [`getpolicy!`](@ref).
"""
function getpolicy(policy :: MCTSPolicy, node :: Node) 
  buffer = zeros(Float32, length(node.children))
  getpolicy!(buffer, policy, node)
  buffer
end

"""
    randomize!(policy)

Resample the noise source of an MCTS policy design `policy`. This method has no
effect on deterministic MCTS policy designs.
"""
randomize!(:: MCTSPolicy) = nothing

Base.copy(policy :: MCTSPolicy) = policy


"""
MCTS policy design that returns the model policy prior of a node.
"""
struct ModelPolicy <: MCTSPolicy end

function getpolicy!(buffer :: Vector{Float32}, :: ModelPolicy, node :: Node)
  resize!(buffer, length(node.children))
  buffer .= node.policy
end

Base.show(io :: IO, :: ModelPolicy) = print(io, "ModelPolicy()")
Base.show(io :: IO, ::MIME"text/plain", :: ModelPolicy) = print(io, "ModelPolicy()")


"""
MCTS policy design that proposes a policy proportional to the visit count of
the MCTS node.
"""
struct VisitCount <: MCTSPolicy end

function getpolicy!(buffer :: Vector{Float32}, :: VisitCount, node :: Node)
  resize!(buffer, length(node.children))
  buffer .= node.visits ./ sum(node.visits)
end

Base.show(io :: IO, :: VisitCount) = print(io, "VisitCount()")
Base.show(io :: IO, ::MIME"text/plain", :: VisitCount) = print(io, "VisitCount()")

"""
MCTS policy design that returns the improved policy estimate from the Gumbel
Alpha Zero publication (Danihelka, Guez, Schrittwieser, Silver 2022).
"""
struct ImprovedPolicy <: MCTSPolicy
  c_visit :: Float32
  c_scale :: Float32
  mixed :: Bool
end

"""
    ImprovedPolicy(; visit_offset = 50, scale = 0.1, mixed = true)

Create an `ImprovedPolicy` design with the given parameters.
"""
function ImprovedPolicy(; visit_offset = 50, scale = 0.1, mixed = true)
  ImprovedPolicy(visit_offset, scale, mixed)
end

"""
    _mixedvalue(node)

Returns an estimate for the value of `node` that takes `node.value`,
`node.policy`, as well as `node.visits` into account.

This value is proposed as an improved value estimate in Appendix D of
(Danihelka, Guez, Schrittwieser, Silver 2022)
"""
function _mixedvalue(node)
  total_visits = 0f0
  visited_policy_weight = 1f-10 # prevent division by zero
  visited_value = 0f0

  @inbounds for index in 1:length(node.children)
    total_visits += node.visits[index]
    if node.visits[index] > 0
      visited_policy_weight += node.policy[index]
      visited_value += node.policy[index] * node.qvalues[index]
    end
  end
  value_mean = visited_value / visited_policy_weight
  (node.value + total_visits * value_mean) / (1 + total_visits)
end

function getpolicy!( buffer :: Vector{Float32}
                   , ex :: ImprovedPolicy
                   , node :: Node )

  resize!(buffer, length(node.children))

  max_visits = maximum(node.visits)
  total_visits = sum(node.visits)

  # If no child has been visited, no reasonable improvement is possible.
  # In this case, return the policy.
  if max_visits == 0
    buffer .= node.policy

  # Else, improve the policy by taking into account the discovered qvalues
  else
    value_fallback = ex.mixed ? _mixedvalue(node) : node.value
    @inbounds for index in 1:length(node.children)
      if node.visits[index] == 0
        q = 0.5f0 * (1 + value_fallback)
      else
        q = 0.5f0 * (1 + node.qvalues[index])
      end
      q_transformed = (ex.c_visit + max_visits) * ex.c_scale * q
      buffer[index] = log(node.policy[index] + 1f-10) + q_transformed
    end
    buffer .= softmax(buffer)
  end
end

Base.show(io :: IO, :: ImprovedPolicy) = print(io, "ImprovedPolicy()")

function Base.show(io :: IO, ::MIME"text/plain", policy :: ImprovedPolicy)
  print(io, "ImprovedPolicy(")
  print(io, "$(policy.c_visit), $(policy.c_scale), $(policy.mixed)")
  print(io, ")")
end


"""
MCTS policy design that anneals another MCTS policy by a given temperature.
"""
struct Anneal <: MCTSPolicy
  policy :: MCTSPolicy
  temperature :: Float32
end

Base.copy(policy :: Anneal) = Anneal(copy(policy.policy), policy.temperature)

"""
    anneal(weights, temperature)

Sharpen or broaden the probability distribution `weights` by applying
the operation `weights.^temperature` (plus normalization) to it.
"""
function anneal(values :: Vector{Float32}, temp :: Float32)
  if temp == 0f0
    vals = zeros(Float32, length(values))
    vals[findmax(values)[2]] = 1
    vals
  elseif temp == 1f0
    values / sum(values)
  else
    # cast to Float64 so that low temperatures make less problems
    logs = log.(Float64.(values)) / temp
    logs .= logs .- maximum(logs)
    weights = exp.(logs)
    Float32.(weights / sum(weights))
  end
end

function getpolicy!(buffer :: Vector{Float32}, policy :: Anneal, node :: Node)
  resize!(buffer, length(node.children))
  getpolicy!(buffer, policy.policy, node)
  buffer .= anneal(buffer, policy.temperature)
end

function randomize!(policy :: Anneal)
  randomize!(policy.policy)
end

Base.show(io :: IO, policy :: Anneal) = show(io, MIME"text/plain"(), policy)

function Base.show(io :: IO, ::MIME"text/plain", policy :: Anneal)
  print(io, "Anneal(")
  print(io, "$(policy.policy), $(policy.temperature)")
  print(io, ")")
end


"""
MCTS policy that distorts another MCTS policy by adding Lognormal noise.
"""
struct Lognormal <: MCTSPolicy
  policy :: MCTSPolicy
  dilution :: Float32
  noise :: Vector{Float32}
end

Pack.@only Lognormal [:policy, :dilution]

"""
    Lognormal(policy, dilution)

Create a MCTS policy design that distorts `policy` by adding a fraction of
`dilution` lognormal noise. The value `dilution` must be between `0` and `1`.
"""
function Lognormal(policy :: MCTSPolicy, dilution :: AbstractFloat)
  @assert 0 <= dilution <= 1 "Unexpected dilution value $dilution"
  Lognormal(policy, dilution, Float32[])
end

Base.copy(policy :: Lognormal) = Lognormal(copy(policy.policy), policy.dilution)

function getpolicy!( buffer :: Vector{Float32}
                   , policy :: Lognormal
                   , node :: Node )
  # If the policy has been randomized, sample new noise on first application
  if isempty(policy.noise)
    resize!(policy.noise, length(node.children))
    policy.noise .= exp.(randn.(Float32, 1:n))
    policy.noise ./= sum(policy.noise)
  end

  # Add random component to the policy
  dil = policy.dilution
  getpolicy!(buffer, policy.policy, node)
  buffer .= dil .* policy.noise .+ (1 .- dil) .* buffer
end

function randomize!(policy :: Lognormal)
  randomize!(policy.policy)
  resize!(policy.noise, 0)
end

Base.show(io :: IO, policy :: Lognormal) = show(io, MIME"text/plain"(), policy)

function Base.show(io :: IO, ::MIME"text/plain", policy :: Lognormal)
  print(io, "Lognormal(")
  print(io, "$(policy.policy), $(policy.dilution)")
  print(io, ")")
end


"""
MCTS policy that distorts another MCTS policy by adding Gumbel(0, 1) noise in
the logit-domain.
"""
struct Gumbel <: MCTSPolicy
  policy :: MCTSPolicy
  noise :: Vector{Float32}
end

Pack.@only Gumbel [:policy]

"""
    Gumbel(policy = ImprovedPolicy())

Create an MCTS policy design that distorts `policy` by adding `Gumbel(0, 1)`
noise in the logit-domain.

The default value of `policy` corresponds to the recommendation in the
publication (Danihelka, Guez, Schrittwieser, Silver 2022).

Note: Due to the properties of the Gumbel distribution, sampling an index from
`policy` distributionally equals taking the argmax of `Gumbel(policy)`.
"""
function Gumbel(policy :: MCTSPolicy = ImprovedPolicy())
  Gumbel(policy, Float32[])
end

Base.copy(policy :: Gumbel) = Gumbel(copy(policy.policy))

function getpolicy!( buffer :: Vector{Float32}
                   , policy :: Gumbel
                   , node :: Node )
  # If the policy has been randomized, sample new noise on the first application
  if isempty(policy.noise)
    resize!(policy.noise, length(node.children))
    @inbounds for index in 1:length(node.children)
      policy.noise[index] = rand(Float32) # Be memory-efficient
    end
    policy.noise .= .- log.(policy.noise .+ 1f-10)
    # This would be the actual Gumbel noise:
    #   policy.noise .= -log.(policy.noise) 
    # but since we have to exponentiate it anyway (we apply it in the
    # logit-domain), we just take the multiplicative inverse
    policy.noise .= 1 ./ policy.noise
  end

  getpolicy!(buffer, policy.policy, node)
  buffer .= policy.noise .* buffer
  buffer ./= sum(buffer)
end

function randomize!(policy :: Gumbel)
  randomize!(policy.policy)
  resize!(policy.noise, 0)
end

Base.show(io :: IO, policy :: Gumbel) = show(io, MIME"text/plain"(), policy)

function Base.show(io :: IO, ::MIME"text/plain", policy :: Gumbel)
  print(io, "Gumbel(")
  print(io, "$(policy.policy)")
  print(io, ")")
end


"""
Abstract type that represents an action selection algorithm.

Action selectors are meant to realize selection schemes during an MCTS run.
However, if applied to nodes returned by calls to `mcts(...)`, they can also be
used to govern higher level action selection in players.

Concrete subtypes must implement

 * `select(selector, node)` for non-root selection,
 * `rootselect(selector, node, budget, phase)` for root selection,
 * `randomize!(selector)` for root selection.

Currently, randomization at non-root nodes is not supported.
"""
abstract type ActionSelector end

Pack.@typed ActionSelector

"""
    select(selector, node; [buffer])

Let `selector` select a child from `node.children` and return its index.
Some selectors need fewer allocations if a `buffer` vector is provided.
"""
function select(sel :: ActionSelector, :: Node; buffer = nothing)
  error("Action selectors of type $(typeof(sel)) cannot select non-root actions.")
end

"""
    rootselect(selector, root, budget, phase; [buffer])

Let `selector` plan a selection strategy at the `root` node with a given power
budget `budget` in phase `phase`. Returns an iterable with entries `(index,
visits)`, which indicates that `root.children[index]` shall be visited `visits`
times. Each time `select` is called at the root node, `phase`, which starts
with value `0`, is incremented by one. Some selectors need fewer allocations
if a `buffer` vector is provided.

The sum of all values `visits` must be smaller than or equal to `budget`.
"""
function rootselect(sel :: ActionSelector, node :: Node, :: Int, :: Int; kwargs...) 
  ((select(sel, node; kwargs...), 1),)
end

"""
    randomize!(selector)

Randomize an action selector `selector`.

Randomization is called once at the beginning of an MCTS run on the root
selector, after the root policy prior has been established.

Currently, there is no interface to support randomization at non-root nodes.
"""
randomize!(:: ActionSelector) = error("To be implemented")

"""
Selector that samples action indices from the policy returned by an
`MCTSPolicy`.
"""
struct SampleSelector <: ActionSelector
  policy :: MCTSPolicy

  @doc """
      SampleSelector(policy)

  Wrap an MCTS policy design `policy` into a `SampleSelector` action selector.
  """
  SampleSelector(policy) = new(policy)
end

Base.copy(sel :: SampleSelector) = SampleSelector(copy(sel.policy))

"""
    sample(probs)

Auxiliary function that samples an index from a probability vector `probs`.
"""
function sample(probs :: Vector{Float32}) :: Int
  @assert all(probs .>= 0) && sum(probs) ≈ 1.0 "probability vector not proper"
  r = rand(Float32)
  index = findfirst(x -> r <= x, cumsum(probs))
  isnothing(index) ? length(probs) : index
end

function select(sel :: SampleSelector, node :: Node; buffer = Float32[])
  getpolicy!(buffer, sel.policy, node)
  sample(buffer)
end

randomize!(sel :: SampleSelector) = randomize!(sel.policy)

Base.show(io :: IO, sel :: SampleSelector) = show(io, MIME"text/plain"(), sel)

function Base.show(io :: IO, ::MIME"text/plain", sel :: SampleSelector)
  print(io, "SampleSelector(")
  print(io, "$(sel.policy)")
  print(io, ")")
end


"""
Selector that picks the action with the highes weight in the policy returned by
an `MCTSPolicy`.
"""
struct MaxSelector <: ActionSelector
  policy :: MCTSPolicy

  @doc """
      MaxSelector(policy)

  Wrap an MCTS policy design `policy` into a `MaxSelector` action selector.
  """
  MaxSelector(policy) = new(policy)
end

Base.copy(sel :: MaxSelector) = MaxSelector(copy(sel.policy))

function select(sel :: MaxSelector, node :: Node; buffer = Float32[])
  getpolicy!(buffer, sel.policy, node)
  findmax(buffer)[2]
end

randomize!(sel :: MaxSelector) = randomize!(sel.policy)

Base.show(io :: IO, sel :: MaxSelector) = show(io, MIME"text/plain"(), sel)

function Base.show(io :: IO, ::MIME"text/plain", sel :: MaxSelector)
  print(io, "MaxSelector(")
  print(io, "$(sel.policy)")
  print(io, ")")
end

"""
Selector that maximizes the PUCT with a given `exploration` parameter.

The PUCT is always calculated with respect to the model policy prior.
"""
struct PUCT <: ActionSelector
  exploration :: Float32
end

"""
    PUCT(exploration = 1.41)

Return a `PUCT` selector with exploration parameter `exploration`.
"""
function PUCT(exploration :: AbstractFloat = 1.41f0)
  PUCT(Float32(exploration))
end

Base.copy(sel :: PUCT) = sel

function select(sel :: PUCT, node :: Node; buffer = nothing)
  # The constant 1f-6 is added to make sure that the model policy matters for
  # finding the maximum even if node.visits only contains zeros.
  weight = sel.exploration * sqrt(sum(node.visits)) + 1f-6

  max_index = 1
  max_puct = -Inf32

  @inbounds for index in 1:length(node.children)
    explore = weight * node.policy[index] / (1 + node.visits[index])
    puct = node.qvalues[index] + explore
    if puct > max_puct
      max_index = index
      max_puct = puct
    end
  end

  max_index
end

randomize!(:: PUCT) = nothing

Base.show(io :: IO, sel :: PUCT) = show(io, MIME"text/plain"(), sel)

function Base.show(io :: IO, ::MIME"text/plain", sel :: PUCT)
  print(io, "PUCT(")
  print(io, "$(sel.exploration)")
  print(io, ")")
end


"""
Select actions that lets the visit counts of a node increase proportional to the
policy returned by an MCTS policy design.
"""
struct VisitPropTo <: ActionSelector
  policy :: MCTSPolicy
end

"""
    VisitPropTo(policy = ImprovedPolicy())

Return an `VisitPropTo` selector that chooses actions proportional to the
policy estimate of the MCTS policy design `policy`.

The default value of `policy` corresponds to the recommendation in the
publication (Danihelka, Guez, Schrittwieser, Silver 2022).
"""
function VisitPropTo()
  VisitPropTo(ImprovedPolicy())
end

Base.copy(sel :: VisitPropTo) = VisitPropTo(copy(sel.policy))

function select(sel :: VisitPropTo, node :: Node; buffer = Float32[])
  getpolicy!(buffer, sel.policy, node)
  buffer .= buffer .- node.visits ./ (sum(node.visits) + 1)
  findmax(buffer)[2]
end

randomize!(sel :: VisitPropTo) = randomize!(sel.policy)

Base.show(io :: IO, sel :: VisitPropTo) = show(io, MIME"text/plain"(), sel)

function Base.show(io :: IO, ::MIME"text/plain", sel :: VisitPropTo)
  print(io, "VisitPropTo(")
  print(io, "$(sel.policy)")
  print(io, ")")
end


"""
Sequential halving root selector.

This selector cannot be applied to non-root nodes.
"""
struct SequentialHalving <: ActionSelector
  policy :: MCTSPolicy
  nactions :: Int
end

"""
    SequentialHalving(policy = Gumbel(), nactions)

Sequential halving root selector that considers (at most) `nactions` actions
with the highest policy values reported by `policy`.

The default choice of policy corresponds to the recommendations in the
publication (Danihelka, Guez, Schrittwieser, Silver 2022).
"""
function SequentialHalving(nactions :: Int)
  SequentialHalving(Gumbel(ImprovedPolicy()), nactions)
end

Base.copy(sel :: SequentialHalving) = SequentialHalving(copy(sel.policy), sel.nactions)

function rootselect( sel :: SequentialHalving,
                     node :: Node,
                     budget :: Int,
                     phase :: Int;
                     buffer = Float32[] )

  nactions = div(sel.nactions, 2^phase)
  nactions = min(nactions, length(node.children))

  if nactions <= 1
    nactions = 1
    visits = budget
  elseif budget <= nactions
    nactions = budget
    visits = 1
  elseif budget <= 2nactions
    visits = 1
  else
    visits = floor(Int, budget / 2 / nactions)
  end

  getpolicy!(buffer, sel.policy, node)
  indices = sortperm(buffer, rev = true)

  [(index, visits) for index in indices[1:nactions]]
end

randomize!(sel :: SequentialHalving) = randomize!(sel.policy)

Base.show(io :: IO, sel :: SequentialHalving) = show(io, MIME"text/plain"(), sel)

function Base.show(io :: IO, ::MIME"text/plain", sel :: SequentialHalving)
  print(io, "SequentialHalving(")
  print(io, "$(sel.policy), $(sel.nactions)")
  print(io, ")")
end


"""
    mctsstep!(node, game, model, selector, cache = nothing)

Conduct a single MCTS step at the game state `game` and record the results
in `node`.

First, `selector` is used to find a node that is still unexpanded. This leaf
node is expanded and initialized via `model`. The value estimated by `model` is
then backpropagated towards the root of `node`. The argument `cache` can be
a [`NodeCache`](@ref) whose buffered nodes are then used for the expansion of nodes.
"""
function mctsstep!( node :: Node
                  , game :: G
                  , model
                  , selector :: ActionSelector
                  ; cache = nothing
                  , buffer = Float32[] ) where {G}

  # Do not modify the original game state
  game = copy(game)

  # Find leaf to expand
  while !isleaf(node)
    index = select(selector, node; buffer)
    child = node.children[index]
    Game.move!(game, child.action)
    node = child
  end

  # Expand leaf
  node.player = Game.activeplayer(game)
  if Game.isover(game)
    node.value = Game.status(game) * node.player
  else
    actions = Game.legalactions(game)
    expandnode!(node, actions, cache)
    prior = Model.apply(model, game, targets = [:value, :policy])
    node.value = prior.value
    policy = @view prior.policy[actions]
    node.policy .= policy ./ (sum(policy) + 1f-6)
    # Symmetry breaking
    for index in 1:length(actions)
      node.policy[index] += 1f-6 * rand(Float32)
    end
  end

  # Value from perspective of player 1
  value1 = node.value * node.player

  # Backpropagate the model value towards the root
  while !isroot(node)
    index = node.index
    parent = node.parent
    value = value1 * parent.player
    visits = parent.visits[index]
    qvalue = parent.qvalues[index]

    parent.qvalues[index] = (visits * qvalue + value) / (visits + 1)
    parent.visits[index] += 1
    node = parent
  end
end


"""
    mcts(game, model, power; kwargs...)

Conduct an MCTS simulation at `game` with power `power` and prior information
derived from `model`. Returns the root node at `game`.

## Keyword Arguments
* `root`: The root node. By passing it explicitly, results from previous \
simulations may be reused.
* `selector = PUCT()`: The action selector at non-root nodes.
* `rootselector = selector`: The action selector at the root node.
* `cache = nothing`: An optional [`NodeCache`](@ref) to reduce node \
allocations in case of repeated calls.
"""
function mcts( game :: G
             , model
             , power
             ; root = rootnode()
             , selector = PUCT()
             , rootselector = selector
             , cache = nothing ) where {G}

  # Since the rootselector is randomized, act on a copy
  rootselector = copy(rootselector)
  randomize!(rootselector)
  buffer = Float32[]

  # If the game is already over, there are no legal actions left. Thus,
  # set the node size to 0 and insert the game result as value.
  if Game.isover(game)
    resize!(root, 0)
    root.value = status(game)
    return root
  end

  # If the root node is a leaf, expand it first. The selector does not matter
  # here since we do not need to descent.
  if isleaf(root)
    mctsstep!(root, game, model, selector; cache, buffer)
    budget = power - 1
  else
    budget = power
  end

  # This function is called on each index and visit count chosen by the root
  # selector
  explore = ((index, visits),) -> begin
    child = root.children[index]
    childgame = Game.move!(copy(game), child.action)
    for _ in 1:visits
      mctsstep!(child, childgame, model, selector; cache, buffer)
    end
    budget -= visits
  end

  # The root selector determines how to distribute the power budget.
  phase = 0
  while budget > 0
    selection = rootselect(rootselector, root, budget, phase; buffer)
    if length(selection) == 1
      explore(selection[1])
    else
      asyncmap(explore, selection; ntasks = length(selection))
    end
    phase += 1
  end

  root
end
