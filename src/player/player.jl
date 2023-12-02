
"""
Abstract type for agents in a two-player board game.

Players act at a higher level of abstraction than models, and their
implementations will usually build upon them (like the
[`IntuitionPlayer`](@ref) or [`MCTSPlayer`](@ref)).

To implement a player, at least the functions [`name`](@ref) and [`think`](@ref)
have to be extended.
"""
abstract type AbstractPlayer{G <: AbstractGame} end

@pack {<: AbstractPlayer} in TypedFormat{MapFormat}

"""
    name(player)

The name of a player.
"""
name(:: AbstractPlayer) = error("Not implemented")

"""
    think(player, game)

Let `player` think about `game` and return a policy proposal.
"""
think(p :: AbstractPlayer, game :: AbstractGame) = error("Not implemented")

"""
    apply(player, game)

Let `player` evaluate `game` and return a named `(value, policy)` tuple.
Only implemented for certain players, like [`IntuitionPlayer`](@ref) and
[`MCTSPlayer`](@ref).

See also [`think`](@ref).
"""
apply(args...; kwargs...) = Model.apply(args...; kwargs...)

function Model.apply(p :: AbstractPlayer, game :: AbstractGame)
  error("Not implemented")
end

"""
    decide(player, game)

Let `player` sample an action after evaluating `think(player, game)`.

See also [`think`](@ref) and [`decidechain`](@ref).
"""
function decide(p :: AbstractPlayer, game :: AbstractGame)
  @assert !Game.isover(game) "Cannot decide on an action in a finished game"
  sample(think(p, game))
end

"""
    decidechain(player, game)

Let `player` sample a chain of actions until the active player changes.

See also [`think`](@ref) and [`decide`](@ref).
"""
function decidechain(p :: AbstractPlayer, game :: AbstractGame)
  actions = ActionIndex[]
  game = copy(game)
  active = Game.activeplayer(game)
  while Game.activeplayer(game) == active && !Game.isover(game)
    action = decide(p, game)
    move!(game, action)
    push!(actions, action)
  end
  actions
end

"""
    move!(game, player)

Modify `game` by letting `player` take one action.

See also [`turn!`](@ref), [`decide`](@ref), and [`decidechain`](@ref).
"""
move!(args...; kwargs...) = Game.move!(args...; kwargs...)

function Game.move!(game :: AbstractGame, p :: AbstractPlayer)
  Game.move!(game, decide(p, game))
end


"""
    turn!(game, player)

Modify `game` by letting `player` play actions until the active player changes.

See also [`move!`](@ref), [`decide`](@ref), and [`decidechain`](@ref).
"""
function turn!(game :: AbstractGame, p :: AbstractPlayer)
  for action in decidechain(p, game)
    move!(game, action)
  end
  game
end

"""
    ntasks(player)

How many tasks the player wants to handle via asyncmap.
"""
ntasks(args...; kwargs...) = Model.ntasks(args...; kwargs...)

Model.ntasks(:: AbstractPlayer) = 1

"""
    gametype(player)

Returns the game type `G <: AbstractGame` that `player` can be applied to.
"""
gametype(args...; kwargs...) = Model.gametype(args...; kwargs...)

Model.gametype(:: AbstractPlayer{G}) where {G <: AbstractGame} = G

childmodel(args...; kwargs...) = Model.childmodel(args...; kwargs...)
basemodel(args...; kwargs...) = Model.basemodel(args...; kwargs...)
playingmodel(args...; kwargs...) = Model.playingmodel(args...; kwargs...)
trainingmodel(args...; kwargs...) = Model.trainingmodel(args...; kwargs...)

Model.basemodel(p :: AbstractPlayer) = nothing
Model.childmodel(p :: AbstractPlayer) = nothing
Model.playingmodel(:: AbstractPlayer) = nothing
Model.trainingmodel(:: AbstractPlayer) = nothing

createid(args...) = Int(div(hash(tuple(args...)), Int(1e14)))


"""
A player with fixed name "random" that considers each legal action as equally
good.
"""
struct RandomPlayer{G} <: AbstractPlayer{G} end

RandomPlayer(G) = RandomPlayer{G}()

function think(:: RandomPlayer, game :: AbstractGame)
  actions = legalactions(game)
  policy = zeros(Float32, policylength(game))
  policy[actions] .= 1f0 / length(actions)
  policy
end

name(p :: RandomPlayer) = "random"

Base.copy(p :: RandomPlayer) = p

function Base.show(io :: IO, p :: RandomPlayer{G}) where {G}
  print(io, "RandomPlayer($G)")
end


"""
A player that returns the policy proposed by a model (with possible annealing).
"""
struct IntuitionPlayer{G <: AbstractGame} <: AbstractPlayer{G}
  model :: AbstractModel
  temperature :: Float32
  name :: String
end

"""
    IntuitionPlayer(model; [temperature, name])
    IntuitionPlayer(player; [temperature, name])

Intuition player that uses `model` to generate policies which are annealed by
`temperature` before making a decision. If a player `player` is passed, the
intuition player shares this player's model.
"""
function IntuitionPlayer( model :: AbstractModel{G}
                        ; temperature = 1.
                        , name = nothing
                        ) where {G <: AbstractGame}
  if isnothing(name)
    id = createid(model, temperature) 
    name = "int-$id"
  end

  IntuitionPlayer{G}(model, temperature, name)
end

function IntuitionPlayer( player :: AbstractPlayer{G}
                        ; temperature = 1.
                        , name = nothing
                        ) where {G <: AbstractGame}

  IntuitionPlayer(playingmodel(player); temperature, name)
end

name(p :: IntuitionPlayer) = p.name

function think( p :: IntuitionPlayer{G}
              , game :: G
              ) where {G <: AbstractGame} 

  # Get all legal actions and their model policy values
  actions = legalactions(game)
  policy = zeros(Float32, policylength(game))

  r = apply(p.model, game, targets = [:policy])
  policy[actions] = r.policy[actions]

  # Return the policy after applying temperature
  anneal(policy, p.temperature)
end

function Model.apply(p :: IntuitionPlayer{G}, game :: G) where {G <: AbstractGame}

  actions = legalactions(game)
  policy = zeros(Float32, policylength(game))

  r = apply(p.model, game, targets = [:value, :policy])
  policy[actions] = r.policy[actions]

  (value = r.value, policy = anneal(policy, p.temperature))
end

Model.ntasks(p :: IntuitionPlayer) = Model.ntasks(p.model)
Model.basemodel(p :: IntuitionPlayer) = basemodel(p.model)
Model.childmodel(p :: IntuitionPlayer) = childmodel(p.model)
Model.playingmodel(p :: IntuitionPlayer) = p.model
Model.trainingmodel(p :: IntuitionPlayer) = trainingmodel(p.model)

function Target.targets(p :: IntuitionPlayer{G}) where {G}
  tm = playingmodel(p)
  isnothing(tm) ? Target.defaulttargets(G) : Target.targets(tm)
end

function switchmodel( p :: IntuitionPlayer{G}
                    , m :: AbstractModel{H}
                    ) where {H <: AbstractGame, G <: H} 
  IntuitionPlayer{G}(m, p.temperature, p.name)
end

function Model.adapt(backend, p :: IntuitionPlayer)
  switchmodel(p, adapt(backend, p.model))
end

Base.copy(p :: IntuitionPlayer) = switchmodel(p, copy(p.model))

function Base.show(io :: IO, p :: IntuitionPlayer{G}) where {G <: AbstractGame}
  print(io, "IntuitionPlayer{$(Game.name(G))}($(p.name), $(p.temperature))")
end

function Base.show( io :: IO
                  , mime :: MIME"text/plain"
                  , p :: IntuitionPlayer{G}
                  ) where {G <: AbstractGame}
  println(io, "IntuitionPlayer{$(Game.name(G))}:")
  print(io, " name: $(p.name)"); println(io)
  print(io, " temp: $(p.temperature)"); println(io)
  print(io, " model: ")
  showindented(io, mime, p.model, indent = 1, indentfirst = false)
end


"""
A player that relies on Markov chain tree search (MCTS) results that are
constructed with the help of a model.
"""
struct MCTSPlayer{G <: AbstractGame} <: AbstractPlayer{G}
  model :: AbstractModel
  power :: Int
  policy :: MCTSPolicy
  selector :: ActionSelector
  rootselector :: ActionSelector
  name :: String
end

"""
    MCTSPlayer(model = RolloutModel(); kwargs...)
    MCTSPlayer(player; kwargs...)

MCTS player powered by a model `model`, which can also be derived from `player`.

By default, a classical PUCT-based MCTS player is created. For an `MCTSPlayer`
constructor with Gumbel Alpha Zero inspired presets, see
[`MCTSPlayerGumbel`](@ref).

## Arguments
- `power = 100`: The number of model queries the player can use per move.
- `policy = VisitCount()`: The final policy extracted from the root node.
- `selector = PUCT()`: The action selector during the mcts run at non-root \
nodes.
- `rootselector = selector`: The action selector at root nodes.
- `temperature`: Convenience option that replaces `policy` by \
`Anneal(policy, temperature)` if provided.
- `name = nothing`: The name of the player. If `nothing`, a random name is \
generated.
"""
function MCTSPlayer( model :: AbstractModel{G}
                   ; power = 100
                   , temperature = nothing
                   , policy = VisitCount()
                   , selector = PUCT()
                   , rootselector = selector
                   , name = nothing 
                   ) where {G <: AbstractGame}

  if isnothing(name)
    id = createid(model, policy, selector, rootselector)
    name = "mcts$(power)-$id"
  end

  if !isnothing(temperature)
    # Prevent annealing already annealed policies
    if policy isa Anneal
      policy = Anneal(policy.policy, temperature)
    else
      policy = Anneal(policy, temperature)
    end
  end

  MCTSPlayer{G}(
    model,
    power,
    policy,
    selector,
    rootselector,
    name,
  )
end

# The default MCTSPlayer uses the RolloutModel
function MCTSPlayer(G :: Type{<: AbstractGame}; kwargs...)
  MCTSPlayer(RolloutModel(G); kwargs...)
end

function MCTSPlayer( player :: IntuitionPlayer{G}
                   , kwargs...
                   ) where {G <: AbstractGame}
  MCTSPlayer(playingmodel(player); kwargs...)
end

function MCTSPlayer( player :: MCTSPlayer
                   ; power = player.power
                   , temperature = nothing
                   , policy = player.policy
                   , selector = player.selector
                   , rootselector = player.rootselector
                   , name = nothing )

  if !isnothing(temperature)
    if policy isa Anneal
      policy = Anneal(policy.policy, temperature)
    else
      policy = Anneal(policy, temperature)
    end
  end
  
  MCTSPlayer(
    playingmodel(player);
    power,
    temperature,
    policy,
    selector,
    rootselector,
    name,
  )
end


"""
    MCTSPlayerGumbel(model/player; [nactions, selector, policy, kwargs...])

Create an [`MCTSPlayer`](@ref) with Gumbel presets. By default, `nactions = 16`,
`selector = VisitPropTo()`, and `policy = ImprovedPolicy()`.

The argument `rootselector = SequentialHalving(nactions)` is passed to
[`MCTSPlayer`](@ref) implicitly. The remaining arguments and defaults are shared
with [`MCTSPlayer`](@ref).
"""
function MCTSPlayerGumbel( args...
                         ; power = 100
                         , temperature = nothing
                         , nactions = 16
                         , selector = VisitPropTo()
                         , policy = ImprovedPolicy()
                         , name = nothing )
  MCTSPlayer(
    args...;
    power,
    temperature,
    policy,
    selector = VisitPropTo(),
    rootselector = SequentialHalving(nactions),
    name,
  )
end

name(p :: MCTSPlayer) = p.name

function think( p :: MCTSPlayer{G}
              , game :: G
              , policy = p.policy
              ) where {G <: AbstractGame}

  root = mcts(game, p.model, p.power; p.selector, p.rootselector)

  actions = legalactions(game)
  buffer = zeros(Float32, policylength(game))
  buffer[actions] .= getpolicy(policy, root)

  buffer
end

function Model.apply(p :: MCTSPlayer{G}, game :: G) where {G <: AbstractGame}
  root = mcts(game, p.model, p.power; p.selector, p.rootselector)
  pol = getpolicy(p.policy, root)
  value = sum(pol .* root.qvalues) # TODO: this can be done better by completing the q-values?

  actions = legalactions(game)
  buffer = zeros(Float32, policylength(game))
  buffer[actions] .= pol

  (value = value, policy = policy)
end

"""
    decidechain(mcts_player, game; cap_power = false)

Let an MCTS player `mcts_player` decide an action chain at `game`.

If the chain consists of several moves, the player reuses the mcts expansions
of the previous decision. Passing `cap_power = true` causes the player to always
use a total power of `mcts_player.power`, even when previous expansions are
available.
"""
function decidechain( p :: MCTSPlayer{G}
                    , game :: G
                    ; cap_power = false ) where {G <: AbstractGame}
  # TODO: make infinite loops impossible!
  actions = ActionIndex[]
  game = copy(game)
  root = rootnode()
  active = Game.activeplayer(game)

  # act as long as the game is not finished and it is our turn
  while !Game.isover(game) && Game.activeplayer(game) == active
    remaining_power = round(Int, sum(root.visits))
    power = cap_power ? p.power - remaining_power : p.power

    root = mcts(game, p.model, power; p.selector, p.rootselector, root)
    pol = getpolicy(p.policy, root)
    
    # select a child index that performed well
    index = sample(pol)

    # move the root to the chosen child and forget the past
    root = root.children[index]
    root.parent = nothing

    # apply and record the action
    action = legalactions(game)[index]
    move!(game, action)
    push!(actions, action)
  end

  actions
end

Model.ntasks(p :: MCTSPlayer) = Model.ntasks(p.model)
Model.basemodel(p :: MCTSPlayer) = basemodel(p.model)
Model.childmodel(p :: MCTSPlayer) = childmodel(p.model)
Model.playingmodel(p :: MCTSPlayer) = p.model
Model.trainingmodel(p :: MCTSPlayer) = trainingmodel(p.model)

function Target.targets(p :: MCTSPlayer{G}) where {G}
  tm = playingmodel(p)
  isnothing(tm) ? Target.defaulttargets(G) : Target.targets(tm)
end

function switchmodel( p :: MCTSPlayer{G}
                    , m :: AbstractModel{H}
                    ) where {H <: AbstractGame, G <: H} 

  MCTSPlayer{G}( m
               , p.power
               , p.policy
               , p.selector
               , p.rootselector
               , p.name )
end

Model.adapt(backend, p :: MCTSPlayer) = switchmodel(p, adapt(backend, p.model))


Base.copy(p :: MCTSPlayer) = switchmodel(p, copy(p.model))

function Base.show(io :: IO, p :: MCTSPlayer{G}) where {G <: AbstractGame}
  print(io, "MCTSPlayer{$(Game.name(G))}")
  print(io, "($(p.name), $(p.power))")
end

function Base.show( io :: IO
                  , mime :: MIME"text/plain"
                  , p :: MCTSPlayer{G}
                  ) where {G <: AbstractGame}
  println(io, "MCTSPlayer{$(Game.name(G))}")
  print(io, " name: $(p.name)\n")
  print(io, " power: $(p.power)\n")
  print(io, " policy: $(p.policy)\n")
  print(io, " selector: $(p.selector)\n")
  print(io, " rootselector: $(p.rootselector)\n")
  print(io, " model: ")
  showindented(io, mime, p.model, indent = 1, indentfirst = false)
end



"""
A player that queries for interaction (via the command line) before making
a decision.
"""
struct HumanPlayer <: AbstractPlayer{AbstractGame}
  name :: String
end

"""
    HumanPlayer([; name])

Human player with name `name`, defaulting to "you".
"""
HumanPlayer(; name = "you") = HumanPlayer(name)

name(p :: HumanPlayer) = p.name

function think(p :: HumanPlayer, game :: AbstractGame)

  # Draw the game
  println()
  draw(game)
  println()

  # Take the user input and return the one-hot policy
  while true
    print("$(p.name): ")
    input = readline()
    try 
      action = parse(Int, input)
      if !isactionlegal(game, action)
        println("Action $input is illegal ($error)")
      else
        policy = zeros(Float32, policylength(game))
        policy[action] = 1f0
        return policy
      end
    catch error
      if isa(error, ArgumentError)
        println("Cannot parse action ($error)")
      else
        println("An unknown error occured: $error")
      end
    end
  end
end

Base.copy(p :: HumanPlayer) = p
Base.show(io :: IO, p :: HumanPlayer) = print(io, "HumanPlayer($(p.name))")


"""
    gametype(players...)

Derive the most general common gametype of the players `players`.
"""
function Model.gametype(p1 :: AbstractPlayer, p2 :: AbstractPlayer, players...)
  players = [p1, p2, players...]
  gt = mapreduce(gametype, typeintersect, players, init = AbstractGame)
  @assert gt != Union{} "Players do not play compatible games"
  @assert !isabstracttype(gt) "Cannot infere concrete game from abstract type"
  gt
end


"""
    pvp(player1, player2 [; instance, callback, draw_after])

Conduct a match between `player1` and `player2` and return the outcome from the
perspective of the starting player `player1`.

The game that `player1` starts with is created via `instance()`. If
`G = gametype(player1, player2)` is a concrete type, `instance = () ->
Game.instance(G)` is passed by default. The call `callback(current_game)` is
issued after each turn. If the game has not ended after `draw_after` moves, a
draw is declared.

See also [`pvpgames`](@ref).
"""
function pvp( p1 :: AbstractPlayer
            , p2 :: AbstractPlayer
            ; instance = gametype(p1, p2)
            , callback = (_) -> nothing
            , draw_after = typemax(Int) )

  if instance isa Type{<: AbstractGame}
    G = instance
    instance = () -> Game.instance(G)
  end

  game = copy(instance())
  moves = 0

  while !isover(game)
    # After draw_after moves, the game ends with a draw
    moves > draw_after && return 0

    if activeplayer(game) == 1
      turn!(game, p1)
    else
      turn!(game, p2)
    end
    callback(game)
    moves += 1
  end

  status(game)
end

"""
    pvpgames(player1, player2 [; instance, callback, draw_after])

Like [`pvp`](@ref), but the vector of game states is returned.
"""
function pvpgames( p1 :: AbstractPlayer
                 , p2 :: AbstractPlayer
                 ; instance = gametype(p1, p2)
                 , callback = (_) -> nothing
                 , draw_after = typemax(Int) )

  if instance isa Type{<: AbstractGame}
    G = instance
    instance = () -> Game.instance(G)
  end

  game  = copy(instance())
  games = [copy(game)]
  moves = 0

  while !isover(game)
    moves > draw_after && return games

    if activeplayer(game) == 1
      turn!(game, p1)
    else
      turn!(game, p2)
    end
    push!(games, copy(game))
    callback(game)
    moves += 1
  end

  games
end

"""
    configure(player; kwargs...)

Configure the model that `player` is based on and return a new player with this
model.
"""
configure(args...; kwargs...) = Model.configure(args...; kwargs...)

function Model.configure(player :: Union{MCTSPlayer, IntuitionPlayer}; kwargs...)
  switchmodel(player, configure(player.model; kwargs...))
end

"""
    isasync(player)

Whether `player` can batch calls to [`think`](@ref) or [`apply`](@ref) in
asynchronous contexts.
"""
isasync(args...; kwargs...) = Model.isasync(args...; kwargs...)

function Model.isasync(player :: Union{MCTSPlayer, IntuitionPlayer})
  isasync(Model.playingmodel(player))
end


