
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
think(:: AbstractPlayer, :: AbstractGame) = error("Not implemented")

"""
    apply(player, game)

Let `player` evaluate `game` and return a named `(value, policy)` tuple.
Only implemented for certain players, like [`IntuitionPlayer`](@ref) and
[`MCTSPlayer`](@ref).

See also [`think`](@ref).
"""
apply(args...; kwargs...) = Model.apply(args...; kwargs...)

function Model.apply(:: AbstractPlayer, :: AbstractGame)
  error("Not implemented")
end

"""
    decide(player, game [; temperature])

Let `player` randomly sample an action after evaluating `think(player, game)`.

If `temperature < 1`, the distribution is sharpened before sampling. Similarly,
it is diluted if `temperature > 1`. If `temperature = 0`, the action with the
highest policy value will always be returned.

See also [`think`](@ref) and [`decideturn`](@ref).
"""
function decide(p :: AbstractPlayer, game :: AbstractGame; temperature = 1f0)
  @assert !Game.isover(game) "Cannot decide on an action in a finished game"
  policy = think(p, game)
  sample(anneal(policy, Float32(temperature)))
end

"""
    decideturn(player, game [; max_actions, temperature])

Let `player` sample a chain of actions until the active player changes.
If `max_actions` is specified, the function will return after this number of
actions even if the active player has not changed yet.

If `temperature < 1`, the distribution is sharpened before sampling. Similarly,
it is diluted if `temperature > 1`. If `temperature = 0`, the action with the
highest policy value will always be returned.

See also [`think`](@ref) and [`decide`](@ref).
"""
function decideturn( p :: AbstractPlayer
                   , game :: AbstractGame
                   ; max_actions = typemax(Int)
                   , temperature = 1f0 )

  actions = ActionIndex[]
  game = copy(game)
  active = Game.mover(game)
  while !Game.isover(game) &&
        Game.mover(game) == active &&
        length(actions) < max_actions

    action = decide(p, game; temperature)
    move!(game, action)
    push!(actions, action)
  end
  actions
end

"""
    move!(game, player [; temperature])

Modify `game` by letting `player` sample and then apply one action with a given
`temperature`.

See also [`turn!`](@ref), [`decide`](@ref), and [`decideturn`](@ref).
"""
move!(args...; kwargs...) = Game.move!(args...; kwargs...)

function Game.move!(game :: AbstractGame, p :: AbstractPlayer; kwargs...)
  Game.move!(game, decide(p, game; kwargs...))
end

"""
    move(game, player [; temperature])

Non-modifying version of [`move!`](@ref).
"""
move(args...; kwargs...) = Game.move(args...; kwargs...)

function Game.move(game :: AbstractGame, p :: AbstractPlayer; kwargs...)
  Game.move(game, decide(p, game; kwargs...))
end


"""
    turn!(game, player [; temperature])

Modify `game` by letting `player` play actions with a given `temperature` until
the active player changes.

See also [`move!`](@ref), [`decide`](@ref), and [`decideturn`](@ref).
"""
function turn!(game :: AbstractGame, p :: AbstractPlayer; kwargs...)
  for action in decideturn(p, game; kwargs...)
    move!(game, action)
  end
  game
end

"""
    turn(game, player [; temperature])

Non-modifying version of [`turn!`](@ref).
"""
function turn(game :: AbstractGame, args...; kwargs...)
  turn!(copy(game), args...; kwargs...)
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

Model.basemodel(:: AbstractPlayer) = nothing
Model.childmodel(:: AbstractPlayer) = nothing
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

function Base.show(io :: IO, :: RandomPlayer{G}) where {G}
  print(io, "RandomPlayer($G)")
end


struct IntuitionPlayer{G <: AbstractGame} <: AbstractPlayer{G}
  model :: AbstractModel
  name :: String
end

"""
A player that returns the policy proposed by a model (with possible annealing).

---

    IntuitionPlayer(model; [name])
    IntuitionPlayer(player; [name])

Create an intuition player powered by the model `model`. If a player is passed,
the intuition player shares this player's model.
"""
function IntuitionPlayer( model :: AbstractModel{G}
                        ; name = nothing
                        ) where {G <: AbstractGame}
  if isnothing(name)
    id = createid(model) 
    name = "int-$id"
  end

  IntuitionPlayer{G}(model, name)
end

function IntuitionPlayer( player :: AbstractPlayer{G}
                        ; name = nothing
                        ) where {G <: AbstractGame}

  IntuitionPlayer(playingmodel(player); name)
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

  policy
end

function Model.apply(p :: IntuitionPlayer{G}, game :: G) where {G <: AbstractGame}

  actions = legalactions(game)
  policy = zeros(Float32, policylength(game))

  r = apply(p.model, game, targets = [:value, :policy])
  policy[actions] = r.policy[actions]

  (value = r.value, policy = policy)
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
  IntuitionPlayer{G}(m, p.name)
end

function Model.adapt(backend, p :: IntuitionPlayer)
  switchmodel(p, adapt(backend, p.model))
end

Base.copy(p :: IntuitionPlayer) = switchmodel(p, copy(p.model))

function Base.show(io :: IO, p :: IntuitionPlayer{G}) where {G <: AbstractGame}
  print(io, "IntuitionPlayer{$(Game.name(G))}($(p.name))")
end

function Base.show( io :: IO
                  , :: MIME"text/plain"
                  , p :: IntuitionPlayer{G}
                  ) where {G <: AbstractGame}
  println(io, "IntuitionPlayer{$(Game.name(G))}:")
  print(io, " name: $(p.name)"); println(io)
  print(io, " model: ")
  show(io, p.model)
end


struct MCTSPlayer{G <: AbstractGame} <: AbstractPlayer{G}
  model :: AbstractModel
  power :: Int
  policy :: MCTSPolicy
  selector :: ActionSelector
  rootselector :: ActionSelector
  draw_bias :: Float32
  name :: String
end

"""
A player that relies on Markov chain tree search (MCTS) results that are
constructed with assistance of a [`Model.AbstractModel`](@ref).

---

    MCTSPlayer(model; kwargs...)
    MCTSPlayer(player; kwargs...)

Create an `MCTSPlayer` powered by the model `model`. If a player `player` is
passed, the `MCTSPlayer` shares this player's model. If no model or player is passed, the classical [`RolloutModel`](@ref) is used.

By default, a conventional PUCT-based MCTS player is created. For an
`MCTSPlayer` constructor with Gumbel Alpha Zero inspired presets, see
[`MCTSPlayerGumbel`](@ref).


## Arguments
- `power = 100`: The number of model queries the player can make per move.
- `policy = VisitCount()`: The final policy extracted from the root node.
- `selector = PUCT()`: The action selector during the MCTS run at non-root \
nodes.
- `rootselector = selector`: The action selector at root nodes.
- `name = nothing`: The name of the player. If `nothing`, a random name is \
generated.
"""
function MCTSPlayer( model :: AbstractModel{G}
                   ; power = 100
                   , policy = VisitCount()
                   , selector = PUCT()
                   , rootselector = selector
                   , draw_bias = 0f0
                   , name = nothing 
                   ) where {G <: AbstractGame}

  if isnothing(name)
    id = createid(model, policy, selector, rootselector)
    name = "mcts$(power)-$id"
  end

  MCTSPlayer{G}(
    model,
    power,
    policy,
    selector,
    rootselector,
    draw_bias,
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
                   , policy = player.policy
                   , selector = player.selector
                   , rootselector = player.rootselector
                   , draw_bias = player.draw_bias
                   , name = nothing )

  MCTSPlayer(
    playingmodel(player);
    power,
    policy,
    selector,
    rootselector,
    draw_bias,
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
                         , nactions = 16
                         , selector = VisitPropTo()
                         , policy = ImprovedPolicy()
                         , name = nothing
                         , draw_bias = 0f0 )
  MCTSPlayer(
    args...;
    power,
    policy,
    selector,
    rootselector = SequentialHalving(nactions),
    draw_bias,
    name,
  )
end

name(p :: MCTSPlayer) = p.name

function think( p :: MCTSPlayer{G}
              , game :: G
              , policy = p.policy
              ) where {G <: AbstractGame}

  root = mcts(
    game,
    p.model,
    p.power;
    p.selector,
    p.rootselector,
    p.draw_bias
  )

  actions = legalactions(game)
  buffer = zeros(Float32, policylength(game))
  buffer[actions] .= getpolicy(policy, root)

  buffer
end

function Model.apply(p :: MCTSPlayer{G}, game :: G) where {G <: AbstractGame}
  root = mcts(
    game,
    p.model,
    p.power;
    p.selector,
    p.rootselector,
    p.draw_bias
  )
  policy = getpolicy(p.policy, root)
  value = sum(policy .* root.qvalues) # TODO: this can be done better by completing the q-values?

  actions = legalactions(game)
  buffer = zeros(Float32, policylength(game))
  buffer[actions] .= policy

  (value = value, policy = buffer)
end

"""
    decideturn(mcts_player, game [; cap_power, max_actions, temperature])

Let an [`MCTSPlayer`](@ref) `mcts_player` decide an action chain for `game`.

If the chain consists of several moves, the player reuses the MCTS expansions
of the previous decision. Passing `cap_power = true` causes the player to always
use a total power (i.e., previous + new expansions) of `mcts_player.power`.
"""
function decideturn( p :: MCTSPlayer{G}
                   , rootgame :: G
                   ; cap_power = false
                   , max_actions = typemax(Int)
                   , exclude = Set{G}()
                   , temperature = 1f0
                   ) where {G <: AbstractGame}

  actions = ActionIndex[]
  game = copy(rootgame)
  root = rootnode()
  active = Game.mover(game)

  # act as long as the game is not finished and it is our turn
  while !Game.isover(game) &&
        Game.mover(game) == active &&
        length(actions) < max_actions

    remaining_power = round(Int, sum(root.visits))
    power = cap_power ? p.power - remaining_power : p.power

    root = mcts(
      game,
      p.model,
      power;
      p.selector,
      p.rootselector,
      root,
      exclude,
      p.draw_bias,
    )

    # this can only be empty if the mcts fails, (usually) because exclude
    # excludes all possible actions
    if isempty(root.children)
      @warn """
      decideturn was called recursively because all actions were excluded
      """
      return decideturn(
        p,
        rootgame;
        cap_power,
        max_actions = max_actions - length(actions),
        exclude,
      )
    end

    pol = getpolicy(p.policy, root)
    pol = anneal(pol, Float32(temperature))

    # get the index of the next action
    index = sample(pol)

    # move the root to the chosen child and forget the past
    root = root.children[index]
    root.parent = nothing

    # apply and record the action
    move!(game, root.action)
    push!(actions, root.action)
    push!(exclude, copy(game))
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
  println(io, " name: $(p.name)")
  println(io, " power: $(p.power)")
  println(io, " policy: $(p.policy)")
  println(io, " selectors: $(p.selector), $(p.rootselector) (root)")
  print(io, " model: ")
  show(io, p.model)
end


struct HumanPlayer <: AbstractPlayer{AbstractGame}
  name :: String
end

"""
A player that queries for interaction (via the command line) before making
a decision.

---

    HumanPlayer([; name])

Human player with name `name`, defaulting to "you".
"""
HumanPlayer(; name = "you") = HumanPlayer(name)

name(p :: HumanPlayer) = p.name

function think(p :: HumanPlayer, game :: AbstractGame)

  # Draw the game
  println()
  visualize(game)
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
    annealfunction(temperature)
    annealfunction([t0, n1 => t1, n2 => t2, ..., nk => tk])
    annealfunction([n1 => t1, n2 => t2, ..., nk => tk])

Create an anneal function. If a float value `temperature` is provided, the
constant function `_ -> temperature` is returned. In the other cases, the anneal
function is equivalent to:
```
n -> if n < n1
  t0
elseif n1 <= n < n2
  t1
elseif n2 <= n < n3
  t2
...
elseif nk <= n
  tk
end
```
If `t0` is not provided explicitly, it is set to zero.
"""
annealfunction(f :: Function) = x -> Float32(f(x))
annealfunction(temp :: Real) = _ -> Float32(temp)

function annealfunction(pairs)
  if pairs[1] isa Real
    t0 = Float32(pairs[1])
    pairs = pairs[2:end]
  else
    t0 = 1f0
  end
  @assert all(p -> isa(p, Pair), pairs) """
  Anneal function creation expected a list of pairs.
  """
  pairs = sort(pairs, by=first)
  n -> begin
    index = findlast(p -> p[1] <= n, pairs)
    isnothing(index) ? t0 : Float32(pairs[index][2])
  end
end

"""
    pvp(player1, player2 [; instance, callback, draw_after, anneal])

Conduct a match between `player1` and `player2` and return the outcome from the
perspective of the starting player `player1`.

The game that `player1` starts with is created via `instance()`. If
`G = gametype(player1, player2)` is a concrete type, `instance = () ->
Game.instance(G)` is passed by default. The call `callback(current_game)` is
issued after each turn. If the game has not ended after `draw_after` moves, a
draw is declared.

A function `anneal`, maping the move counter to a temperature, can be used to dynamically sharpen the policies returned by (`Player.think`)[@ref] of the players.

See also [`pvpgames`](@ref).
"""
function pvp( p1 :: AbstractPlayer
            , p2 :: AbstractPlayer
            ; instance = gametype(p1, p2)
            , callback = (_) -> nothing
            , draw_after = typemax(Int)
            , anneal = _ -> 1f0 )

  anneal = annealfunction(anneal)

  if instance isa Type{<: AbstractGame}
    G = instance
    instance = () -> Game.instance(G)
  end

  game = copy(instance())
  moves = 0

  while !isover(game)
    p = mover(game) == 1 ? p1 : p2
    max_actions = draw_after - moves
    temperature = anneal(moves)
    for action in decideturn(p, game; max_actions, temperature)
      move!(game, action)
      callback(game)
      moves += 1
    end
    if moves >= draw_after
      return Game.draw
    end
  end

  status(game)
end

# TODO: make this pvpmatch, and establish a Jtac.Game.Match{G} type!
"""
    pvpgames(player1, player2 [; instance, callback, draw_after, anneal])

Like [`pvp`](@ref), but the vector of game states is returned.
"""
function pvpgames( p1 :: AbstractPlayer
                 , p2 :: AbstractPlayer
                 ; instance = gametype(p1, p2)
                 , callback = (_) -> nothing
                 , draw_after = typemax(Int)
                 , anneal = _ -> 1f0 )

  anneal = annealfunction(anneal)

  if instance isa Type{<: AbstractGame}
    G = instance
    instance = () -> Game.instance(G)
  end

  game  = copy(instance())
  games = [copy(game)]
  moves = 0

  while !isover(game) && moves < draw_after
    p = mover(game) == 1 ? p1 : p2
    max_actions = draw_after - moves
    temperature = anneal(moves)
    for action in decideturn(p, game; max_actions, temperature)
      move!(game, action)
      callback(game)
      push!(games, copy(game))
      moves += 1
    end
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


