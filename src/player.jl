
# -------- Players ----------------------------------------------------------- #

"""
Players are named entities that can evaluate game states via the `think`
function to yield policies.
"""
abstract type Player{G <: Game} end

"""
    think(player, game)

Let `player` think about `game` and return a policy to you.
"""
think(p :: Player, game :: Game) :: Vector{Float32} = error("Not implemented")

"""
    decide(player, game)

Let `player` make an action decision about `game`, based on the policy
`think(player, game)`.
"""
decide(p :: Player, game :: Game) = choose_index(think(p, game))

# Convenience function to automatically alter the game
turn!(game :: Game, p :: Player) = apply_action!(game, decide(p, game))

# Each player must have a name for comparison in tournaments etc.
"""
    name(player)

The name of a player.
"""
name(:: Player) :: String = error("Not implemented")

"""
    ntasks(player)

How many tasks the player wants to handle via asyncmap.
"""
ntasks(:: Player) = 1

# For automatic game interference
gametype(:: Player{G}) where {G <: Game} = G

# Players with potentially trainable models can be asked to return them
playing_model(:: Player) = nothing
training_model(:: Player) = nothing

# Features that are supported by the player. Used for automatic feature
# detection during the generation of datasets in selfplays
features(:: Player) = []


# -------- Random Player ----------------------------------------------------- #

"""
A player with name "random" that always chooses a random (but allowed) action.
"""
struct RandomPlayer <: Player{Game} end

function think(:: RandomPlayer, game :: Game)
  l = length(legal_actions(game))
  ones(l) / l
end

name(p :: RandomPlayer) = "random"
Base.copy(p :: RandomPlayer) = p


# -------- Intuition Player -------------------------------------------------- #

"""
A player that relies on the policy returned by a model.
"""
struct IntuitionPlayer{G} <: Player{G}
  model :: Model
  temperature :: Float32
  name :: String
end

"""
    IntuitionPlayer(model [; temperature, name])
    IntuitionPlayer(player [; temperature, name])

Intuition player that uses `model` to generate policies which are cooled/heated
by `temperature` before making a decision. If provided a `player`, the
IntuitionPlayer shares this player's model and temperature.
"""
function IntuitionPlayer( model :: Model{G}
                        ; temperature = 1.
                        , name = nothing
                        ) where {G <: Game}
  if isnothing(name)
    id = Int(div(hash((model, temperature)), Int(1e14)))
    name = "intuition-$id"
  end

  IntuitionPlayer{G}(model, temperature, name)

end

function IntuitionPlayer( player :: Player{G}
                        ; temperature = player.temperature
                        , name = nothing
                        ) where {G <: Game}

  IntuitionPlayer( playing_model(player)
                 , temperature = temperature
                 , name = name )

end

function think(p :: IntuitionPlayer{G}, game :: G) where {G <: Game}
  
  # Get all legal actions and their model policy values
  actions = legal_actions(game)
  policy = zeros(Float32, policy_length(game))

  policy[actions] = apply(p.model, game).policy[actions]
  
  # Return the action that the player decides for
  if p.temperature == 0
    probs = zeros(Float32, length(policy))
    probs[findmax(policy)[2]] = 1.
  else
    weighted_policy = policy.^(1/p.temperature)
    probs = weighted_policy / sum(weighted_policy)
  end

  probs

end

name(p :: IntuitionPlayer) = p.name
ntasks(p :: IntuitionPlayer) = ntasks(p.model)
playing_model(p :: IntuitionPlayer) = p.model
training_model(p :: IntuitionPlayer) = training_model(p.model)

function features(p :: IntuitionPlayer) 
  tm = training_model(p)
  isnothing(tm) ? Feature[] : features(tm)
end

function switch_model( p :: IntuitionPlayer{G}
                     , m :: Model{H}
                     ) where {H <: Game, G <: H} 
  IntuitionPlayer{G}(m, p.temperature, p.name)
end

Base.copy(p :: IntuitionPlayer) = switch_model(p, copy(p.model))
swap(p :: IntuitionPlayer) = switch_model(p, swap(p.model))


# -------- MCTS Player ------------------------------------------------------- #

"""
A player that relies on Markov chain tree search policies that are constructed
with the support of a model.
"""
struct MCTSPlayer{G} <: Player{G}

  model :: Model

  power :: Int
  temperature :: Float32
  exploration :: Float32

  name :: String

end

"""
    MCTSPlayer([model; power, temperature, exploration, name])
    MCTSPlayer(player [; power, temperature, exploration, name])

MCTS Player powered by `model`, which defaults to `RolloutModel`. The model can
also be derived from `player` (this does not create a copy of the model).
"""
function MCTSPlayer( model :: Model{G}
                   ; power = 100
                   , temperature = 1.
                   , exploration = 1.41
                   , name = nothing 
                   ) where {G <: Game}

  if isnothing(name)
    id = Int(div(hash((model, temperature)), Int(1e14)))
    name = "mcts$(power)-$id"
  end

  MCTSPlayer{G}(model, power, temperature, exploration, name)

end

# The default MCTSPlayer uses the RolloutModel
MCTSPlayer(; kwargs...) = MCTSPlayer(RolloutModel(); kwargs...)


function MCTSPlayer( player :: IntuitionPlayer{G}
                   ; temperature = player.temperature
                   , kwargs...
                   ) where {G <: Game}

  MCTSPlayer(playing_model(player); temperature = temperature, kwargs...)

end

function MCTSPlayer( player :: MCTSPlayer{G}; kwargs... ) where {G <: Game}
  MCTSPlayer(playing_model(player); kwargs...)
end

function think(p :: MCTSPlayer{G}, game :: G) where {G <: Game}

  # Improved policy over the allowed actions
  pol = mctree_policy( p.model
                     , game
                     , power = p.power
                     , temperature = p.temperature
                     , exploration = p.exploration)

  # Full policy vector
  policy = zeros(Float32, policy_length(game))
  policy[legal_actions(game)] = pol

  policy

end

name(p :: MCTSPlayer) = p.name
ntasks(p :: MCTSPlayer) = ntasks(p.model)
playing_model(p :: MCTSPlayer) = p.model
training_model(p :: MCTSPlayer) = training_model(p.model)

function features(p :: MCTSPlayer)
  tm = training_model(p)
  isnothing(tm) ? Feature[] : features(tm)
end

function switch_model( p :: MCTSPlayer{G}
                     , m :: Model{H}) where {H <: Game, G <: H} 

  MCTSPlayer{G}( m
               , p.power
               , p.temperature
               , p.exploration
               , p.name )
end

Base.copy(p :: MCTSPlayer) = switch_model(p, copy(p.model))
swap(p :: MCTSPlayer) = switch_model(p, swap(p.model))


# -------- Human Player ------------------------------------------------------ #

"""
A player that queries for interaction before making a decision.
"""
struct HumanPlayer <: Player{Game}
  name :: String
end

"""
    HumanPlayer([; name])

Human player with name `name`, defaulting to "you".
"""
HumanPlayer(; name = "you") = HumanPlayer(name)

function think(p :: HumanPlayer, game :: Game)

  # Draw the game
  println()
  draw(game)

  # Take the user input and return the one-hot policy
  while true
    print("$(p.name): ")
    input = readline()
    try 
      action = parse(Int, input)
      if !is_action_legal(game, action)
        println("Action $input is illegal ($error)")
      else
        policy = zeros(Float32, policy_length(game))
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

name(p :: HumanPlayer) = p.name

Base.copy(p :: HumanPlayer) = p


# -------- PvP --------------------------------------------------------------- #

function pvp(p1 :: Player, p2 :: Player, game :: Game; callback = (_) -> nothing)

  game = copy(game)

  while !is_over(game)
    if current_player(game) == 1
      turn!(game, p1)
    else
      turn!(game, p2)
    end
    callback(game)
  end

  status(game)

end

pvp(p1 :: Player, p2 :: Player) = pvp(p1, p2, derive_gametype([p1, p2])())


function pvp_games(p1 :: Player, p2 :: Player, game :: Game)

  game  = copy(game)
  games = [copy(game)]

  while !is_over(game)
    if current_player(game) == 1
      turn!(game, p1)
    else
      turn!(game, p2)
    end
    push!(games, copy(game))
  end

  games

end

function pvp_games(p1 :: Player, p2 :: Player)
  pvp_games(p1, p2, derive_gametype([p1, p2])())
end
