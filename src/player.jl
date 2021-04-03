
# -------- Players ----------------------------------------------------------- #

"""
Players are named entities that can evaluate game states via the `think`
function to yield policies.
"""
abstract type AbstractPlayer{G <: AbstractGame} end

"""
    think(player, game)

Let `player` think about `game` and return a policy.
"""
think(p :: AbstractPlayer, game :: AbstractGame) :: Vector{Float32} = error("Not implemented")

"""
    decide(player, game)

Let `player` make an action decision about `game`, based on the policy
`think(player, game)`.
"""
decide(p :: AbstractPlayer, game :: AbstractGame) = choose_index(think(p, game))

"""
    turn!(game, player)

Modify `game` by letting `player` take one turn.
"""
turn!(game :: AbstractGame, p :: AbstractPlayer) = apply_action!(game, decide(p, game))

"""
    name(player)

The name of a player.
"""
name(:: AbstractPlayer) :: String = error("Not implemented")

"""
    ntasks(player)

How many tasks the player wants to handle via asyncmap.
"""
Model.ntasks(:: AbstractPlayer) = 1

# For automatic game interference
Model.gametype(:: AbstractPlayer{G}) where {G <: AbstractGame} = G

# Players with potentially trainable models can be asked to return them
Model.base_model(p :: AbstractPlayer)   = nothing
Model.playing_model(:: AbstractPlayer)  = nothing
Model.training_model(:: AbstractPlayer) = nothing

# Features that are supported by the player. Used for automatic feature
# detection during the generation of datasets in selfplays
Model.features(:: AbstractPlayer) = Feature[]


# Player ids to get more unique default names
get_id(args...) = Int(div(hash(tuple(args...)), Int(1e14)))


# -------- Random Player ----------------------------------------------------- #

"""
A player with name "random" that always chooses a random (but allowed) action.
"""
struct RandomPlayer <: AbstractPlayer{AbstractGame} end

function think(:: RandomPlayer, game :: AbstractGame)
  actions = legal_actions(game)
  policy = zeros(Float32, policy_length(game))
  policy[actions] .= 1f0 / length(actions)
  policy
end

name(p :: RandomPlayer) = "random"
Base.copy(p :: RandomPlayer) = p


# -------- Intuition Player -------------------------------------------------- #

"""
A player that relies on the policy returned by a model.
"""
struct IntuitionPlayer{G} <: AbstractPlayer{G}
  model :: AbstractModel
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
function IntuitionPlayer( model :: AbstractModel{G}
                        ; temperature = 1.
                        , name = nothing
                        ) where {G <: AbstractGame}
  if isnothing(name)
    id = get_id(model, temperature) 
    name = "intuition-$id"
  end

  IntuitionPlayer{G}(model, temperature, name)

end

function IntuitionPlayer( player :: AbstractPlayer{G}
                        ; temperature = player.temperature
                        , name = nothing
                        ) where {G <: AbstractGame}

  IntuitionPlayer( playing_model(player)
                 , temperature = temperature
                 , name = name )

end

function think( p :: IntuitionPlayer{G}
              , game :: G
              ) :: Vector{Float32} where {G <: AbstractGame} 
  
  # Get all legal actions and their model policy values
  actions = legal_actions(game)
  policy = zeros(Float32, policy_length(game))

  policy[actions] = apply(p.model, game).policy[actions]
  
  # Return the action that the player decides for
  if p.temperature == 0f0

    one_hot(length(policy), findmax(policy)[2])

  else

    weights = policy.^(1/p.temperature)
    weights / sum(weights)

  end

end

name(p :: IntuitionPlayer) = p.name

# For convenience, extend some parts of the model interface to players
Model.ntasks(p :: IntuitionPlayer) = Model.ntasks(p.model)
Model.base_model(p :: IntuitionPlayer) = base_model(p.model)
Model.playing_model(p :: IntuitionPlayer) = p.model
Model.training_model(p :: IntuitionPlayer) = training_model(p.model)

function Model.features(p :: IntuitionPlayer) 
  tm = training_model(p)
  isnothing(tm) ? Feature[] : features(tm)
end

function switch_model( p :: IntuitionPlayer{G}
                     , m :: AbstractModel{H}
                     ) where {H <: AbstractGame, G <: H} 
  IntuitionPlayer{G}(m, p.temperature, p.name)
end

Base.copy(p :: IntuitionPlayer) = switch_model(p, copy(p.model))
swap(p :: IntuitionPlayer) = switch_model(p, swap(p.model))


# -------- MCTS Player ------------------------------------------------------- #

"""
A player that relies on Markov chain tree search policies that are constructed
with the support of a model.
"""
struct MCTSPlayer{G} <: AbstractPlayer{G}

  model :: AbstractModel

  power :: Int
  temperature :: Float32
  exploration :: Float32
  dilution :: Float32

  name :: String

end

"""
    MCTSPlayer([model; power, temperature, exploration, name])
    MCTSPlayer(player [; power, temperature, exploration, name])

MCTS Player powered by `model`, which defaults to `RolloutModel`. The model can
also be derived from `player` (this does not create a copy of the model).
"""
function MCTSPlayer( model :: AbstractModel{G}
                   ; power = 100
                   , temperature = 1.
                   , exploration = 1.41
                   , dilution = 0.0
                   , name = nothing 
                   ) where {G <: AbstractGame}

  if isnothing(name)
    id = get_id(model, temperature, exploration, dilution)
    name = "mcts$(power)-$id"
  end

  MCTSPlayer{G}(model, power, temperature, exploration, dilution, name)

end

# The default MCTSPlayer uses the RolloutModel
MCTSPlayer(; kwargs...) = MCTSPlayer(RolloutModel(); kwargs...)


function MCTSPlayer( player :: IntuitionPlayer{G}
                   ; temperature = player.temperature
                   , kwargs...
                   ) where {G <: AbstractGame}

  MCTSPlayer(playing_model(player); temperature = temperature, kwargs...)

end

function MCTSPlayer( player :: MCTSPlayer{G}; kwargs... ) where {G <: AbstractGame}
  MCTSPlayer(playing_model(player); kwargs...)
end

function think( p :: MCTSPlayer{G}
              , game :: G
              ) :: Vector{Float32} where {G <: AbstractGame}

  # Improved policy over the allowed actions
  p = mctree_policy( p.model
                   , game
                   , power = p.power
                   , temperature = p.temperature
                   , exploration = p.exploration
                   , dilution = p.dilution )

  # Full policy vector
  policy = zeros(Float32, policy_length(game))
  policy[legal_actions(game)] = p

  policy

end

name(p :: MCTSPlayer) = p.name
Model.ntasks(p :: MCTSPlayer) = Model.ntasks(p.model)
Model.playing_model(p :: MCTSPlayer) = p.model
Model.base_model(p :: MCTSPlayer) = base_model(p.model)
Model.training_model(p :: MCTSPlayer) = training_model(p.model)

function Model.features(p :: MCTSPlayer)
  tm = training_model(p)
  isnothing(tm) ? Feature[] : features(tm)
end

function switch_model( p :: MCTSPlayer{G}
                     , m :: AbstractModel{H}) where {H <: AbstractGame, G <: H} 

  MCTSPlayer{G}( m
               , p.power
               , p.temperature
               , p.exploration
               , p.dilution
               , p.name )
end

Base.copy(p :: MCTSPlayer) = switch_model(p, copy(p.model))
swap(p :: MCTSPlayer) = switch_model(p, swap(p.model))


# -------- Human Player ------------------------------------------------------ #

"""
A player that queries for interaction before making a decision.
"""
struct HumanPlayer <: AbstractPlayer{AbstractGame}
  name :: String
end

"""
    HumanPlayer([; name])

Human player with name `name`, defaulting to "you".
"""
HumanPlayer(; name = "you") = HumanPlayer(name)

function think(p :: HumanPlayer, game :: AbstractGame)

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

function derive_gametype(players)

  gt = mapreduce(gametype, typeintersect, players, init = AbstractGame)

  @assert gt != Union{} "Players do not play compatible games"
  @assert !isabstracttype(gt) "Cannot infere concrete game from abstract type"

  gt
end


"""
    pvp(player1, player2 [; game, callback])

Conduct one match between `player1` and `player2`. The `game` that
`player1` starts with is infered automatically if possible.
`callback(current_game)` is called after each turn. The game outcome from
perspective of `player1` (-1, 0, 1) is returned.
"""
function pvp( p1 :: AbstractPlayer
            , p2 :: AbstractPlayer
            ; game :: AbstractGame = derive_gametype([p1, p2])()
            , callback = (_) -> nothing )

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

"""
    pvp_games(player1, player2 [; game, callback])

Conduct one match between `player1` and `player2`. The `game` that
`player1` starts with is infered automatically if possible.
`callback(current_game)` is called after each turn. The vector of played game
states is returned.
"""
function pvp_games( p1 :: AbstractPlayer
                  , p2 :: AbstractPlayer
                  ; game :: AbstractGame = derive_gametype([p1, p2])()
                  , callback = (_) -> nothing )

  game  = copy(game)
  games = [copy(game)]

  while !is_over(game)
    if current_player(game) == 1
      turn!(game, p1)
    else
      turn!(game, p2)
    end
    push!(games, copy(game))
    callback(game)
  end

  games

end

