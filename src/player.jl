
# A player is an agent that can change a game by choosing actions to perform
abstract type Player{G <: Game} end

# This method must be implemented by each player
think(game :: Game, p :: Player) :: ActionIndex = error("Not implemented")

# Convenience function to automatically alter the game
turn!(game :: Game, p :: Player) = apply_action!(game, think(game, p))

# It is nice to have a name for each player if we want to do tournaments etc.
name(p :: Player) :: String = error("Not implemented")

# A player that always chooses random actions from allowed ones
struct RandomPlayer <: Player{Game} end

think(game :: Game, p :: RandomPlayer) = random_action(game)

name(p :: RandomPlayer) = "random"


# A player that has a model that it can ask for decision making 
struct MCTPlayer{G} <: Player{G}
  model :: Model{G}
  power :: Int
  temperature :: Float32
  name :: String
end

function MCTPlayer(model :: Model{G}; 
                   power = 100, temperature = 1., name = nothing) where {G <: Game}
  if name == nothing
    id = Int(div(hash((model, temperature)), Int(1e14)))
    name = "mct$(power)-$id"
  end
  MCTPlayer{G}(model, power, temperature, name)
end

# The default MCTPlayer uses the RolloutModel
MCTPlayer(; kwargs...) = MCTPlayer(RolloutModel(); kwargs...)

function think(game :: G, p :: MCTPlayer{G}) where {G <: Game}
  mctree_action(p.model, game, power = p.power, temperature = p.temperature)
end

name(p :: MCTPlayer) = p.name


# Player that uses the model policy decision directly
# The temperature controls how strictly/loosely it follows the policy
struct IntuitionPlayer{G} <: Player{G}
  model :: Model{G}
  temperature :: Float32
  name :: String
end

function IntuitionPlayer(model :: Model{G}; 
                      temperature = 1., name = nothing) where {G <: Game}
  if name == nothing
    id = Int(div(hash((model, temperature)), Int(1e14)))
    name = "policy-$id"
  end
  IntuitionPlayer{G}(model, temperature, name)
end

function think(game :: G, p :: IntuitionPlayer{G}) where {G <: Game}
  
  # Get all legal actions and their model policy values
  actions = legal_actions(game)
  policy = p.model(game)[actions .+ 1]
  
  # Return the action that the player decides for
  if p.temperature == 0
    index = findmax(policy)[2]
  else
    weighted_policy = policy.^(1/p.temperature)
    index = choose_index(weighted_policy / sum(weighted_policy))
  end
  actions[index]
end

name(p :: IntuitionPlayer) = p.name


# Human player that queries for interaction
# Relies on implemented draw() method for the game
struct HumanPlayer <: Player{Game}
  name :: String
end

HumanPlayer() = HumanPlayer("player")

function think(game :: Game, p :: HumanPlayer) :: ActionIndex
  println()
  draw(game)
  while true
    print("$(p.name): ")
    input = readline()
    try 
      action = parse(Int, input)
      if !is_action_legal(game, action)
        println("Action $input is illegal ($error)")
      else
        return action
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

# Let players play versus players

function pvp(p1 :: Player, p2 :: Player, game :: Game)
  game = copy(game)
  while !is_over(game)
    if current_player(game) == 1
      turn!(game, p1)
    else
      turn!(game, p2)
    end
  end
  status(game)
end

function pvp(p1 :: Player{G1}, p2 :: Player{G2}) where {G1, G2}
  # Find the most concrete game type
  G = typeintersect(G1, G2)
  # Check that it really is a concrete type
  if G == Game
    error("Cannot infere a concrete game from the provided players")
  end
  pvp(p1, p2, G())
end


#function duel(game, p1 :: Player, p2 :: Player)

