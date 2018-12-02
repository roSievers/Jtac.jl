
# A player is an agent that can change a game by choosing actions to perform
abstract type Player end

# This method must be implemented by each player
think(game :: Game, p :: Player) :: ActionIndex = error("Not implemented")

# Convenience function to automatically alter the game
turn!(game :: Game, p :: Player) = apply_action!(game, think(game, p))


# A player that always chooses random actions from allowed ones
struct RandomPlayer <: Player end

think(game :: Game, p :: RandomPlayer) :: ActionIndex = random_action(game)


# A player that has a model that it can ask for decision making 
struct MCTPlayer <: Player 
  model :: Model
  power :: Int
  temperature :: Float32
end

function MCTPlayer(model; power = 100, temperature = 1.) 
  MCTPlayer(model, power, temperature)
end

function think(game :: Game, p :: MCTPlayer) :: ActionIndex
  mctree_action(game, power = p.power, model = p.model, temperature = p.temperature)
end


# Player that uses the model policy decision directly
# The temperature controls how strictly/loosely it follows the policy
struct PolicyPlayer <: Player
  model :: Model
  temperature :: Float32
end

PolicyPlayer(model :: Model; temperature = 1.) = PolicyPlayer(model, temperature)

function think(game :: Game, p :: PolicyPlayer) :: ActionIndex
  
  # Get the model policy
  policy = p.model(game)[2:end]
  
  # Set policy predictions for illegal actions to 0
  actions = legal_actions(game)
  policy[actions] .= 0.

  # Return the action that the player decides for
  if p.temperature == 0
    findmax(p.model(game)[2:end])[2]
  else
    weighted_policy = p.model(game)[2:end].^(1/p.temperature)
    choose_index(weighted_policy / sum(weighted_policy))
  end
end


# Human player that queries for interaction
# Relies on implemented draw() method for the game
struct HumanPlayer <: Player 
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


# Let players play versus players

function pvp(game, p1 :: Player, p2 :: Player)
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
