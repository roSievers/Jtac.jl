
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
end

MCTPlayer(model) = MCTPlayer(model, 100)

function think(game :: Game, p :: MCTPlayer) :: ActionIndex
  mctree_action(game, power = p.power, model = p.model)
end


# Player that uses the model policy decision directly
struct PolicyPlayer <: Player
  model :: Model
end

function think(game :: Game, p :: PolicyPlayer) :: ActionIndex
  policy = p.model(game)[2:end]
  argmax(policy)
end


# Player that uses the soft (random) model policy decision directly
struct SoftPolicyPlayer <: Player
  model :: Model
end

function think(game :: Game, p :: SoftPolicyPlayer) :: ActionIndex
  policy = p.model(game)[2:end]
  choose_index(policy)
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
