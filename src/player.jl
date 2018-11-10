
# A player is an agent that can change a game by making turns
abstract type Player end

turn!(game :: Game, p :: Player) :: Nothing = error("Not implemented")


# A player that always chooses random actions from allowed ones
struct RandomPlayer <: Player end

turn!(game :: Game, p :: RandomPlayer) :: Nothing = random_turn!(game)


# A player that has a model that it can ask for decision making 
struct MCTPlayer <: Player 
  model :: Model
  power :: Int
end

MCTPlayer(model, power = 100) = MCTPlayer(model, power)

function turn!(game :: Game, p :: MCTPlayer) :: Nothing
  mctree_turn!(game, power = p.power, model = p.model)
  nothing
end


# Player that uses the model policy decision directly
struct PolicyPlayer <: Player
  model :: Model
end

function turn!(game :: Game, p :: PolicyPlayer) :: Nothing
  policy = p.model(game)[2:end]
  argmax(policy)
end

# Player that uses the soft (random) model policy decision directly
struct SoftPolicyPlayer <: Player
  model :: Model
end

function turn!(game :: Game, p :: SoftPolicyPlayer) :: Nothing
  policy = p.model(game)[2:end]
  r = rand()
  action = findfirst(x -> r <= x, cumsum(policy))
  @assert action != nothing "policy vector is no proper probability"
  action
end


# Human player that queries for interaction
# Relies on implemented draw() method for the game
struct HumanPlayer <: Player 
  name :: String
end

HumanPlayer() = HumanPlayer("player")

function turn!(game :: Game, p :: HumanPlayer) :: Nothing
  draw(game)
  while true
    print("$(p.name): ")
    input = readline()
    try 
      action = parse(Int, input)
      apply_action!(game, action)
      break
    catch error
      if isa(error, ArgumentError)
        println("Cannot parse action ($error)")
      else
        println("Action $input is illegal ($error)")
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
