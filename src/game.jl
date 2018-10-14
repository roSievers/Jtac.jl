
# A Jtacs Game
abstract type Game end

# Status of the game
#   nothing  : the game is not over yet
#   1, 0, -1 : victory of player 1, draw, or victory of player 2
# Other values lead to undefined behavior
const Status = Union{Nothing, Int}

# We are not sure about the concrete implementation of Status yet,
# so we wrap it in a Status constructor that has the following properties:
#   - is_over(Status(nothing)) must return false
#   - is_over(Status(value)) must return true for value = 1,0,-1
#   - Status(value) == value for value = 1,0,-1
Status(value) = value

is_over(s :: Status) = s != nothing

with_default(s :: Status, default :: Int) :: Int = is_over(s) ? s : default

# The actions of a Jtacs Game are represented by canonical indices
# from 1 to policy_size(game)
const ActionIndex = Int


# When using a game with broadcasting functionality, 
# do not try to iterate over it
Broadcast.broadcastable(game :: Game) = Ref(game)
 
Base.copy(:: Game) = error("unimplemented") 

function legal_actions(:: Game) :: Vector{ActionIndex}
  error("unimplemented")
end

function apply_action!(:: Game, :: ActionIndex) :: Nothing
  error("unimplemented")
end

function policy_size(:: Type{Game}) :: UInt
  error("unimplemented")
end

function policy_size(game :: G) :: UInt where G <: Game
  policy_size(G)
end

# Return the game status
status(game :: Game) :: Status = error("unimplemented")

# Whether the game has finished yet
is_over(game :: Game) = is_over(status(game))

# Data representation of the game as layered 2d image
representation(:: Game) :: Array{Float32, 3} = error("unimplemented")

# Size of the data representation of the game
Base.size(:: Game) :: Tuple{Int, Int, Int} = error("unimplemented")

# Convenience functions 
random_action(game :: Game) = rand(legal_actions(game))
random_turn!(game :: Game)  = apply_action!(game, random_action(game))

function random_playout!(game :: Game) :: Game
  while !is_over(game)
    random_turn!(game)
  end
  game
end

function random_playout(game :: Game) :: Game
  random_playout!(copy(game))
end

