# Jtacs: game.jl

# A Jtacs Game with two competing players. The game can only end by victory of
# Player 1 or 2, or by draw. A realization of this abstract type is meant to
# describe the state of the game, which can be modified by actions (represented
# by action indices).
abstract type Game end

# Status of the game
#   nothing  : the game is not over yet
#   1, 0, -1 : victory of player 1, draw, or victory of player 2
# Other values lead to undefined behavior
#const Status = Union{Nothing, Int}
const Status = Int

# The actions on a Jtacs game are represented by canonical indices
# from 1 to policy_length(game)
const ActionIndex = Int


# We have not decided about the concrete implementation of Status yet,
# so we wrap it in a Status constructor that has the following properties:
#   - is_over(Status()) must return false
#   - is_over(Status(value)) must return true for value = 1,0,-1
#   - Status(value) == value for value = 1,0,-1
Status(value :: Int) = value
Status() = 42

is_over(s :: Status) = (s != 42)

with_default(s :: Status, default :: Int) :: Int = is_over(s) ? s : default


# When using a game with broadcasting functionality, 
# do not try to iterate over it
Broadcast.broadcastable(game :: Game) = Ref(game)

# Whether the game has finished yet
is_over(game :: Game) = is_over(status(game))
 

#
# Interface functions that have to be implemented by concrete games
#

Base.copy(:: Game) :: Game = error("unimplemented") 
status(game :: Game) :: Status = error("unimplemented")
current_player(game :: Game) :: Int = error("unimplemented") # Must be 1 or -1.

legal_actions(:: Game) :: Vector{ActionIndex} = error("unimplemented")
apply_action!(:: Game, :: ActionIndex) :: Nothing = error("unimplemented")

# Data representation of the game as layered 2d image from the perspective of
# the active player (active player plays with 1, other with -1)
representation(:: Game) :: Array{Float32, 3} = error("unimplemented")

# Convenience function if we want to convert several games in a batch
function representation(games :: Vector{G}) :: Array{Float32, 4} where G <: Game
  @assert !isempty(games) "Cannot compute data representation of empty game vector"
  results = zeros(Float32, (size(games[1])..., length(games)))
  for i in 1:length(games)
    results[:,:,:,i] = representation(games[i])
  end
  results
end

# Size of the data representation of the game
Base.size(:: Type{Game}) :: Tuple{Int, Int, Int} = error("unimplemented")
Base.size(:: Type{G}, v) where {G} = size(G)[v]

Base.size(:: G) where {G <: Game} = size(G)
Base.size(:: G, v) where {G <: Game} = size(G, v)

# Length of the policy vector
policy_length(:: Type{Game}) :: Int = error("unimplemented")
policy_length(:: G) where {G <: Game} = policy_length(G)


#
# Convenience functions 
#


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

# This should be implemented by the specific game for performance reasons
function is_action_legal(game :: Game, action :: ActionIndex)
  action in legal_actions(game)
end

# This should be implemented using the symmetries of the specific game
function augment(game :: Game, label) :: Tuple{Vector{Game}, Vector{Vector{Float32}}}
  [game], [label]
end

#
# Optional interface functions
#

draw(:: Game) :: Nothing = error("drawing not available for games of type $(typeof(game))")

