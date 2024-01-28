# Implements a single stack Nim game.
# https://en.wikipedia.org/wiki/Nim
# > Nim is typically played as a misère game, in which the player to take the
# > last object loses.
#
# The interesting thing about this game is that we can represent it as
# executing multiple actions in a single turn. This will help us prepare
# the infrastructure for Paco Ŝako.

# We represent the board as a vector of 15 integers from {0, 1}.
# The player may take away any of the remaining tokens of the board or pass.
# Passing is only allowed, after the player has taken at least one token.


mutable struct Nim2 <: AbstractGame
  remaining :: Int
  active_player :: Int
  actions_left :: Int
  status :: Status
end

Nim2() = Nim2(15, 1, 3, Status())

function Base.copy(s :: Nim2) :: Nim2
    Nim2(
      s.remaining,
      s.active_player,
      s.actions_left,
      s.status,
    )
end


function Base.:(==)(a::Nim2, b::Nim2)
  all([ a.remaining .== b.remaining
      , a.active_player == b.active_player
      , a.status == b.status
      , a.actions_left == b.actions_left ])
end

mover(game :: Nim2) :: Int = game.active_player

# Returns a list of the legal actions
# 1: Take a token
# 2: End turn
function legalactions(game :: Nim2) :: Vector{ActionIndex}
  if isover(game)
    ActionIndex[]
  elseif game.actions_left == 0
    [2]
  elseif game.actions_left == 3
    [1]
  else
    [1,2]
  end
end

# You can take a piece (action=1), if you still have actions left.
# You can end your turn if you have taken at least one piece
function isactionlegal(game :: Nim2, action :: ActionIndex) :: Bool
  if action == 1
    game.actions_left > 0
  elseif action == 2
    game.actions_left < 3
  else
    error("action $action not defined!")
  end
end

function move!(game :: Nim2, index :: ActionIndex) :: Nim2
  @assert isactionlegal(game, index) "Action $index is not allowed."

  if index == 2
    game.active_player = -game.active_player
    game.actions_left = 3
    game
  else
    game.remaining -= 1
    game.actions_left -= 1
    # Update the status cache
    game.status = nim_status(game, game.active_player)
    game
  end
end

# execute this function after a player takes a token to decide the game status.
function nim_status(game :: Nim2, active_player :: Int) :: Status
  # The player that takes the last tokes looses the game
  if game.remaining == 0
    Status(-active_player)
  else
    Status()
  end
end

status(game :: Nim2) :: Status = game.status

policylength(:: Type{Nim2}) :: Int = 2

# Size of the data representation of the game
# T T T T T T T T T T T T T T T A A A
# Where T represents a token on the board and M represents an action left.
Base.size(:: Type{Nim2}) :: Tuple{Int, Int, Int} = (15 + 3, 1, 1)

# Data representation of the game as layered 2d image
function array(game :: Nim2) :: Array{Float32, 3}
  repr = zeros(Float32, 15+3)
  for i in 1:game.remaining
    repr[i] = 1
  end
  for i in 1:game.actions_left
    repr[15 + i]
  end
  reshape(repr, (15+3, 1, 1))
end

function represent_actions_left2(actions_left :: Int) :: Vector{Float32}
  if actions_left == 0
    [0, 0, 0]
  elseif actions_left == 1
    [1, 0, 0]
  elseif actions_left == 2
    [1, 1, 0]
  else
    [1, 1, 1]
  end
end

hash(game :: Nim2) = Base.hash((game.remaining, game.active_player, game.actions_left))

function draw(game :: Nim2) :: Nothing
  println(" Nim: $(game.remaining) remaining, $(game.actions_left) actions left, $(game.active_player) to play.")
end
