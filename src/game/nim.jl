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

mutable struct Nim <: AbstractGame
  board :: Vector{Int}
  active_player :: Int
  actions_left :: Int
  status :: Status
end

Nim() = Nim(ones(Int, 15), 1, 3, Status())

function Base.copy(s :: Nim) :: Nim
  Nim(
    copy(s.board),
    s.active_player,
    s.actions_left,
    s.status,
  )
end


function Base.:(==)(a::Nim, b::Nim)
  all([ all(a.board .== b.board)
      , a.active_player == b.active_player
      , a.status == b.status
      , a.actions_left == b.actions_left ])
end


mover(game :: Nim) :: Int = game.active_player


# Returns a list of the Indices of all legal actions
function legalactions(game :: Nim) :: Vector{ActionIndex}
  if isover(game)
    ActionIndex[]
  elseif game.actions_left == 0
    [16]
  else
    tokens_left = ActionIndex[ index for index in 1:15 if game.board[index] == 1 ]
    if game.actions_left == 3
      tokens_left
    else
      [tokens_left; 16]
    end
  end
end

# A action is legal, if the board position is still empty
function isactionlegal(game :: Nim, index :: ActionIndex) :: Bool
  if index == 16
    game.actions_left < 3 && !isover(game)
  elseif game.actions_left > 0
    game.board[index] == 1 && !isover(game)
  else
    false
  end
end

function move!(game :: Nim, index :: ActionIndex) :: Nim
  @assert isactionlegal(game, index) "Action $index is not allowed."

  if index == 16
    game.active_player = -game.active_player
    game.actions_left = 3
    game
  else
    game.board[index] = 0
    game.actions_left -= 1
    # Update the status cache
    game.status = nim_status(game.board, game.active_player)
    game
  end
end

# execute this function after a player takes a token to decide the game status.
function nim_status(board :: Vector{Int}, active_player :: Int) :: Status
  # The player that takes the last tokes looses the game
  if iszero(board)
    Status(-active_player)
  else
    Status()
  end
end

status(game :: Nim) :: Status = game.status

policylength(:: Type{Nim}) :: Int = 16

# Size of the data representation of the game
# T T T T T T T T T T T T T T T A A A
# Where T represents a token on the board and M represents an action left.
Base.size(:: Type{Nim}) :: Tuple{Int, Int, Int} = (15 + 3, 1, 1)

# Data representation of the game as layered 2d image
function array(game :: Nim) :: Array{Float32, 3}
  reshape([game.board; represent_actions_left(game.actions_left)], (15+3, 1, 1))
end

function represent_actions_left(actions_left :: Int) :: Vector{Float32}
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

# TODO: Define a probabilistic augment method.

hash(game :: Nim) = Base.hash(game.board)

function draw(game :: Nim) :: Nothing
  tokenSymbols = Dict(1 => "T", 0 => "_")
  actionSymbols = Dict(1 => "A", 0 => "_")

  for i in 1:15
    print(" $(tokenSymbols[game.board[i]])")
  end
  print(" |")
  actions = represent_actions_left(game.actions_left)
  for i in 1:3
    print(" $(actionSymbols[actions[i]])")
  end
  println()
end
