# Implements 4 in a row on a 5 x 5 board
# https://en.wikipedia.org/wiki/M,n,k-game
# According to wikipedia this is a draw given perfect play.
# We intend this game to be the next step after tic tac toe and a convenient
# stepping stone on our way to meta-tac.

mutable struct TicTac554 <: Game
  board :: Vector{Int}
  current_player :: Int
  status :: Status
end

TicTac554() = TicTac554(zeros(Int, 25), 1, Status())

function Base.copy(s :: TicTac554) :: TicTac554
  TicTac554(
    copy(s.board), 
    s.current_player,
    s.status,
  )
end

current_player(game :: TicTac554) :: Int = game.current_player

# Returns a list of the Indices of all legal actions
function legal_actions(game :: TicTac554) :: Vector{ActionIndex}
  if is_over(game)
    ActionIndex[]
  else
    ActionIndex[ index for index in 1:25 if game.board[index] == 0 ]
  end
end

# A action is legal, if the board position is still empty
function is_action_legal(game :: TicTac554, index :: ActionIndex) :: Bool
  game.board[index] == 0 && !is_over(game)
end

function apply_action!(game :: TicTac554, index :: ActionIndex) :: TicTac554
  @assert is_action_legal(game, index) "Action $index is not allowed."
  # Update the board state
  game.board[index] = game.current_player
  game.current_player = -game.current_player
  
  # Update the status cache
  game.status = tic_tac_554_status(game.board, index)
  game
end

# This function determines the status of the board. It may only be called after
# the current player has placed a stone at changed_index and it assumes that
# the game was previously undetermined.
function tic_tac_554_status(board :: Vector{Int}, changed_index :: ActionIndex) :: Status
  # Reshape input data to 5x5 matrix form
  matrix = reshape(board, (5, 5))
  ci = CartesianIndices((5, 5)) 
  x, y = Tuple(ci[changed_index])
  moving_player = board[changed_index]

  # Check row
  if matrix[2, y] == matrix[3, y] == matrix[4, y] == moving_player
    if matrix[1, y] == moving_player || matrix[5, y] == moving_player
      return Status(moving_player)
    end
  end

  # Check column
  if matrix[x, 2] == matrix[x, 3] == matrix[x, 4] == moving_player
    if matrix[x, 1] == moving_player || matrix[x, 5] == moving_player
      return Status(moving_player)
    end
  end

  # Check diagonal (going down right)
  if x == y && matrix[2, 2] == matrix[3, 3] == matrix[4, 4] == moving_player
    if matrix[1, 1] == moving_player || matrix[5, 5] == moving_player
      return Status(moving_player)
    end
  end
  
  if x == y + 1
    if matrix[2, 1] == matrix[3, 2] == matrix[4, 3] == matrix[5, 4] == moving_player
      return Status(moving_player)
    end
  end

  if x + 1 == y
    if matrix[1, 2] == matrix[2, 3] == matrix[3, 4] == matrix[4, 5] == moving_player
      return Status(moving_player)
    end
  end

  # Check diagonal (going down left / up right)
  if x + y == 6 && matrix[4, 2] == matrix[3, 3] == matrix[2, 4] == moving_player
    if matrix[5, 1] == moving_player || matrix[1, 5] == moving_player
      return Status(moving_player)
    end
  end

  if x + y == 7
    if matrix[2, 5] == matrix[3, 4] == matrix[4, 3] == matrix[5, 2] == moving_player
      return Status(moving_player)
    end
  end

  if x + y == 5
    if matrix[1, 4] == matrix[2, 3] == matrix[3, 2] == matrix[4, 1] == moving_player
      return Status(moving_player)
    end
  end

  # Check if there are empty spaces left.
  if all(x -> x != 0, matrix)
    Status(0)
  else
    Status()
  end
end

status(game :: TicTac554) :: Status = game.status

policy_length(:: Type{TicTac554}) :: Int = 25

# Size of the data representation of the game
Base.size(:: Type{TicTac554}) :: Tuple{Int, Int, Int} = (5, 5, 1)

# Data representation of the game as layered 2d image
function representation(game :: TicTac554) :: Array{Float32, 3}
  reshape(game.current_player .* game.board, (5, 5, 1))
end

function augment(game :: TicTac554, label :: Vector{Float32})
  boards = apply_dihedral_group(reshape(game.board, (5,5))) 
  games = [ TicTac554(reshape(b, (25,)), game.current_player, game.status) for b in boards ]

  matpol = reshape(label[2:end], (5, 5))
  matpols = apply_dihedral_group(matpol)
  labels = [ vcat(label[1], reshape(mp, (25,))) for mp in matpols ]

  games, labels
end


function draw(game :: TicTac554) :: Nothing
  board = reshape(game.board, (5,5))
  symbols = Dict(1 => "X", -1 => "O", 0 => "⋅")

  for i in 1:5
    for j in 1:5
      print(" $(symbols[board[j,i]])")
      if j != 5
        print(" │") 
      end
    end
    println()
    if i != 5
      println(" $(repeat("─", 17))") 
    end
  end
end
