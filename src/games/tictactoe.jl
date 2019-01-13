# Implements Tic Tac Toe as a simple game for debugging purposes

mutable struct TicTacToe <: Game
  board :: Vector{Int}
  current_player :: Int
  status :: Status
end

TicTacToe() = TicTacToe(zeros(Int, 9), 1, Status())

function Base.copy(s :: TicTacToe) :: TicTacToe
  TicTacToe(
    copy(s.board), 
    s.current_player,
    s.status,
  )
end

current_player(game :: TicTacToe) :: Int = game.current_player

# Returns a list of the Indices of all legal actions
function legal_actions(game :: TicTacToe) :: Vector{ActionIndex}
  if is_over(game)
    ActionIndex[]
  else
    ActionIndex[ index for index in 1:9 if game.board[index] == 0 ]
  end
end

# A action is legal, if the board position is still empty
function is_action_legal(game :: TicTacToe, index :: ActionIndex) :: Bool
  game.board[index] == 0 &&
  !is_over(game)
end

function apply_action!(game :: TicTacToe, index :: ActionIndex) :: TicTacToe
  @assert is_action_legal(game, index) "Action $index is not allowed."
  # Update the board state
  game.board[index] = game.current_player
  game.current_player = -game.current_player
  
  # Update the status cache
  game.status = tic_tac_toe_status(game.board)
  game
end

function tic_tac_toe_status(board :: Vector{Int}) :: Status
  matrix = reshape(board, (3, 3))
  # Iterate rows and columns
  for i = 1:3
    if matrix[1,i] == matrix[2,i] == matrix[3,i] != 0
      return Status(matrix[1, i])
    elseif matrix[i,1] == matrix[i,2] == matrix[i,3] != 0
      return Status(matrix[i, 1])
    end
  end
  # Diagonals
  if matrix[1, 1] == matrix[2, 2] == matrix[3, 3] != 0
    Status(matrix[2, 2])
  elseif matrix[1, 3] == matrix[2, 2] == matrix[3, 1] != 0
    Status(matrix[2, 2])
  elseif all(x -> x != 0, matrix)
    Status(0)
  else
    Status()
  end
end

status(game :: TicTacToe) :: Status = game.status

policy_length(:: Type{TicTacToe}) :: Int = 9

# Size of the data representation of the game
Base.size(:: Type{TicTacToe}) :: Tuple{Int, Int, Int} = (3, 3, 1)

# Data representation of the game as layered 2d image
function representation(game :: TicTacToe) :: Array{Float32, 3}
  reshape(game.current_player .* game.board, (3, 3, 1))
end

hmirror(matrix) = matrix[end:-1:1, :]
vmirror(matrix) = matrix[:, end:-1:1]

function apply_dihedral_group(matrix)
  [
    matrix |> copy,
    matrix |> hmirror,
    matrix |> transpose,
    matrix |> vmirror,
    matrix |> transpose |> hmirror,
    matrix |> hmirror |> transpose,
    matrix |> vmirror |> hmirror,
    matrix |> vmirror |> hmirror |> vmirror
  ]
end

function augment(game :: TicTacToe, label :: Vector{Float32})
  boards = apply_dihedral_group(reshape(game.board, (3,3))) 
  games = [ TicTacToe(reshape(b, (9,)), game.current_player, game.status) for b in boards ]

  matpol = reshape(label[2:end], (3, 3))
  matpols = apply_dihedral_group(matpol)
  labels = [ vcat(label[1], reshape(mp, (9,))) for mp in matpols ]

  games, labels
end


function draw(game :: TicTacToe) :: Nothing
  board = reshape(game.board, (3,3))
  symbols = Dict(1 => "X", -1 => "O", 0 => "⋅")

  for i in 1:3
    for j in 1:3
      print(" $(symbols[board[j,i]])")
      if j != 3
        print(" │") 
      end
    end
    println()
    if i != 3
      println(" $(repeat("─", 9))") 
    end
  end
end
