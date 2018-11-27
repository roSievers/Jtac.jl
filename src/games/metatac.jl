# The total board is made up of 9 single boards
mutable struct MetaTac <: Game
  board :: Vector{Int}
  current_player :: Int
  # The focus is either 0 (no focus) or indicates the allowed board
  focus :: Int
  # Stores the Status for the whole game
  status_cache :: Status
  # Stores the status for each region
  region_status_cache :: Vector{Status}
end

function MetaTac()
  MetaTac(zeros(Int, 81), 1, 0, Status(), [Status() for i=1:9])
end

function Base.copy(s :: MetaTac) :: MetaTac
  MetaTac(
    copy(s.board), 
    s.current_player, 
    s.focus, 
    s.status_cache,
    copy(s.region_status_cache)
  )
end

current_player(game :: MetaTac) :: Int = game.current_player

# Returns a list of the Indices of all legal actions
function legal_actions(game :: MetaTac) :: Vector{ActionIndex}
  ActionIndex[ index for index in 1:81 if is_action_legal(game, index) ]
end

# A action is legal, if the following conditions hold:
#   - The game is still active
#   - The region is available
#   - The board position is still empty
function is_action_legal(game :: MetaTac, index :: ActionIndex) :: Bool
  game.board[index] == 0 &&
  is_region_allowed(game, region(index)) &&
  !is_over(game)
end

# Determines, whether a given region can be played in.
function is_region_allowed(game, r) :: Bool
  (game.focus == 0 || game.focus == r) && !is_over(game.region_status_cache[r])
end

# Finds the region 1d index for a given cell 1d index, value is in 1:9
function region(index) :: Int
  col = mod(div(index - 1, 3), 3)
  row = div(index-1, 27)
  1 + col + 3 * row
end

# Finds the inner index for a given cell index, value is in 1:9
function local_index(index) :: Int
  col = mod1(index, 3)
  row = mod(div(index-1, 9), 3)
  col + 3 * row
end

# Copies a region into a new matrix.
function region_view(game, r)
  row = 3 * mod(r-1, 3) + 1
  col = 3 * div(r-1, 3) + 1
  reshape(game.board, (9, 9))[row:row + 2, col:col + 2]
end

function apply_action!(game :: MetaTac, index :: ActionIndex) :: Nothing
  @assert is_action_legal(game, index) "Action $index is not allowed."
  # Update the board state
  game.board[index] = game.current_player
  game.current_player = -game.current_player
  
  # Update the status cache
  r = region(index)
  game.region_status_cache[r] = tic_tac_toc_status(region_view(game, r))
  game.status_cache = tic_tac_toc_status(game.region_status_cache)

  # Update the focus. This must be done AFTER the status cache is refreshed.
  game.focus = local_index(index)
  if is_over(game.region_status_cache[game.focus])
    game.focus = 0
  end
  nothing
end

status(game :: MetaTac) :: Status = game.status_cache

# Returns ( false, * ) => Game is not over yet
# Returns ( true, {-1, 0, 1} ) => Game is over and we return the winner
function tic_tac_toc_status(game :: MetaTac, outer_index) :: Tuple{ Bool, Int }
  start_index = combined_index(outer_index, 1)
  single_board = game.board[start_index:start_index+8]
  tic_tac_toc_status(single_board)
end

# Implements the win condition for the large board
# We need to additionaly verify, that there are legal actions left
function tic_tac_toc_status(board :: Vector{Status}) :: Status
  matrix = reshape(with_default.(board, 0), (3, 3))
  s = tic_tac_toc_status(matrix)
  if is_over(s)
    s
  elseif all(is_over, board) #all(is_over.(board))
    Status(0)
  else
    Status()
  end
end

# Takes a 3 * 3 Array of {1, 0, -1} and returns the status for this board.
function tic_tac_toc_status(board :: Matrix{Int}) :: Status
  # Iterate rows and columns
  for i = 1:3
    if board[1,i] == board[2,i] == board[3,i] != 0
      return Status(board[1, i])
    elseif board[i,1] == board[i,2] == board[i,3] != 0
      return Status(board[i, 1])
    end
  end
  # Diagonals
  if board[1, 1] == board[2, 2] == board[3, 3] != 0
    Status(board[2, 2])
  elseif board[1, 3] == board[2, 2] == board[3, 1] != 0
    Status(board[2, 2])
  elseif all(x -> x != 0, board)
    Status(0)
  else
    Status()
  end
end

policy_length(:: Type{MetaTac}) :: UInt = 81

# Data representation of the game as layered 2d image
function representation(game :: MetaTac) :: Array{Float32, 3}
  data = zeros(Float32, 81, 2)
  data[:, 1] = game.current_player .* game.board
  data[legal_actions(game), 2] .= 1
  reshape(data, (9, 9, 2))
end

# Size of the data representation of the game
Base.size(:: MetaTac) :: Tuple{Int, Int, Int} = (9, 9, 2)


function draw(game :: MetaTac) :: Nothing

  board = reshape(game.board, (9,9))
  symbols = Dict(1 => "X", -1 => "O", 0 => "⋅")

  for i in 1:9
    for j in 1:9
      print(" $(symbols[board[j,i]])")
      if j == 3 || j == 6 
        print(" │") 
      end
    end
    println()
    if i == 3 || i == 6  
      println(" $(repeat("─", 21))") 
    end
  end
end


