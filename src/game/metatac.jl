
"""
Meta tic tac toe implementation.  
"""
mutable struct MetaTac <: AbstractGame
  board :: Vector{Int}
  active_player :: Int
  # The focus is either 0 (no focus) or indicates the allowed board
  focus :: Int
  # Stores the Status for the whole game
  status_cache :: Status
  # Stores the status for each region
  region_status_cache :: Vector{Status}
end

MetaTac() = MetaTac(zeros(Int, 81), 1, 0, Game.undecided, [Game.undecided for i=1:9])

isaugmentable(:: Type{MetaTac}) = true

function Base.copy(s :: MetaTac) :: MetaTac
  MetaTac(
    copy(s.board), 
    s.active_player, 
    s.focus, 
    s.status_cache,
    copy(s.region_status_cache)
  )
end

function Base.:(==)(a::MetaTac, b::MetaTac)
  all([ all(a.board .== b.board)
      , a.active_player == b.active_player
      , a.focus == b.focus
      , a.status_cache == b.status_cache
      , all(a.region_status_cache .== b.region_status_cache) ])
end

activeplayer(game :: MetaTac) :: Int = game.active_player

# Returns a list of the Indices of all legal actions
function legalactions(game :: MetaTac) :: Vector{ActionIndex}
  ActionIndex[ index for index in 1:81 if isactionlegal(game, index) ]
end

# A action is legal, if the following conditions hold:
#   - The game is still active
#   - The region is available
#   - The board position is still empty
function isactionlegal(game :: MetaTac, index :: ActionIndex) :: Bool
  game.board[index] == 0 &&
  is_region_allowed(game, region(index)) &&
  !isover(game)
end

# Determines, whether a given region can be played in.
function is_region_allowed(game, r) :: Bool
  (game.focus == 0 || game.focus == r) && !isover(game.region_status_cache[r])
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

function move!(game :: MetaTac, index :: ActionIndex) :: MetaTac
  @assert isactionlegal(game, index) "Action $index is not allowed."
  # Update the board state
  game.board[index] = game.active_player
  game.active_player = -game.active_player
  
  # Update the status cache
  r = region(index)
  game.region_status_cache[r] = tic_tac_toc_status(region_view(game, r))
  game.status_cache = tic_tac_toc_status(game.region_status_cache)

  # Update the focus. This must be done AFTER the status cache is refreshed.
  game.focus = local_index(index)
  if isover(game.region_status_cache[game.focus])
    game.focus = 0
  end
  game
end

status(game :: MetaTac) :: Status = game.status_cache

# Returns ( false, * ) => Game is not over yet
# Returns ( true, {-1, 0, 1} ) => Game is over and we return the winner
function tic_tac_toc_status(game :: MetaTac, outer_index) :: Tuple{Bool, Int}
  start_index = combined_index(outer_index, 1)
  single_board = game.board[start_index:start_index+8]
  tic_tac_toc_status(single_board)
end

# Implements the win condition for the large board
# We need to additionaly verify, that there are legal actions left
function tic_tac_toc_status(board :: Vector{Status}) :: Status
  matrix = map(board) do s
    isover(s) ? Int(s) : 0
  end
  matrix = reshape(matrix, (3, 3))
  s = tic_tac_toc_status(matrix)
  if isover(s)
    s
  elseif all(isover, board) #all(isover.(board))
    Game.draw
  else
    Game.undecided
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
    Game.draw
  else
    Game.undecided
  end
end

policylength(:: Type{MetaTac}) :: Int = 81

# Size of the data representation of the game
Base.size(:: Type{MetaTac}) :: Tuple{Int, Int, Int} = (9, 9, 3)

# Data representation of the game as layered 2d image
function array(game :: MetaTac) :: Array{Float32, 3}
  data = zeros(Float32, 81, 3)
  board = game.active_player .* game.board
  data[board .== 1, 1] .= 1
  data[board .== -1, 2] .= 1
  data[legalactions(game), 3] .= 1
  reshape(data, (9, 9, 3))
end

function augment(game :: MetaTac)

  boards = applygroup(DihedralGroup(), reshape(game.board, (9,9))) 
  caches = applygroup(DihedralGroup(), reshape(game.region_status_cache, (3,3)))

  if game.focus == 0
    focii = fill(0, length(boards))
  else
    focii = applygroup(DihedralGroup(), game.focus, (3,3))
  end

  map(boards, caches, focii) do board, cache, focus
    board = reshape(board, (81,))
    cache = reshape(cache, (9,))
    MetaTac(board, game.active_player, focus, game.status_cache, cache)
  end

end

function augment(game :: MetaTac, label :: Vector{Float32})

  matpol = reshape(label, (9, 9))
  matpols = applygroup(DihedralGroup(), matpol)
  labels = [ reshape(mp, (81,)) for mp in matpols ]

  augment(game), labels
end

function Base.hash(game :: MetaTac)
  Base.hash((game.board, game.active_player, game.focus))
end

function Base.isequal(a :: MetaTac, b :: MetaTac)
  all(a.board .== b.board) &&
  a.active_player == b.active_player &&
  a.focus == b.focus
end

function visualize(io :: IO, game :: MetaTac) :: Nothing

  board = reshape(game.board, (9,9))
  symbols = Dict(1 => "X", -1 => "O", 0 => "⋅")

  for i in 1:9
    for j in 1:9
      print(io, " $(symbols[board[j,i]])")
      if j == 3 || j == 6 
        print(io, " ║") 
      end
    end
    if i != 9
      println(io)
    end
    if i == 3 || i == 6  
      println(io, "═══════╬═══════╬═══════") 
    end
  end
end

function Base.show(io :: IO, game :: MetaTac)
  moves = count(!isequal(0), game.board)
  m = moves == 1 ? "1 move" : "$moves moves"
  if isover(game)
    print(io, "MetaTac($m, $(status(game)) won)")
  else
    print(io, "MetaTac($m, $(activeplayer(game)) moving)")
  end
end

function Base.show(io :: IO, :: MIME"text/plain", game :: MetaTac)
  moves = count(!isequal(0), game.board)
  m = moves == 1 ? "1 move" : "$moves moves"
  s = "MetaTac game with $m and "
  if isover(game)
    println(io, s, "result $(status(game)):")
  else
    println(io, s, "player $(activeplayer(game)) moving:")
  end
  visualize(io, game)
end
