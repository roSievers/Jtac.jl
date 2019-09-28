# Implements a generic m, n, k game.
# https://en.wikipedia.org/wiki/M,n,k-game
# m: width
# n: height
# k: stones in a row required to win a game.

mutable struct MNKGame{M, N, K} <: Game
  board :: Vector{Int} # We may want https://github.com/JuliaArrays/StaticArrays.jl
  current_player :: Int
  status :: Status
  move_count :: Int
end

const TicTacToe = MNKGame{3, 3, 3}

MNKGame{M, N, K}() where {M, N, K} = MNKGame{M, N, K}(zeros(Int, M * N), 1, Status(), 0)

function Base.copy(s :: MNKGame{M, N, K}) :: MNKGame{M, N, K} where {M, N, K}
  MNKGame{M, N, K}(
    copy(s.board), 
    s.current_player,
    s.status,
    s.move_count,
  )
end

current_player(game :: MNKGame) :: Int = game.current_player

# Returns a list of the Indices of all legal actions
function legal_actions(game :: MNKGame{M, N}) :: Vector{ActionIndex} where {M, N}
  if is_over(game)
    ActionIndex[]
  else
    result = Vector{ActionIndex}(undef, M*N - game.move_count)
    i = 1
    for j in 1:(M*N)
      if game.board[j] == 0
        result[i] = ActionIndex(j)
        i += 1
      end
    end
    @assert size(result)[1] + 1 == i "There should be $(size(result)[1] + 1) legal actions but we found $(i)."

    result
  end
end

# A action is legal, if the board position is still empty
function is_action_legal(game :: MNKGame{M, N}, index :: ActionIndex) :: Bool where {M, N}
  game.board[index] == 0 && !is_over(game)
end

function apply_action!(game :: MNKGame{M, N, K}, index :: ActionIndex) :: MNKGame{M, N, K} where {M, N, K}
  @assert is_action_legal(game, index) "Action $index is not allowed."
  # Update the board state
  game.board[index] = game.current_player
  game.current_player = -game.current_player
  
  # Update the status cache
  game.move_count += 1
  game.status = tic_tac_mnk_status(game, index)
  game
end

# All possible directions for winning rows, in flat index notation.
const search_directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

function bound_check(m, n, fx, fy)
  fx > 0 && fx <= m && fy > 0 && fy <= n
end

function tic_tac_mnk_status(game :: MNKGame{M, N, K}, changed_index :: ActionIndex) where {M, N, K}
  ci = CartesianIndices((M, N)) 
  x, y = Tuple(ci[changed_index])
  moving_player = game.board[changed_index]

  for (dx, dy) in search_directions
    count = 1 # There is always the freshly placed token.
    # forward search along the direction
    # Seek using a focus point at (fx, fy).
    (fx, fy) = (x + dx, y + dy)
    while bound_check(M, N, fx, fy) && game.board[fx + (fy - 1) * M] == moving_player
      count += 1
      fx += dx
      fy += dy
    end
    # backward search along the direction
    (fx, fy) = (x - dx, y - dy)
    while bound_check(M, N, fx, fy) && game.board[fx + (fy - 1) * M] == moving_player
      count += 1
      fx -= dx
      fy -= dy
    end

    if count >= K
      return Status(moving_player)
    end
  end

  # Check if there are empty spaces left.
  if game.move_count == M*N
    Status(0)
  else
    Status()
  end
end

status(game :: MNKGame) :: Status = game.status

function policy_length(:: Type{<:MNKGame{M, N}}) :: Int where {M, N}
  M * N
end

# Size of the data representation of the game
function Base.size(:: Type{<:MNKGame{M, N}}) :: Tuple{Int, Int, Int} where {M, N}
  (M, N, 1)
end

# Data representation of the game as layered 2d image
function representation(game :: MNKGame{M, N}) :: Array{Float32, 3} where {M, N}
  reshape(game.current_player .* game.board, (M, N, 1))
end


function augment(game :: MNKGame{M, N, K}, label :: Vector{Float32}) where {M, N, K}
  if M == N
    boards = apply_dihedral_group(reshape(game.board, (M, N))) 
    games = [ MNKGame{M, N, K}(reshape(b, (M * N,)), game.current_player, game.status, game.move_count) for b in boards ]
  
    matpol = reshape(label[2:end], (M, N))
    matpols = apply_dihedral_group(matpol)
    labels = [ vcat(label[1], reshape(mp, (M * N,))) for mp in matpols ]
  
    games, labels
  else
    boards = apply_klein_four_group(reshape(game.board, (M, N))) 
    games = [ MNKGame{M, N, K}(reshape(b, (M * N,)), game.current_player, game.status) for b in boards ]
  
    matpol = reshape(label[2:end], (M, N))
    matpols = apply_klein_four_group(matpol)
    labels = [ vcat(label[1], reshape(mp, (M * N,))) for mp in matpols ]
  
    games, labels
  end
end


function draw(game :: MNKGame{M, N}) :: Nothing where {M, N}
  board = reshape(game.board, (M,N))
  symbols = Dict(1 => "X", -1 => "O", 0 => "⋅")

  for i in 1:N
    for j in 1:M
      print(" $(symbols[board[j,i]])")
      if j != M
        print(" │")
      end
    end
    println()
    if i != N
      println("───$(repeat("┼───", M - 1))")
    end
  end
end
