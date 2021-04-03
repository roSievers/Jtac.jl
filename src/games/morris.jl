"""
Implementation of https://en.wikipedia.org/wiki/Three_men%27s_morris
 
The set of legal moves is always a subset of 1:9, but the meaning may
differ depending on the game state. It can have three meanings:
  1. Place a new stone at the given position
  2. Lift an existing stone at the given position
  3. Place the lifted stone at the given position

  X - O - .
  | \\ | / |
  . - X - O
  | / | \\ |
  O - X - .
"""
mutable struct Morris <: AbstractGame
  board :: Vector{Int} # We may want https://github.com/JuliaArrays/StaticArrays.jl
  current_player :: Int
  lifted :: Int # A number in 1:9 indicates a lifted piece, 0 indicates no lifted piece.
  status :: Status
  placements_left :: Int # Number of pieces that must still be placed
  actions_left :: Int # A move counter after which the game ends in a draw
end

Morris() where {M, N, K} = Morris(zeros(Int, 9), 1, 0, Status(), 6, 100)

function Base.copy(s :: Morris) :: Morris
  Morris(
    copy(s.board), 
    s.current_player,
    s.lifted,
    s.status,
    s.placements_left,
    s.actions_left,
  )
end

current_player(game :: Morris) :: Int = game.current_player

const adjacency = [
  [2, 4, 5], [1, 3, 5], [2, 5, 6], [1, 5, 7], [1, 2, 3, 4, 6, 7, 8, 9],
  [3, 5, 9], [4, 5, 8], [7, 5, 9], [5, 6, 8]
  ]

# Returns a list of the Indices of all legal actions
function legal_actions(game :: Morris) :: Vector{ActionIndex}
  if game.actions_left <= 0
    []
  # Are we in the "place new piece" phase?
  elseif game.placements_left > 0
    # Pre-allocate a vector with the known correct size.
    result = Vector{ActionIndex}(undef, game.placements_left + 3)
    i = 1
    for j in 1:9
      if game.board[j] == 0
        result[i] = ActionIndex(j)
        i += 1
      end
    end
    result
  elseif game.lifted == 0
    result = Vector{ActionIndex}(undef, 3)
    # All pieces may be lifted. Lifting a piece that has no moves leads to
    # a loss.
    i = 1
    for j in 1:9
      if game.board[j] == game.current_player
        if i > 3
          draw(game)
        end
        result[i] = ActionIndex(j)
        i += 1
      end
    end
    result
  else
    # Placing a piece is only allowed on connected fields
    filter(i -> game.board[i] == 0, adjacency[game.lifted])
  end
end

# Legality of an action depends on the current game phase
function is_action_legal(game :: Morris, index :: ActionIndex) :: Bool
  if game.actions_left <= 0
    # The game is already be over
    false
  elseif game.placements_left > 0
    # All empty positions are legal placement targets.
    # The game can already be over if the first player has placed all their
    # tokens in a row.
    game.board[index] == 0 && !is_over(game)
  elseif game.lifted == 0
    # The player wants to lift a piece
    # They are allowed to lift any piece of their own color.
    game.board[index] == game.current_player
  else
    # The current player has lifted a piece at game.lifted. They can place it
    # on any empty adjacent position.
    index in adjacency[game.lifted] && game.board[index] == 0 && !is_over(game)
  end
end

function apply_action!(game :: Morris, index :: ActionIndex) :: Morris
  @assert is_action_legal(game, index) "Action $index is not allowed."

  # Update the board state
  if game.placements_left > 0
    game.board[index] = game.current_player
    # update victory condition
    game.status = morris_status(game, index)
    game.current_player = -game.current_player
    game.placements_left -= 1
  elseif game.lifted == 0
    game.lifted = index
    game.board[index] = 0
    if !lift_is_legal(game, index)
      game.status = Status(-game.current_player)
    end
  else
    game.board[index] = game.current_player
    game.lifted = 0
    game.actions_left -= 1
    # update victory condition
    game.status = morris_status(game, index)
    game.current_player = -game.current_player
  end
  game
end

function lift_is_legal(game :: Morris, index :: ActionIndex) :: Bool
  # If a player lifts a piece that has no legal moves they forfeit the game.
  for i in adjacency[index]
    if game.board[i] == 0
      return true
    end
  end
  false
end

function morris_status(game :: Morris, changed_index :: ActionIndex) :: Status
  if game.actions_left <= 0
    return Status(0)
  end

  current = game.current_player

  # This is essentially a manually unrolled nested loop
  isWon = if changed_index == 1
    (game.board[2] == current && game.board[3] == current) ||
    (game.board[5] == current && game.board[9] == current) ||
    (game.board[4] == current && game.board[7] == current)
  elseif changed_index == 2
    (game.board[1] == current && game.board[3] == current) ||
    (game.board[5] == current && game.board[8] == current)
  elseif changed_index == 3
    (game.board[1] == current && game.board[2] == current) ||
    (game.board[5] == current && game.board[7] == current) ||
    (game.board[6] == current && game.board[9] == current)
  elseif changed_index == 4
    (game.board[1] == current && game.board[7] == current) ||
    (game.board[5] == current && game.board[6] == current)
  elseif changed_index == 5
    (game.board[1] == current && game.board[9] == current) ||
    (game.board[2] == current && game.board[8] == current) ||
    (game.board[3] == current && game.board[7] == current) ||
    (game.board[4] == current && game.board[6] == current)
  elseif changed_index == 6
    (game.board[3] == current && game.board[9] == current) ||
    (game.board[4] == current && game.board[5] == current)
  elseif changed_index == 7
    (game.board[1] == current && game.board[4] == current) ||
    (game.board[3] == current && game.board[5] == current) ||
    (game.board[8] == current && game.board[9] == current)
  elseif changed_index == 8
    (game.board[2] == current && game.board[5] == current) ||
    (game.board[7] == current && game.board[9] == current)
  elseif changed_index == 9
    (game.board[1] == current && game.board[5] == current) ||
    (game.board[3] == current && game.board[6] == current) ||
    (game.board[7] == current && game.board[8] == current)
  else
    throw("changed_index is $(changed_index), which is out of bounds.")
  end

  if isWon
    Status(game.current_player)
  else
    Status()
  end
end

status(game :: Morris) :: Status = game.status

policy_length(:: Type{Morris}) :: Int = 9

"""
Size of the data representation of the game
The second channel encodes game.actions_left
"""
Base.size(:: Type{Morris}) :: Tuple{Int, Int, Int} = (3, 3, 2)

# Data representation of the game as layered 2d image
function representation(game :: Morris) :: Array{Float32, 3}
  # first layer
  firstLayer = reshape(game.current_player .* game.board, (3, 3, 1))
  # countdown layer
  countdown = game.actions_left
  secondLayer = Vector{Float32}(undef, 9)
  for i in 1:9
    secondLayer[i] = countdown % 2
    countdown = countdown / 2
  end
  secondLayer = reshape(secondLayer, (3, 3, 1))

  cat(firstLayer, secondLayer, dims = 3)
end


# function augment(game :: MNKGame{M, N, K}, label :: Vector{Float32}) where {M, N, K}
#   if M == N
#     boards = apply_dihedral_group(reshape(game.board, (M, N))) 
#     games = [ MNKGame{M, N, K}(reshape(b, (M * N,)), game.current_player, game.status, game.move_count) for b in boards ]
  
#     matpol = reshape(label[2:end], (M, N))
#     matpols = apply_dihedral_group(matpol)
#     labels = [ vcat(label[1], reshape(mp, (M * N,))) for mp in matpols ]
  
#     games, labels
#   else
#     boards = apply_klein_four_group(reshape(game.board, (M, N))) 
#     games = [ MNKGame{M, N, K}(reshape(b, (M * N,)), game.current_player, game.status) for b in boards ]
  
#     matpol = reshape(label[2:end], (M, N))
#     matpols = apply_klein_four_group(matpol)
#     labels = [ vcat(label[1], reshape(mp, (M * N,))) for mp in matpols ]
  
#     games, labels
#   end
# end


function draw(game :: Morris) :: Nothing
  board = game.board
  symbols = Dict(1 => "X", -1 => "O", 0 => "⋅")

  println(" $(symbols[board[1]]) - $(symbols[board[2]]) - $(symbols[board[3]])")
  println(" | \\ | / |")
  println(" $(symbols[board[4]]) - $(symbols[board[5]]) - $(symbols[board[6]])")
  println(" | / | \\ |")
  println(" $(symbols[board[7]]) - $(symbols[board[8]]) - $(symbols[board[9]])")
end
