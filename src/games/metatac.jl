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
    MetaTac(zeros(Int, 81), 1, 0, Status(nothing), [Status(nothing) for i=1:9])
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

# Returns a list of the Indices of all legal actions
function legal_actions(game :: MetaTac) :: Vector{ActionIndex}
    filter(x -> is_action_legal(game, x), 1:81)
end

# A action is legal, if the following conditions hold:
#   - The game is still active
#   - The region is available
#   - The board position is still empty
function is_action_legal(game, index) :: Bool
    !is_over(game) &&
    is_region_allowed(game, region(index)) &&
    game.board[index] == 0
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

# Copies a region into a new array.
function region_view(game, r)
    row = 3 * mod(r-1, 3) + 1
    col = 3 * div(r-1, 3) + 1
    reshape(reshape(game.board, (9, 9))[row:row + 2, col:col + 2], (9, ))
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

function status(game :: MetaTac) :: Status
    game.status_cache
end

# Returns ( false, * ) => Game is not over yet
# Returns ( true, {-1, 0, 1} ) => Game is over and we return the winner
function tic_tac_toc_status(game :: MetaTac, outer_index) :: Tuple{ Bool, Int }
    start_index = combined_index(outer_index, 1)
    single_board = game.board[start_index:start_index+8]
    tic_tac_toc_status(single_board)
end

# Implements the win condition for the large board
function tic_tac_toc_status(board :: Vector{Status}) :: Status
    tic_tac_toc_status(with_default.(board, 0))
end

function tic_tac_toc_status(board :: Vector{Int})
    tic_tac_toc_status(reshape(board, (3, 3)))
end

# Takes a 3 * 3 Array of {1, 0, -1} and returns the status for this board.
function tic_tac_toc_status(board :: Matrix{Int}) :: Status
    # Iterate rows and columns
    for i = 1:3
        if check_triple(board[:, i])
            return Status(board[1, i])
        elseif check_triple(board[i, :])
            return Status(board[i, 1])
        end
    end
    # Diagonals
    if board[1, 1] == board[2, 2] == board[3, 3] != 0
        Status(board[2, 2])
    elseif board[1,3] == board[2, 2] == board[3, 1] != 0
        Status(board[2, 2])
    elseif all(board .!= 0)
        Status(0)
    else
        Status(nothing)
    end
end

function check_triple(e :: Vector{Int}) :: Bool
    e[1] == e[2] == e[3] != 0
end