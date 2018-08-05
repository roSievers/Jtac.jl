println("Let's play ultimate Tic Tac Toe")

include("helpers.jl")
include("drawing.jl")

# The total board is made up of 9 single boards
# 9 consecutive entries form a single board.
# The single board is stored in the following shape:
# 1 2 3
# 4 5 6
# 7 8 9.
mutable struct GameState
    board :: Array{Int8,1}
    current_player :: Int8
    # The focus is either 0 (no focus) or indicates the allowed board
    focus :: Int8
end

function new_game()
    GameState(zeros(Int8, 81), 1, 0)
end

game = new_game()

print_board(game)

function place(game, index)
    if game.focus != 0 && outer_index(index) != game.focus
        println("You must place in the focused single board.")
        return
    end
    game.board[index] = game.current_player
    game.current_player = -game.current_player
    game.focus = inner_index(index)
    print_board(game)
end

function legal_actions(game :: GameState) :: Array{Int8}
    if game.focus == 0
        non_decided_boards = (1:9)[!single_board_decided.(game, 1:9)]
        @show non_decided_boards
        vcat(legal_actions.(game, non_decided_boards)...)
    else
        legal_actions(game, game.focus)
    end
end

# Findet im fokusiertem single board die freien Felder.
function legal_actions(game :: GameState, outer_index) :: Array{Int8}
    filter(x -> game.board[x] == 0, combined_index.(outer_index, 1:9))
end

function single_board_decided(game :: GameState, outer_index) :: Bool
    start_index = combined_index(outer_index, 1)
    single_board = game.board[start_index:start_index+8]
    @show single_board
    for i = 1:3
        if check_triple(single_board[single_index.(1:3, i)])
            return true
        elseif check_triple(single_board[single_index.(i, 1:3)])
            return true
        end
    end
    if single_board[1] == single_board[5] == single_board[9] != 0
        true
    elseif single_board[3] == single_board[5] == single_board[7] != 0
        true
    else
        false
    end
end

function check_triple(e :: Vector{Int8}) :: Bool
    e[1] == e[2] == e[3] != 0
end