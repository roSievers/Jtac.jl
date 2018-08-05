println("Let's play ultimate Tic Tac Toe")

include("helpers.jl")
include("drawing.jl")

# 9 consecutive entries form a single board.
# The single board is stored in the following shape:
# 1 2 3
# 4 5 6
# 7 8 9.
mutable struct GameState
    board :: Array{Int8,1}
    current_player :: Int8
end

game = GameState(zeros(Int8, 81), 1)

print_board(game.board)

function place(game, index)
    game.board[index] = game.current_player
    game.current_player = 3 - game.current_player
    print_board(game.board)
end