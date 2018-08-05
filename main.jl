println("Let's play ultimate Tic Tac Toe")

include("helpers.jl")
include("drawing.jl")

# 9 consecutive entries form a single board.
# The single board is stored in the following shape:
# 1 2 3
# 4 5 6
# 7 8 9.
board = zeros(Int8, 81)

board[1] = 1
board[9] = 2

print_board(board)