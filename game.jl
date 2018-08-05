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

function Base.copy(s :: GameState) :: GameState
    GameState(copy(s.board), s.current_player, s.focus)
end