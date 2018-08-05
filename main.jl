println("Let's play ultimate Tic Tac Toe")

include("game.jl")
include("helpers.jl")
include("drawing.jl")
include("model.jl")
include("mc.jl")

game = new_game()

print_board(game)

function place!(game, index)
    if game.focus != 0 && outer_index(index) != game.focus
        error("You must place in the focused single board.")
    end
    game.board[index] = game.current_player
    game.current_player = -game.current_player
    game.focus = inner_index(index)
    if single_board_decided(game, game.focus)
        game.focus = 0
    end
    # print_board(game)
end

random_action(game) = rand(legal_actions(game))

function place_random(game)
    place!(game, random_action(game))
end

function simulate_game()
    game = new_game()
    while !game_result(game)[1]
        place_random(game)
        # println("")
    end
end

function random_playout(game)
    while !game_result(game)[1]
        place_random(game)
    end
    game_result(game)[2]
end

function legal_actions(game :: GameState) :: Array{Int8}
    if game.focus == 0 || single_board_decided(game, game.focus)
        non_decided_boards = (1:9)[.!single_board_decided.(game, 1:9)]
        vcat(legal_actions.(game, non_decided_boards)...)
    else
        legal_actions(game, game.focus)
    end
end

# Findet im fokusiertem single board die freien Felder.
function legal_actions(game :: GameState, outer_index) :: Array{Int8}
    filter(x -> game.board[x] == 0, combined_index.(outer_index, 1:9))
end

function game_result(game) :: Tuple{ Bool, Int8 }
    inner_results = single_board_result.(game, 1:9)
    is_over = all(inner_results .|> first)
    outer_board = single_board_result.(game, 1:9) .|> second
    result = single_board_result(outer_board)
    if is_over
        (true, result[2])
    else
        result
    end
end

# Returns ( false, * ) => Game is not over yet
# Returns ( true, {-1, 0, 1} ) => Game is over and we return the winner
function single_board_result(game :: GameState, outer_index) :: Tuple{ Bool, Int8 }
    start_index = combined_index(outer_index, 1)
    single_board = game.board[start_index:start_index+8]
    single_board_result(single_board)
end

single_board_decided(game, outer_index) = single_board_result(game, outer_index)[1]

function single_board_result(board :: Array{Int8}) :: Tuple{ Bool, Int8 }
    for i = 1:3
        if check_triple(board[single_index.(1:3, i)])
            return (true, board[single_index(1, i)])
        elseif check_triple(board[single_index.(i, 1:3)])
            return (true, board[single_index(i, 1)])
        end
    end
    if board[1] == board[5] == board[9] != 0
        (true, board[5])
    elseif board[3] == board[5] == board[7] != 0
        (true, board[5])
    elseif all(board .!= 0)
        (true, 0)
    else
        (false, 0)
    end
end

function check_triple(e :: Vector{Int8}) :: Bool
    e[1] == e[2] == e[3] != 0
end


model = RolloutModel()
root = Node()
expand_tree_by_one!(root, game, model)
expand_tree_by_one!(root, game, model)
