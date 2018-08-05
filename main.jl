include("game.jl")
include("helpers.jl")
include("drawing.jl")
include("model.jl")
include("mc.jl")

function greeting()
    println("Let's play ultimate Tic Tac Toe")
    game = new_game()
    print_board(game)
end

greeting()

function watch_ai_game(power = 1000)
    game = new_game()
    while !game_result(game)[1]
        ai_turn!(game, power)
        print_board(game)
        println(" ")
        println(" ")
    end
end