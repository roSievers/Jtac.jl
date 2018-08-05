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

function profile()
    @time ai_turn!(new_game(), 1000)
    game = new_game()
    @profile ai_turn!(game)
    println("loading profile view")
    using ProfileView
    ProfileView.view()
end

function mctree_vs_random(power = 1000)
    game = new_game()
    while !game_result(game)[1]
        if game.current_player == 1
            ai_turn!(game, power)
        else
            random_turn!(game)
        end
        # print_board(game)
        # println(" ")
        # println(" ")
    end
    game_result(game)[2]
end