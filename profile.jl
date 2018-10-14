
using Profile
using ProfileView

function profile()
    @time ai_turn!(new_game(), 1000)
    game = new_game()
    @profile ai_turn!(game)
    println("loading profile view")
    ProfileView.view()
end


