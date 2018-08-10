abstract type Model end

struct DummyModel <: Model
end

struct RolloutModel <: Model
end

function apply(model :: DummyModel, game :: GameState) :: Tuple{Float64, Array{Float64}}
    ( 0, ones(81)/81 )
end

function apply(model :: RolloutModel, game :: GameState) :: Tuple{Float64, Array{Float64}}
    current_player = game.current_player
    game_result = random_playout!(game)
    ( game_result * current_player, ones(81)/81 )
end