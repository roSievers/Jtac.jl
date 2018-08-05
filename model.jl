abstract type Model end

struct DummyModel <: Model
end

struct RolloutModel <: Model
end

function apply(model :: DummyModel, game :: GameState) :: Tuple{Float64, Array{Float64}}
    ( 0, ones(81)/81 )
end

function apply(model :: RolloutModel, game :: GameState) :: Tuple{Float64, Array{Float64}}
    game_result = random_playout(game)
    ( game_result * game.current_player, ones(81)/81 )
end