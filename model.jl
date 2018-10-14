abstract type Model end

modelweights(::Model) = Any[]
modelstate(::Model) = Any[]

function apply(model :: M, game :: GameState) where M <: Model
    apply(modelweights(model), modelstate(model), M, game)
end

# Dummy Model

struct DummyModel <: Model
end

function apply(weights :: Vector{Any}, state :: Vector{Any}, :: Type{DummyModel}, game :: GameState) :: Array{Float32}
    Float32[ 0; ones(81)/81 ]
end

# Rollout Model

struct RolloutModel <: Model
end

# Executes random moves until the game is over and reports the result as the value
function apply(weights :: Vector{Any}, state :: Vector{Any}, :: Type{RolloutModel}, game :: GameState) :: Array{Float32}
    game_result = random_playout!(copy(game))
    Float32[ game_result * game.current_player; ones(81)/81 ]
end


# Linear Model

struct LinearModel <: Model
    W :: Matrix{Float32}
    b :: Vector{Float32}
end

LinearModel() = [ randn(Float32, (82, 81)), randn(Float32, 82) ]

function modelweights(m :: LinearModel)
    [ W, b ]
end

# W * (state.current_player * state.board ++ state.legal_regions) + b

function apply(weights :: Vector{Any}, state :: Vector{Any}, :: Type{LinearModel}, game :: GameState) :: Array{Float32}
    weights[1] * (state.current_player * state.board) + weights[2]
end

# Abstand halten!
