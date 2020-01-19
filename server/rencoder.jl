
using JSON2 
using Jtac

# Explicitly typed values for json encoding

struct Typed{T}
  typ :: String
  value :: T 
end

Typed(value :: T) where {T} = Typed{T}(string(T) |> lowercase, value)

convert(::Type{Typed{T}}, v :: T) where {T} = Typed(v)

# Images and Layers

abstract type Layer end

struct Image
  layers :: Vector{Typed{L} where {L <: Layer}} 
  width :: Int
  height :: Int
  name :: String
  value :: Union{Float32, Nothing}
end

Image(layers :: Vector{Layer}, kwargs...) = Image(Typed.(layers), kwargs...)

# Specific layers

struct Heatmap <: Layer
  data :: Vector{Float32}
  name :: String
  style :: String
  min :: Float32
  max :: Float32
end

struct Tokens <: Layer
  data :: Vector{String}
  name :: String
  style :: String
end

struct Actions <: Layer
  data :: Vector{Int}
  name :: String
end

# Game states and network outputs to Images

function Image(game :: TicTacToe, policy :: Vector{Float32}, value :: Float32)
  tokens = map(game.board) do x
    x == 1 ? "X" : (x == -1 ? "O" : "")
  end

  actions = legal_actions(game)
  action_data = -ones(Int, length(game.board))
  action_data[actions] = actions

  Image([ Heatmap(policy, "policy", "greyscale", 0., 1.)
        , Tokens(tokens, "tokens", "black")
        , Actions(action_data, "actions") ], 3, 3, "TicTacToe", value)
end

