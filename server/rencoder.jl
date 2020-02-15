
# -------- Explicitly Typed values for automated json encoding --------------- #

struct Typed{T}
  typ :: String
  value :: T 

  # Overwrite default constructor
  Typed(value) = new{typeof(value)}(typeof(value) |> string |> lowercase, value)
end


# -------- Individual Layers that compose an Image --------------------------- #

abstract type Layer end

const Point = Tuple{Float32, Float32}

struct Lines <: Layer
  data :: Vector{Tuple{Point, Point}}
  name :: String
  style :: String
end

Lines(data; name = "lines", style = "black") = Lines(data, name, style)

function Lines(xs, ys; kwargs...)
  data = map(xs, ys) do x, y
    ((x[1], y[1]), (x[2], y[2]))
  end
  Lines(data; kwargs...)
end

struct Heatmap <: Layer
  data :: Vector{Float32}
  name :: String
  style :: String
  min :: Float32
  max :: Float32
end

function Heatmap( data
                ; min = minimum(data)
                , max = maximum(data)
                , name = "heatmap"
                , style = "grey" )
  Heatmap(data, name, style, min, max)
end

struct Tokens <: Layer
  data :: Vector{String}
  name :: String
  style :: String
end

function Tokens(game :: Union{MNKGame, MetaTac}; name = "tokens", style = "black")
  tokens = Dict(1  => "X", -1 => "O", 0  => "")
  Tokens(map(x -> tokens[x], game.board), name, style)
end


struct Actions <: Layer
  data :: Vector{Int}
  name :: String
end

function Actions(game :: Union{MNKGame, MetaTac}; name = "actions")
  actions = legal_actions(game)
  action_data = -ones(Int, policy_length(game))
  action_data[actions] = actions
  Actions(action_data, name)
end


# -------- An image as a stack of layers with some geometry ------------------ #

struct Image
  layers
  width :: Int
  height :: Int
  name :: String
  value :: Union{Float32, Nothing}

  # Overwrite default constructor
  Image(layers :: Vector{Layer}, args...) = new(Typed.(layers), args...)
end

function Image( game :: Union{MNKGame, MetaTac}
              , policy :: Vector{Float32}
              , value :: Float32 )

  m, n = size(game)[1:2]

  heatmap = Heatmap(policy, min = 0., max = 1.)
  name = typeof(game) |> string |> lowercase

  layers = [heatmap, Tokens(game), Actions(game)]

  if isa(game, MetaTac)
    x = Point[(3, 3), (6, 6), (0, 9), (0, 9)] 
    y = Point[(0, 9), (0, 9), (3, 3), (6, 6)]
    push!(layers, Lines(x, y))
  end

  Image(layers, m, n, name, value)
end

