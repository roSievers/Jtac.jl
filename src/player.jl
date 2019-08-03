
# A player is an agent that can change a game by choosing actions to perform
abstract type Player{G <: Game} end

# This method yields a probability distribution over all legal actions
think(game :: Game, p :: Player) :: Vector{Float32} = error("Not implemented")

# Randomly decide for an action based on the thought out policy
function decide(game :: Game, p :: Player) :: ActionIndex
  actions = legal_actions(game)
  actions[choose_index(think(game, p))]
end

# Convenience function to automatically alter the game
turn!(game :: Game, p :: Player) = apply_action!(game, decide(game, p))

# It is nice to have a name for each player if we want to do tournaments etc.
name(p :: Player) :: String = error("Not implemented")

# A player that always chooses random actions from allowed ones
struct RandomPlayer <: Player{Game} end

function think(game :: Game, :: RandomPlayer)
  l = length(legal_actions(game))
  ones(l) / l
end

name(p :: RandomPlayer) = "random"

gametype(:: Player{G}) where {G <: Game} = G

# A Markov chain tree search player with a model it can ask for decision making 
struct MCTSPlayer{G} <: Player{G}
  model :: Model{G}
  power :: Int
  temperature :: Float32
  exploration :: Float32
  name :: String
end

function MCTSPlayer( model :: Model{G}
                   ; power = 100
                   , temperature = 1.
                   , exploration = 1.41
                   , name = nothing 
                   ) where {G <: Game}

  if isnothing(name)
    id = Int(div(hash((model, temperature)), Int(1e14)))
    name = "mcts$(power)-$id"
  end
  MCTSPlayer{G}(model, power, temperature, exploration, name)

end

# The default MCTSPlayer uses the RolloutModel
MCTSPlayer(; kwargs...) = MCTSPlayer(RolloutModel(); kwargs...)

function think(game :: G , p :: MCTSPlayer{G}) where {G <: Game}

  mctree_policy( p.model
               , game
               , power = p.power
               , temperature = p.temperature
               , exploration = p.exploration)

end

name(p :: MCTSPlayer) = p.name

training_model(p :: MCTSPlayer) = training_model(p.model)

# Player that uses the model policy decision directly
# The temperature controls how strictly/loosely it follows the policy
struct IntuitionPlayer{G} <: Player{G}
  model :: Model{G}
  temperature :: Float32
  name :: String
end

function IntuitionPlayer( model :: Model{G}
                        ; temperature = 1.
                        , name = nothing
                        ) where {G <: Game}
  if isnothing(name)
    id = Int(div(hash((model, temperature)), Int(1e14)))
    name = "intuition-$id"
  end

  IntuitionPlayer{G}(model, temperature, name)

end

function think(game :: G, p :: IntuitionPlayer{G}) where {G <: Game}
  
  # Get all legal actions and their model policy values
  actions = legal_actions(game)
  output = p.model(game) |> Array{Float32} # convert potential gpu-output
  policy = output[actions .+ 1]
  
  # Return the action that the player decides for
  if p.temperature == 0
    probs = zeros(Float32, length(policy))
    probs[findmax(policy)[2]] = 1.
  else
    weighted_policy = policy.^(1/p.temperature)
    probs = weighted_policy / sum(weighted_policy)
  end

  probs

end

name(p :: IntuitionPlayer) = p.name

training_model(p :: IntuitionPlayer) = training_model(p.model)

# Human player that queries for interaction
# Relies on implemented draw() method for the game
struct HumanPlayer <: Player{Game}
  name :: String
end

HumanPlayer() = HumanPlayer("player")

function think(game :: Game, p :: HumanPlayer)
  println()
  draw(game)
  while true
    print("$(p.name): ")
    input = readline()
    try 
      action = parse(Int, input)
      if !is_action_legal(game, action)
        println("Action $input is illegal ($error)")
      else
        actions = legal_actions(game)
        return Float32[a == action ? 1. : 0. for a in actions]
      end
    catch error
      if isa(error, ArgumentError)
        println("Cannot parse action ($error)")
      else
        println("An unknown error occured: $error")
      end

    end
  end
end

name(p :: HumanPlayer) = p.name

# Let players play versus players

function pvp(p1 :: Player, p2 :: Player, game :: Game)
  game = copy(game)
  while !is_over(game)
    if current_player(game) == 1
      turn!(game, p1)
    else
      turn!(game, p2)
    end
  end
  status(game)
end


function derive_gametype(players)
  gt = mapreduce(gametype, typeintersect, players, init = Game)

  @assert gt != Union{} "Players do not play compatible games"
  @assert !isabstracttype(gt) "Cannot infere game from abstract type"

  gt
end

pvp(p1 :: Player, p2 :: Player) = pvp(p1, p2, derive_gametype([p1, p2])())

