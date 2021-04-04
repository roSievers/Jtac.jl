
# -------- Status ------------------------------------------------------------ #

const Status = Int # for code readability

"""
    Status()
    Status(val)

Create a representation of the game status. Calling the function without
arguments stands for a game status that is still undecided. Calling it with
argument `val`, which can be 1, -1, or 0, indicates a game won by the first or
second player, or a draw.
"""
function Status end

Status(value :: Int) = value
Status() = 42

is_over(s :: Status) = (s != 42)
with_default(s :: Status, default :: Int) = is_over(s) ? s : default


# -------- Game -------------------------------------------------------------- #

const ActionIndex = Int # for code readability

"""
A Game with two competing players that play alternately.

The game can only end by victory of the first or second player, or by draw.
Concrete subtypes of `Game` must implement a number of functions that allow
to find out the current player, the game status, or the legal actions.
"""
abstract type AbstractGame end

Broadcast.broadcastable(game :: AbstractGame) = Ref(game)
 
Base.copy(:: AbstractGame) = error("unimplemented") 
Base.size(:: Type{AbstractGame}) = error("unimplemented") # must be a 3-tuple
Base.size(:: G) where {G <: AbstractGame} = size(G)

"""
    status(game)

Status of `game`.
"""
status(game :: AbstractGame) = error("unimplemented")

"""
    current_player(game)

Current player of `game`. 1 stands for the first player and -1 for
the second.
"""
current_player(game :: AbstractGame) = error("unimplemented")

"""
    legal_actions(game)

Vector of legal actions of `game`.
"""
legal_actions(:: AbstractGame) :: Vector{ActionIndex} = error("unimplemented")

"""
    apply_action!(game, action)

Modify `game` by applying `action`.
"""
apply_action!(:: AbstractGame, :: ActionIndex) :: AbstractGame = error("unimplemented")

"""
    is_over(game)

Returns whether a game is finished or not.
"""
is_over(game :: AbstractGame) = is_over(status(game))

# Data representation of the game as layered 2d image from the perspective of
# the active player (active player plays with 1, other with -1)
"""
    representation(game)
    representation(games)

Data representation of `game` as three-dimensional array. If a vector
of `games` is given, a four-dimensional array is returned.
"""
representation(:: AbstractGame) = error("unimplemented")

function representation(games :: Vector{G}) where G <: AbstractGame

  @assert !isempty(games) "Invalid representation of empty game vector"

  results = zeros(Float32, (size(games[1])..., length(games)))

  for i in 1:length(games)
    results[:,:,:,i] = representation(games[i])
  end

  results

end

"""
    policy_length(game)
    policy_length(gametype)

Maximal number of legal actions.
"""
policy_length(:: Type{AbstractGame}) :: Int = error("unimplemented")
policy_length(:: G) where {G <: AbstractGame} = policy_length(G)


"""
    random_action(game)

Random legal action for `game`.
"""
random_action(game :: AbstractGame) = rand(legal_actions(game))

"""
    random_turn!(game)

Modify `game` by taking a single random action if the game is not finished yet.
"""
function random_turn!(game :: AbstractGame)
  is_over(game) ? game : apply_action!(game, random_action(game))
end

"""
    random_turns!(game, steps)
    random_turns!(game, steprange)

Modify `game` by taking several random actions.
"""
function random_turns!(game :: AbstractGame, steps :: Int)
  for _ in 1:steps
    random_turn!(game)
  end
  game
end

function random_turns!(game :: AbstractGame, range)
  steps = rand(range) :: Int
  random_turns!(game, steps)
end


"""
    random_playout!(game)

Play off `game` with random actions by both players.
"""
function random_playout!(game :: AbstractGame)

  while !is_over(game)
    random_turn!(game)
  end

  game

end

"""
    random_playout(game)

Random playout with initial state `game` without modifying it.
"""
function random_playout(game :: AbstractGame)

  random_playout!(copy(game))

end

"""
    is_action_legal(game, action)

Check if `action` is a legal action for `game`.
"""
function is_action_legal(game :: AbstractGame, action :: ActionIndex)

  action in legal_actions(game)

end

"""
    augment(game, label)

Returns a tuple `(games, labels)` that are augmented versions of `game` and
`label` via the game's symmetry group.
"""
function augment(game :: AbstractGame, label :: Array{Float32}) 
  @warn "Augmentation not implemented for type $(typeof(game))" maxlog = 1
  [game], [label]
end


"""
    draw([io,] game)

Draw a unicode representation of `game`.
"""
draw(io :: IO, game :: AbstractGame) :: Nothing = error("drawing $(typeof(game)) not implemented.")
draw(game) = draw(stdout, game)

