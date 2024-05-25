
"""
Enum that represents the status of a game. Can be `loss`, `draw`, `win`, or
`undecided`.
"""
@enum Status loss=-1 draw=0 win=1 undecided=42

@pack Status in AliasFormat{Int}

isover(s :: Status) = (s != undecided)

"""
Alias for [`Int`](@ref).
"""
const ActionIndex = Int # for code readability

"""
Abstract type for board games with two competing players that play in turns.

In order to guarantee basic functionality, the following minimal interface
must be implemented by any concrete subtype `G` of `AbstractGame`:
- `status(game :: G)`: Return the game status. See [`Status`](@ref).
- `mover(game :: G)`: Return the moving player indicator (-1 or 1).
- `legalactions(game :: G)`: Return a vector of legal action indices.
- `move!(game :: G, action)`: Apply the action defined by action index \
  `action` to `game`.
- `instance(G)`: Return an (empty) instance of `G`. Defaults to `G()`.
- `Base.copy(game :: G)`: Return a copy of `game`.

For neural network support, the following methods must additionally be
implemented:
- `policylength(:: Type{G})`: Length of the game policy vector. Equals the \
  maximal possible action index.
- `array(game :: G)`: Return a 3d array representation of `game`. Used as \
  input for neural network models (see [`Model.NeuralModel`](@ref)).
- `Base.size(:: Type{G})`: The size of the array representation of games of \
  type `G`.

Further methods may be specialized to improve performance or functionality.
- `array!(buffer, games :: Vector{G})`: In-place mutating version of `array` \
  for vectors of games. 
- `randominstance(:: Type{G})`: Return a random instance of game type `G`.
- `isaugmentable(:: Type{G}) / augment(game :: G)`: Support for data \
  augmentation by exploiting game symmetries.
- `isover(game :: G)`: Check whether the game is over or not. May be faster \
  than calculating the game status via `status(game)`.
- `Base.hash(game :: G)`: A hash function that is required for caching games \
  efficiently (see [`Model.CachingModel`](@ref)).
- `Base.isequal(a :: G, b :: G)`: Check if the game states `a` and `b` can be \
  considered to be functionally equivalent. Necessary for loop detection.
- `movecount(game :: G)`: Count the number of moves that have been applied to \
  `game`.
- `moverlabel(game :: G)`: String representation of the mover (-1 or 1).
- `movelabel(game :: G, action)`: String representation of `action` at `game`.
- `turnlabel(game :: G, actions)`: String representation of an action chain.
- `turncount(game :: G)`: Count the number of turns.
- `halfturncount(game :: G)`: Count the number of completed halfturns.
"""
abstract type AbstractGame end

@pack {<: AbstractGame} in MapFormat

Broadcast.broadcastable(game :: AbstractGame) = Ref(game)

function Base.copy(game :: AbstractGame)
  error("Base.copy not implemented for games of type $(typeof(game))") 
end

"""
    Base.hash(game)

Return a `UInt64` hash value of a game. Games that lead to the same data
representation (see `Game.array`) with the same active player should have the
same hash value.
"""
function Base.hash(game :: AbstractGame)
  error("Base.hash not implemented for games of type $(typeof(game))")
end

"""
    Base.isequal(game_a, game_b)

Check if two game states `game_a` and `game_b` are functionally equivalent.
This function is for example used to prevent loops in [`Player.decideturn`](@ref).
"""
function Base.isequal(:: G, :: G) where {G <: AbstractGame}
  error("Base.isequal not implemented for games of type $G")
end


"""
    status(game)

Status of the game `game`.

See also [`Status`](@ref).
"""
status(game :: AbstractGame) = error("not implemented")

"""
    mover(game)

Active player indicator for the game state `game`. The values 1 / -1 correspond
to the first / second player.
"""
mover(game :: AbstractGame) = error("not implemented")

"""
    legalactions(game)

Vector of actions that are legal at the game state `game`.
"""
legalactions(:: AbstractGame) :: Vector{ActionIndex} = error("not implemented")

"""
    isactionlegal(game, action)

Check if `action` is a legal action for `game`.
"""
function isactionlegal(game :: AbstractGame, action :: ActionIndex)
  action in legalactions(game)
end

"""
    move!(game, action)
    move!(game, actions)

Modify `game` by applying `action`, or all actions in the iterable `actions`.
Returns `game`.

See also [`move`](@ref) and [`randommove!`](@ref).
"""
function move!(:: AbstractGame, :: ActionIndex) :: AbstractGame
  error("not implemented")
end

function move!(game :: AbstractGame, actions)
  foreach(action -> move!(game, action), actions)
  game
end

"""
    move(game, action)
    move(game, actions)

Return the game state that results if `action` or `actions` are applied to
`game`. Does not alter `game`.

See also [`move!`](@ref).
"""
move(game :: AbstractGame, action) = move!(copy(game), action)

"""
    isover(game)

Returns `true` if the game `game` has finished and `false` if it is still
undecided.
"""
isover(game :: AbstractGame) = isover(status(game))

"""
    isaugmentable(game)
    isaugmentable(G)

Returns `true` if the game `game` or gametype `G` is augmentable and `false` \
otherwise.
"""
isaugmentable(g :: AbstractGame) = isaugmentable(typeof(g))
isaugmentable(:: Type{<: AbstractGame}) = false

"""
    policylength(G)
    policylength(game)

Maximal length of `legalactions(game)`, which is a static function of games of
type `G`.
"""
policylength(:: Type{AbstractGame}) :: Int = error("not implemented")
policylength(:: G) where {G <: AbstractGame} = policylength(G)

"""
    randomaction(game)

Returns a random legal action for `game`.

See also [`randommove!`](@ref).
"""
randomaction(game :: AbstractGame) = rand(legalactions(game))

"""
    randommove!(game)

Modify `game` by taking a single random action (if the game is not finished).

See also [`move!`](@ref), [`randommove`](@ref), [`randomaction`](@ref),
[`randomturn!`](@ref), and [`rollout!`](@ref).
"""
function randommove!(game :: AbstractGame)
  isover(game) ? game : move!(game, randomaction(game))
end

"""
    randommove!(game, steps)
    randommove!(game, steprange)

Modify `game` by taking several random actions. Returns `game`.

If an integer `steps` is passed, `steps` random actions are applied. If a range
`steprange` is passed, a random number `rand(steprange)` of actions are applied.
"""
function randommove!(game :: AbstractGame, steps :: Integer)
  for _ in 1:steps
    randommove!(game)
  end
  game
end

function randommove!(game :: AbstractGame, range)
  steps = rand(range)
  @assert steps isa Integer "Random range must be of type Integer"
  randommove!(game, steps)
end

"""
    randommove(game, args...)

Non-mutating version of [`randommove!`](@ref).
"""
randommove(game :: AbstractGame, args...) = randommove!(copy(game), args...)

"""
    randomturn!(game)

Modify `game` by taking random actions until the active player changes. Returns
`game`.

See also [`randomaction`](@ref), and [`rollout!`](@ref).
"""
function randomturn!(game :: AbstractGame)
  active = mover(game)
  while active == mover(game)
    randommove!(game)
  end
  game
end

"""
    randomturn(game)

Non-mutating version of [`randomturn!`](@ref).
"""
randommove(game :: AbstractGame) = randommove!(copy(game))

"""
    rollout!(game)

Play a match with random actions that starts at state `game`. Modifies `game`.

See [`rollout`](@ref) for a non-mutating version. See also
[`randommove!`](@ref) and [`randomturn!`](@ref).
"""
function rollout!(game :: AbstractGame, callback :: Function = _ -> nothing)
  while !isover(game)
    randommove!(game)
    callback(game)
  end
  game
end

"""
    rollout(game)

Return the final game of a random match started at initial state `game`.

See [`rollout!`](@ref) for a mutating version. See also
[`randommove!`](@ref) and [`randomturn!`](@ref).
"""
function rollout(game :: AbstractGame, callback :: Function = _ -> nothing)
  rollout!(copy(game), callback)
end

"""
    augment(game, label)

Returns a tuple `(games, labels)` with augmented versions of `game` and
`label`. Exploits the game's symmetry group, if any.
"""
function augment(game :: AbstractGame, label :: Array{Float32}) 
  @warn "Augmentation not implemented for type $(typeof(game))" maxlog = 1
  [game], [label]
end


"""
    visualize([io,] game)

Draw a unicode representation of `game`.
"""
function visualize(io :: IO, game :: AbstractGame)
  error("visualizing $(typeof(game)) not implemented.")
end

visualize(game :: AbstractGame) = visualize(stdout, game)

"""
    name(G)
    name(game)

String representation of the game name of `game` or game type `G`.
"""
name(G :: Type{<: AbstractGame}) = split(string(G), ".")[end]
name(game :: AbstractGame) = name(typeof(game))

"""
    instance(G)
    instance(game)

Obtain a game instance. Defaults to `G()` for a given game type `G`. Returns
`game` if `game` is of type `AbstractGame`.

See also [`randominstance`](@ref).
"""
instance(game :: AbstractGame) = game
instance(G :: Type{<: AbstractGame}) = G()

"""
    randominstance(G)

Return a randomized instance of game type `G`.

See also [`instance`](@ref).
"""
randominstance(G :: Type{<: AbstractGame}) = randommove!(G(), 1:5)

"""
    moverlabel(game)

Return a string representation of the current mover (-1 or 1).
"""
moverlabel(game) = string(mover(game))

"""
    movelabel(game, action)

Return a string representation of the action `action` at game state `game`.
"""
movelabel(game, action) = string(action)

"""
    turnlabel(game, actions)

Return a string representation of the action `action` at game state `game`.
"""
function turnlabel(game, actions)
  games = Game.movegames(game, actions)
  join(map(movelabel, games, actions), ">")
end


"""
    deriveaction(game_a, game_b)

Derive the action that moved `game_a` into `game_b`.
Returns `nothing` if no such action exists.
"""
function deriveaction(game_a, game_b)
  actions = legalactions(game_a)
  index = findfirst(actions) do action
    isequal(move(game_a, action), game_b)
  end
  if isnothing(index)
    nothing
  else
    actions[index]
  end
end

"""
    deriveactions(games)

Derive the list of actions that lead to the game state sequence `games`.
Throws an exception if the game states are inconsistent.
"""
function deriveactions(games :: Vector{<: AbstractGame})
  actions = ActionIndex[]
  @assert !isempty(games) "Cannot reconstruct actions for empty game sequence"
  current = games[1]

  for game in games[2:end]
    action = deriveaction(current, game)
    @assert !isnothing(action) """
    Subsequent game states not connected via legal action
    """
    push!(actions, action)
    current = game
  end

  actions
end

"""
    movegames(game, actions)

Apply `actions` to `game` consecutively and return a vector of resulting games,
a copy of `game` included.
"""
function movegames(game, actions)
  games = [copy(game)]
  for action in actions
    @assert isactionlegal(games[end], action) """
    Action $action is illegal.
    """
    game = move(games[end], action)
    push!(games, game)
  end
  games
end

"""
   movecount(game)

Return the number of actions that have been applied to `game`. Returns `0` if
not implemented for a specific game type.
"""
movecount(:: AbstractGame) = 0

"""
    turncount(game)

Return the number of turns of `game`. Here, one turn is a sequence of moves that
switch the mover from 1 to -1 and back to 1.

If not implemented for a specific game type, this function always returns 0.

See also [`movecount`](@ref) and [`halfturncount`](@ref).
"""
turncount(game :: AbstractGame) = 0

"""
    halfturncount(game)

Return the number of halfturns of `game`. Here, one halfturn is a sequence of
moves that switches the mover from 1 to -1 or -1 to 1.

If not implemented for a specific game type, this function always returns 0.

See also [`movecount`](@ref) and [`turncount`](@ref).
"""
halfturncount(game :: AbstractGame) = 0
