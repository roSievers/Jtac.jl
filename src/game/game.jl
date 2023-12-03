
"""
Enum that represents the status of a game. Can be `loss`, `draw`, `win`, or
`undecided`.
"""
@enum Status loss=-1 draw=0 win=1 undecided=42

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
- `activeplayer(game :: G)`: Return the active player indicator (-1 or 1).
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
- `hash(game :: G)`: A hash function that is required for caching games \
  efficiently (see [`Model.CachingModel`](@ref)).
- `moves(game :: G)`: Count the number of moves that have been applied to \
  `game`.
"""
abstract type AbstractGame end

@pack {<: AbstractGame} in MapFormat

Broadcast.broadcastable(game :: AbstractGame) = Ref(game)

Base.copy(:: AbstractGame) = error("not implemented") 
Base.size(:: Type{<: AbstractGame}) = error("not implemented")
Base.size(:: G) where {G <: AbstractGame} = size(G)

"""
    status(game)

Status of the game `game`.

See also [`Status`](@ref).
"""
status(game :: AbstractGame) = error("not implemented")

"""
    activeplayer(game)

Active player indicator for the game state `game`. The values 1 / -1 correspond
to the first / second player.
"""
activeplayer(game :: AbstractGame) = error("not implemented")

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

See also [`randommove!`](@ref).
"""
function move!(:: AbstractGame, :: ActionIndex) :: AbstractGame
  error("not implemented")
end

function move!(game :: AbstractGame, actions)
  foreach(action -> move!(game, action), actions)
  game
end

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
    array(game)
    array(games)

Data representation of `game` as three-dimensional `Array{Float32}`. If a vector
`games` is passed, a four-dimensional array is returned.

See also [`array!`](@ref) and [`arraybuffer`](@ref).
"""
array(:: AbstractGame) = error("not implemented")

function array(games :: Vector{G}) where G <: AbstractGame
  @assert !isempty(games) "Cannot produce representation of empty game vector"
  buf = arraybuffer(G, length(games))
  for i in 1:length(games)
    buf[:,:,:,i] .= array(games[i])
  end
  buf
end

"""
    array!(buffer, games)

Fill the array `buffer` with the array representation of `games`. Amounts to
`buffer[:,:,:,1:length(games)] .= array(games)`, but can be implemented more
efficiently.

The buffer array must satisfy `size(buffer)[1:3] .== size(games[1])` as well
as `size(buffer, 4) >= length(games)`. To create suitably sized buffers, see
[`arraybuffer`](@ref).
"""
function array!(buf, games :: Vector{<: AbstractGame})
  @assert !isempty(games) "Cannot produce representation of empty game vector"
  @assert size(buf)[1:3] == size(games[1])
  @assert size(buf, 4) >= length(games)
  repr = array(games)
  repr = convert(typeof(buf), repr)
  buf[:, :, :, 1:length(games)] .= repr
  nothing
end

"""
    arraybuffer(G, batchsize)

Create an uninitialized `Float32` array that can hold the array representation
of up to `batchsize` games of type `G`.
"""
function arraybuffer(G :: Type{<: AbstractGame}, batchsize)
  zeros(Float32, size(G)..., batchsize)
end

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

See also [`move!`](@ref), [`randomaction`](@ref), [`randomturn!`](@ref), and
[`randommatch!`](@ref).
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
    randomturn!(game)

Modify `game` by taking random actions until the active player changes. Returns
`game`.

See also [`randomaction`](@ref), and [`randommatch!`](@ref).
"""
function randomturn!(game :: AbstractGame)
  active = activeplayer(game)
  while active == activeplayer(game)
    randommove!(game)
  end
  game
end

"""
    randommatch!(game)

Play a match with random actions that starts at state `game`. Modifies `game`.

See [`randommatch`](@ref) for a non-mutating version. See also
[`randommove!`](@ref) and [`randomturn!`](@ref).
"""
function randommatch!(game :: AbstractGame, callback :: Function = _ -> nothing)
  while !isover(game)
    randommove!(game)
    callback(game)
  end
  game
end

"""
    randommatch(game)

Return the final game of a random match started at initial state `game`.

See [`randommatch!`](@ref) for a mutating version. See also
[`randommove!`](@ref) and [`randomturn!`](@ref).
"""
function randommatch(game :: AbstractGame, callback :: Function = _ -> nothing)
  randommatch!(copy(game), callback)
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
    branch(game; steps = 1)

Randomly branch `game` by applying `steps` random actions. Returns a branched
copy of `game`.
"""
function branch(game; steps = 1)
  randommove!(copy(game), steps)
end

"""
    branch(; prob, steps = 1)

Return a branching function that branches `game` via `steps` random steps with
a probability of `prob`. If the game is not branched, the branching function
returns `nothing`. Otherwise, the branched game is returned.
"""
function branch(; prob, steps = 1)
  game -> begin
    if rand() < prob
      randommove!(copy(game), steps)
    else
      nothing
    end
  end
end

"""
    hash(game)

Return a `UInt64` hash value of a game. Games that lead to the same data
representation (see `Game.array`) with the same active player should have the
same hash value.
"""
function hash(game :: AbstractGame)
  error("hashing for game $(typeof(game)) not implemented")
end

"""
   moves(game)

Return the number of actions that have been applied to `game`. Returns `0` if
not implemented for a specific game type.
"""
moves(:: AbstractGame) = 0
