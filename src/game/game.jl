
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

Pack.@untyped AbstractGame

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

Modify `game` by applying `action`. Returns `game`.
"""
apply_action!(:: AbstractGame, :: ActionIndex) :: AbstractGame = error("unimplemented")

"""
    apply_actions!(game, actions)

Modify `game` by applying all actions in the iterable `actions`. Returns `game`.
"""
function apply_actions!(game :: AbstractGame, actions)
  for action in actions apply_action!(game, action) end
  game
end

"""
    is_over(game)

Returns whether a game is finished or not.
"""
is_over(game :: AbstractGame) = is_over(status(game))

"""
    is_augmentable(game)

Whether a game or gametype is augmentable or not.
"""
is_augmentable(g :: AbstractGame) = is_augmentable(typeof(g))
is_augmentable(:: Type{<: AbstractGame}) = false

# Data representation of the game as layered 2d image from the perspective of
# the active player (active player plays with 1, other with -1)
"""
    array(game)
    array(games)

Data representation of `game` as three-dimensional array. If a vector
of `games` is given, a four-dimensional array is returned. See also `Game.array!`
"""
array(:: AbstractGame) = error("unimplemented")

function array(games :: Vector{G}) where G <: AbstractGame
  @assert !isempty(games) "Cannot produce representation of empty game vector"
  buf = array_buffer(G, length(games))
  for i in 1:length(games)
    buf[:,:,:,i] = array(games[i])
  end
  buf
end

function array!(buf, games :: Vector{<: AbstractGame})
  @assert !isempty(games) "Cannot produce representation of empty game vector"
  @assert size(buf)[1:3] == size(games[1])
  @assert size(buf, 4) >= length(games)
  repr = array(games)
  repr = Model.adapt_atype(buf, repr)
  buf[:, :, :, 1:length(games)] .= repr
  nothing
end

"""
    array_buffer(G, batchsize)

Create an uninitialized `Float32` array that can hold the array representation
of up to `batchsize` games of type `G`.
"""
function array_buffer(G ::Type{<: AbstractGame}, batchsize)
  zeros(Float32, size(G)..., batchsize)
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
function random_playout!(game :: AbstractGame, callback :: Function = _ -> nothing )

  while !is_over(game)
    random_turn!(game)
    callback(game)
  end

  game

end

"""
    random_playout_count_moves(game)

Random playout that counts the moves that happened and returns this.
"""
function random_playout_count_moves(game :: AbstractGame)::Int64
  counter = 0
  random_playout!(copy(game), _ -> counter = counter + 1)
  counter
end

"""
    random_playout(game)

Random playout with initial state `game` without modifying it.
"""
function random_playout(game :: AbstractGame, callback :: Function = _ -> nothing )

  random_playout!(copy(game), callback)

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

"""
    name(G)
    name(game)

Pretty printable name of the game.
"""
name(G :: Type{<: AbstractGame}) = split(string(G), ".")[end]
name(game :: AbstractGame) = name(typeof(game))

"""
    instance(game)
    instance(G [, random])

Obtain a game instance. Acts as identity on arguments `game` of type
`AbstractGame`. If a game type `G` is provided, a new instance is created.

The optional keyword argument `random` can be used to stochastically randomize
the created instance. For example, if `random = 0.7`, then with probability
`0.3` the default instance `G()` and with probability `0.7` a randomized
instance `random_instance(G)` is returned.
"""
instance(game :: AbstractGame) = game

function instance(G :: Type{<: AbstractGame}; randomize = false)
   @assert isconcretetype(G) "Cannot instantiate abstract game $G"
   if rand() < randomize
     random_instance(G)
   else
     G()
   end
end

"""
    random_instance(G)

Return a randomized instance of game `G`.
"""
random_instance(G :: Type{<: AbstractGame}) = random_turns!(G(), 1:5)

"""
    branch(game; prob, steps = 1)
    branch(; prob, steps = 1)

Randomly branch `game` with probability `prob` by applying `steps` random
actions. If the game is not branched `nothing` is returned. Otherwise,
the branched game is returned.

The second method returns a branching function `game -> branch(game, ...)`.
"""
function branch(game; prob, steps = 1)
  if rand() < prob
    random_turns!(copy(game), steps)
  else
    nothing
  end
end

branch(; prob, steps = 1) = game -> branch(game; prob, steps)

"""
    hash(game)

`UInt64` hash value of a game. Games that lead to the same data representation
(see `Game.array`) with the same current player should have same hash value.
"""
hash(game :: AbstractGame) = error("hashing for game $(typeof(game)) not implemented")

"""
   moves(game)

Return the number of actions that have been applied to `game`. Returns `0` if
not implemented for a specific game type.
"""
moves(:: AbstractGame) = 0
