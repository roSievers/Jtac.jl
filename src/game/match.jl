
"""
Structure for conveniently storing and accessing matches.

A match is a sequence of game states that result from one another by applying
legal actions.
"""
struct Match{G <: AbstractGame}
  games :: Vector{G}
  actions :: Vector{ActionIndex}
  turns :: Vector{Tuple{Int, Int}}
end

function Match(game :: AbstractGame)
  Match([copy(game)], ActionIndex[], Tuple{Int,Int}[])
end

function Match(game :: AbstractGame, actions)
  match = Match(game)
  for action in actions
    move!(match, action)
  end
  match
end

function Match(games :: Vector{<: AbstractGame})
  actions = Game.deriveactions(games)
  Match(games[1], actions)
end

function move!(match :: Match, action :: ActionIndex)
  prev = match.games[end]
  game = move(prev, action)
  push!(match.games, game)
  push!(match.actions, action)

  l = length(match.actions)
  if isempty(match.turns)
    push!(match.turns, (1, 1))
  elseif mover(game) == mover(prev) != mover(match.games[end-2])
    push!(match.turns, (l, l))
  else
    turn = match.turns[end]
    match.turns[end] = (turn[1], turn[2]+1) 
  end

  nothing
end


status(match :: Match) = status(match.games[end])
isover(match :: Match) = isover(match.games[end])
mover(match :: Match) = mover(match.games[end])
legalactions(match :: Match) = legalactions(match.games[end])

moverlabel(match :: Match) = moverlabel(match.games[end])
movelabel(match :: Match) = movelabel(match.games[end])
turnlabel(match :: Match) = turnlabel(match.games[end])

randomaction(match :: Match) = randomaction(match.games[end])
randommove!(match :: Match) = move!(match, randomaction(match))

function randommatch(game :: AbstractGame)
  match = Match(game)
  while !isover(match)
    randommove!(match)
  end
  match
end

movecount(match :: Match) = length(match.actions)
turncount(match :: Match) = ceil(Int, length(match.turns) / 2)
halfturncount(match :: Match) = length(match.turns)

function getaction(match, index :: Integer)
  @assert 1 <= index <= length(match.actions)
  match.actions[index]
end

function getmove(match, index :: Integer)
  @assert 1 <= index <= length(match.actions)
  match.games[index], match.actions[index]
end

function getturn(match, index :: Integer, mover :: Integer)
  @assert mover in (-1, 1) "Expected mover -1 or 1"
  idx = mover == 1 ? 2index - 1 : 2index
  gethalfturn(match, idx)
end

function gethalfturn(match, index :: Integer)
  @assert 1 <= index <= halfturncount(match) """
  Match does not have $index halfturns.
  """
  turn = match.turns[index]
  match.games[turn[1]], match.actions[turn[1]:turn[2]]
end

function Base.show(io :: IO, ::MIME"text/plain", m :: Match{G}) where {G <: AbstractGame}
  s = Game.status(m.games[end])
  print(io, "Match{$G}($s) with $(movecount(m)) moves and $(turncount(m)) turns")
end

