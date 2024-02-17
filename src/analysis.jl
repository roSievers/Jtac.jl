
"""
Structure that contains a player-analysis of an individual game state.

The game state itself as well as all game states reachable via legal actions are
evaluated by the player.
"""
struct GameAnalysis{G <: AbstractGame}
  game :: G # game state subject to the analysis
  mover :: Int # active mover. 1 or -1
  actions :: Vector{ActionIndex} # list of legal actions
  policy :: Vector{Float32} # policy produced by the player applied on `game`
  value :: Float32 # value of `game`, stored from perspective of mover 1
  values :: Vector{Float32} # values of game states reachable via legal actions
end

"""
    analyzegame(player, game)

Let `player` analyze the game state `game` and return an [`GameAnalysis`](@ref)
object.
"""
function analyzegame(player, game)
  actions = Game.legalactions(game)
  value, policy = Model.apply(player, game)
  value *= Game.mover(game) # perspective of player 1

  values = asyncmap(actions) do action
    game_alt = Game.move(game, action)
    if Game.isover(game_alt)
      value_alt = Int(Game.status(game_alt))
    else
      value_alt, _ = Model.apply(player, game_alt)
    end
    Float32(value_alt) * Game.mover(game_alt)
  end

  GameAnalysis(
    copy(game),
    mover(game),
    actions,
    policy,
    value,
    values,
  )
end

function Base.show(io :: IO, ::MIME"text/plain", a :: GameAnalysis)
  worst, best = a.mover .* extrema(a.mover .* a.values)
  @printf io "Mover %s acts:\n" moverlabel(a.game)
  @printf io " value    %5.2f\n" a.value
  @printf io " Δ-best   %+.2f\n" (best - a.value)
  @printf io " Δ-worst  %+.2f\n" (worst - a.value)
  @printf io " actions   %d\n" length(a.actions)
  @printf io " kendall  %5.2f\n" kendall(a)
  @printf io " proposal  "

  for (action, impact) in goodmoves(a)
    print(io, movelabel(a.game, action))
    str = @sprintf " (%+.2f) " impact
    printstyled(io, str, color = 247)
  end
end


"""
Structure that contains a player-analysis of a single move.

Additionally to a [`GameAnalysis`](@ref), it also stores an action.
"""
struct MoveAnalysis{G <: AbstractGame}
  game_analysis :: GameAnalysis{G}
  action :: ActionIndex
  value_after :: Float32
end

"""
    MoveAnalysis(game_analysis, action)

Construct a move analysis for action `action` based on `game_analysis`.
"""
function MoveAnalysis(a :: GameAnalysis, action :: ActionIndex)
  @assert Game.isactionlegal(a.game, action) """
  Action index provided for move analysis is not legal.
  """
  index = findfirst(isequal(action), a.actions)
  MoveAnalysis(a, action, a.values[index])
end

"""
    analyzemove(player, game, action)

Let `player` analyze the action `action` on game `game`. Returns a
[`MoveAnalysis`](@ref) object.
"""
function analyzemove(player, game, action)
  @assert Game.isactionlegal(game, action) """
  Action index provided for move analysis is not legal.
  """
  a = analyzegame(player, game)
  index = findfirst(isequal(action), a.actions)
  value_after = a.values[index]
  MoveAnalysis(a, action, value_after)
end

"""
    analyzemoves(player, games, actions)
    analyzemoves(player, game, actions)
    analyzemoves(player, games)

Batch-evaluation of `analyzemove(player, game, action)` for each pair
`(game, action)` in `zip(games, actions)`.

If a single game state `game` is provided, the actions are understood to be
consecutively applied to `game`. If only a vector of games is given, actions are
derived via [`Game.deriveactions`](@ref).
"""
function analyzemoves( player
                     , games :: Vector{<: AbstractGame}
                     , actions :: Vector{ActionIndex} )
  asyncmap(games, actions) do game, action
    analyzemove(player, game, action)
  end
end

function analyzemoves(player, games :: Vector{<: AbstractGame})
  actions = Game.deriveactions(games)
  analyzemoves(player, games, actions)
end

function analyzemoves(player, game :: AbstractGame, actions)
  games = Game.movegames(game, actions)
  analyzemoves(player, games, actions)
end

function Base.show(io :: IO, ::MIME"text/plain", a :: MoveAnalysis)
  game = a.game_analysis.game
  mover = a.game_analysis.mover
  moverlabel = Game.moverlabel(game)
  movelabel = Game.movelabel(game, a.action)
  worst, best = mover .* extrema(mover .* a.game_analysis.values)

  @printf io "Mover %s picks action %s\n" moverlabel movelabel
  @printf io " value    %5.2f > %.2f\n" a.game_analysis.value a.value_after
  @printf io " Δ        %+.2f " (a.value_after - a.game_analysis.value)
  printcomment(io, a); println()
  @printf io " Δ-best   %+.2f\n" (best - a.game_analysis.value)
  @printf io " Δ-worst  %+.2f\n" (worst - a.game_analysis.value)
  @printf io " rank      %d of %d\n" moverank(a)...
  @printf io " surprise  %.2f\n" surprise(a)
  @printf io " kendall   %.2f\n" kendall(a)
  @printf io " proposal  "
  for (action, impact) in goodmoves(a)
    print(io, Game.movelabel(game, action))
    str = @sprintf " (%+.2f) " impact
    printstyled(io, str, color = 247)
  end
end

"""
Structure that contains a player-analysis of a single turn.
"""
struct TurnAnalysis{G <: AbstractGame}
  game :: G # initial game state subject to the turn analysis
  mover :: Int # active mover
  actions :: Vector{ActionIndex} # turn as a sequence of actions

  move_analyses :: Vector{MoveAnalysis{G}} # analyses of the moves along turn
  alternatives :: Vector{TurnAnalysis{G}} # comparison object for the turn
end

"""
    analyzebestturn(player, game)

Analyze `game` and return an analysis of the best turn according to `player`.
"""
function analyzebestturn(player, game :: G) where {G <: AbstractGame}
  mover = Game.mover(game)
  a = analyzegame(player, game)

  state = copy(game)
  move_analyses = MoveAnalysis{G}[]
  actions = ActionIndex[]

  while !Game.isover(state) && mover == Game.mover(state)
    a = analyzegame(player, state)
    action, _ = bestmove(a)
    push!(actions, action)
    push!(move_analyses, MoveAnalysis(a, action))
    state = Game.move(state, action)
  end

  TurnAnalysis(
    game,
    mover,
    actions,
    move_analyses,
    TurnAnalysis{G}[],
  )
end

"""
    analyzeturn(player, game, actions)
    analyzeturn(player, games)

Let `player` analyze the turn determined by moving `game` along `actions`.

If a vector of games is passed, actions are derived via
[`Game.deriveactions`](@ref)
"""
function analyzeturn(player, game :: G, actions) where {G <: AbstractGame}
  @assert length(actions) >= 1 """
  Turn must consist of at least one action.
  """
  mover = Game.mover(game)
  games = Game.movegames(game, actions)
  @assert all(isequal(mover), Game.mover.(games[1:end-1])) """
  Mover must not alternate during the turn.
  """
  @assert Game.isover(games[end]) || Game.mover(games[end]) != mover """
  Mover must alternate at the end of the turn.
  """

  # conduct analysis of actual turn and best term in async
  moves = alternative = nothing

  @sync begin
    @async moves = analyzemoves(player, game, actions)
    @async alternative = analyzebestturn(player, game)
  end

  TurnAnalysis(game, mover, actions, moves, [alternative])
end

function analyzeturn(player, games :: Vector{<: AbstractGame})
  actions = Game.deriveactions(games)
  analyzeturn(player, game, actions)
end

function Base.show(io :: IO, ::MIME"text/plain", a :: TurnAnalysis)
  game = a.game
  mover = a.mover
  moverlabel = Game.moverlabel(game)
  turnlabel = Game.turnlabel(game, a.actions)
  value = a.move_analyses[1].game_analysis.value
  value_after = a.move_analyses[end].value_after

  @printf io "Mover %s takes turn %s:\n" moverlabel turnlabel
  @printf io " value    %5.2f > %.2f\n" value value_after
  @printf io " Δ        %+.2f " impact(a)
  printcomment(io, a); println()
  @printf io " Δ-alt    %+.2f\n" bestturn(a)[2]
  # @printf io " rank      %d of %d\n" moverank(ma)...
  @printf io " surprise  %.2f\n" surprise(a)
  @printf io " kendall   %.2f\n" kendall(a)
  @printf io " proposal  "
  for (actions, impact) in goodturns(a)
    print(io, Game.turnlabel(game, actions))
    str = @sprintf " (%+.2f) " impact
    printstyled(io, str, color = 247)
  end
end

#
# Metrics
#

"""
    bestmove(player, game)
    bestmove(game_analysis)  
    bestmove(move_analysis)  
    bestmove(turn_analysis)  

Return the single best action index for the current game state together with
its impact.
"""
function bestmove(game_analysis :: GameAnalysis)
  index = findmax(game_analysis.mover .* game_analysis.values)[2]
  action = game_analysis.actions[index]
  action, impact(game_analysis, action)
end

bestmove(player, game :: AbstractGame) = bestmove(analyzegame(player, game))
bestmove(a:: MoveAnalysis) = bestmove(a.game_analysis)
bestmove(a :: TurnAnalysis) = bestmove(a.move_analyses[1].game_analysis)

"""
    goodmoves(player, game; at_most = 3)
    goodmoves(game_analysis; at_most = 3)
    goodmoves(move_analysis; at_most = 3)
    goodmoves(turn_analysis; at_most = 3)

Return a list of action indices for moves that the player considers to be
suggestable.
"""
goodmoves(a) = [bestmove(a)] # TODO

"""
    bestturn(player, game)
    bestturn(turn_analysis)

Return a chain of actions that represent the best turn for the current game
state, according to the analysis.
"""
function bestturn(a :: TurnAnalysis)
  turn_options = [a, a.alternatives...]
  impact_best, index = findmax(turn_options) do ta
    impact(ta, relative = true)
  end
  (turn_options[index].actions, a.mover * impact_best)
end

bestturn(player, game) = analyzebestturn(player, game).actions

"""
    goodturns(player, game; at_most = 3)
    goodmoves(turn_analysis; at_most = 3)

Return a list of action lists for turns that the player considers to be
suggestable.
"""
goodturns(a) = [bestturn(a)] # TODO

"""
    impact(player, game, actions; relative = false)
    impact(game_analysis, action; relative = false)
    impact(move_analysis; relative = false)
    impact(turn_analysis; relative = false)

Value change due to the move(s) or turn. Reported from the perspective of mover
1 unless `relative = true`.
"""
function impact(player, game :: AbstractGame, actions; relative = false)
  games = [game, Game.move(game, actions)]
  values = asyncmap(games) do game
    Game.mover(game) * Model.apply(player, game).value
  end
  delta = values[2] - values[1]
  relative ? mover * delta : delta
end

function impact(a :: GameAnalysis, action; relative = false)
  impact(MoveAnalysis(a, action); relative)
end

function impact(a :: MoveAnalysis; relative = false)
  mover = Game.mover(a.game_analysis.game)
  delta = a.value_after - a.game_analysis.value
  relative ? mover * delta : delta
end

function impact(a :: TurnAnalysis; relative = false)
  mover = Game.mover(a.game)
  ma = a.move_analyses
  delta = ma[end].value_after - ma[1].game_analysis.value
  relative ? mover * delta : delta
end


"""
    kendall(game_analysis)
    kendall(move_analysis)
    kendall(turn_analysis)

Get the kendall concordance metric of an analysis object. Quantifies how
consistent the player policy is to its value judgement.

Note that this function evaluates the *player* conducting an analysis, *not* the
quality of the move or turn that is analyzed. A score of 1 means perfect
consistency of the player, while -1 means perfect anti-consistency.

In case of a turn analysis, the minimal concordance along the chain of actions
is returned.
"""
function kendall(a :: GameAnalysis)
  if length(a.actions) == 1
    1f0
  else
    x = a.policy[a.actions]
    y = a.mover .* a.values
    xy = kendalldot(x, y, y)
    xx = kendalldot(x, x, y)
    yy = kendalldot(y, y, y)
    xy / sqrt(xx) / sqrt(yy)
  end
end

kendall(a :: MoveAnalysis) = kendall(a.game_analysis)
kendall(a :: TurnAnalysis) = minimum(kendall, a.move_analyses)

function kendalldot(x :: Vector, y :: Vector, w :: Vector)
  @assert length(x) == length(y)
  @assert length(x) > 1
  s = 0
  for i in 1:length(x), j in i+1:length(x)
    s += sign(x[i] - x[j]) * sign(y[i] - y[j]) * abs(w[i] - w[j])
  end
  s
end

function surprise(a :: MoveAnalysis)
  ga = a.game_analysis
  1 - (ga.policy[a.action] / maximum(ga.policy))
end

function surprise(a :: TurnAnalysis)
  maximum(surprise, a.move_analyses)
end

function moverank(ma :: MoveAnalysis)
  active = mover(ma.game_analysis.game)
  value = active * ma.value_after
  values = sort(active .* ma.game_analysis.values, rev = true)
  nactions = length(ma.game_analysis.values)
  findfirst(isequal(value), values), nactions
end

function printcomment(io :: IO, impact, impact_alt)
  # The marker ! means that the player is surprised by the best turn/move
  # Normally we would expect that the white player can only reduce the value
  # of the game with their moves, and the black player can only increase it.
  # That is how game theory in these trees works. If we violate this principle,
  # the player is surprised. This points at a weakness.
  if impact_alt > 0.8 printstyled(io, "!!! ")
  elseif impact_alt > 0.4 printstyled(io, "!! ")
  elseif impact_alt > 0.1 printstyled(io, "! ")
  end

  # The marker ? means that the player rates the played turn/move as a bad move.
  # yellow: the move improves the value of the game, but not as much as the best
  # red: the move reduces the value of the game more than the best move
  color = impact > 0 ? (:yellow) : (:red)
  delta = impact_alt - impact

  comment = sym -> printstyled(io, sym * @sprintf(" %.2f", delta); color)

  if delta > 0.8 comment("???")
  elseif delta > 0.4 comment("??")
  elseif delta > 0.1 comment("?")
  end
end

function printcomment(io :: IO, a :: MoveAnalysis)
  mover = a.game_analysis.mover
  impact = mover * (a.value_after - a.game_analysis.value)
  impact_alt = maximum(mover .* (a.game_analysis.values .- a.game_analysis.value))
  printcomment(io, impact, impact_alt)
end

function printcomment(io :: IO, a :: TurnAnalysis)
  mover = a.mover
  impact = mover * Analysis.impact(a)
  impact_alt = mover * bestturn(a)[2]
  printcomment(io, impact, impact_alt)
end

printcomment(args...) = printcomment(stdout, args...)

"""
    analyzemove(player, match, index)

Let `player` analyze the move `index` in `match`.
"""
function analyzemove(player, m :: Match, index :: Integer)
  @assert 1 <= index <= length(m.actions) """
  Move with index $index does not exist in this match.
  """
  analyzemove(player, m.games[index], m.actions[index])
end

"""
    analyzeturn(player, match, index, mover)

Let `player` analyze the turn `index` of mover `mover` in `match`.
"""
function analyzeturn(player, m :: Match, index, mover)
  game, actions = Game.getturn(m, index, mover)
  analyzeturn(player, game, actions)
end

function analyzehalfturn(player, m :: Match, index)
  game, actions = Game.gethalfturn(m, index)
  analyzeturn(player, game, actions)
end

struct MatchAnalysis{G <: AbstractGame}
  match :: Match{G}
  turn_analyses :: Vector{TurnAnalysis{G}}
end

function analyze(player, m :: Match)
  as = asyncmap(1:length(m.turns)) do index
    analyzehalfturn(player, m, index)
  end
  MatchAnalysis(m, as)
end

function reportturns(io :: IO, a :: MatchAnalysis{G}) where {G <: AbstractGame}
  turnlabels = map(a.turn_analyses) do ta
    Game.turnlabel(ta.game, ta.actions)
  end
  labellength = maximum(length, turnlabels)
  spaces = max(5, labellength) - 5
  str = join([
    " turn",
    " value",
    "     Δ",
    " Δ-alt",
    repeat(" ", spaces) * " moves",
    "",
  ], "  ")
  crayon = crayon"white bg:black"
  println(io, crayon, str, inv(crayon))
  for (index, ta) in enumerate(a.turn_analyses)
    turncount = div(index, 2) + index % 2
    reportturn(io, ta, turncount, labellength)
    if index < length(a.turn_analyses)
      println(io)
    end
  end
end

reportturns(a :: MatchAnalysis) = reportturns(stdout, a)

function reportturn(io :: IO, a :: TurnAnalysis, index, labellength)
  cr = a.mover == 1 ? crayon"16 bg:253 bold" : crayon"white bg:black"
  str = join([
    @sprintf(" %4s", string(index)),
    @sprintf(" %5.2f", a.move_analyses[1].game_analysis.value),
    @sprintf(" %+.2f", impact(a)),
    @sprintf(" %+.2f", bestturn(a)[2]),
    @sprintf(" %*s", labellength, Game.turnlabel(a.game, a.actions)),
    "",
  ], "  ")

  print(io, cr, str, inv(cr), " ")
  printcomment(io, a)
end


function Base.show(io :: IO, :: MIME"text/plain", a :: MatchAnalysis)
  reportturns(io, a)
end

# TODO: reportmoves function!
