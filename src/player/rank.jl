
"""
Structure that captures the results of matches between different players. Returned by [`compete`](@ref).

See also [`Ranking`](@ref).
"""
struct Results
  players :: Vector{String}
  outcomes :: Array{Int, 3}
end

@pack Results in MapFormat

function Results(players :: Vector{<: AbstractPlayer}, outcomes)
  Results(map(name, players), outcomes)
end

function Base.merge(rs :: Results ...)
  players = mapreduce(r -> r.players, vcat, rs)
  unique!(players)
  m = length(players)
  outcomes = zeros(Int, m, m, 3)
  for r in rs
    rn = length(r.players)
    idx = map(r.players) do player
      findfirst(isequal(player), players)
    end
    for i in 1:rn, j in 1:rn, k in 1:3
      outcomes[idx[i], idx[j], k] += r.outcomes[i, j, k]
    end
  end
  Results(players, outcomes)
end

"""
    planmatches(n, players, active)

Returns `n` (roughly evenly distributed) tuples `(p1, p2)`, where `p1` and `p2`
are in `players`, such that at least one of `p1` or `p2` is in `active`.
"""
function planmatches(n, players, active)
  matches = []
  while true
    for p1 in players, p2 in players
      if p1 != p2 && (p1 in active || p2 in active)
        push!(matches, (p1, p2))
        if length(matches) >= n
          return Random.shuffle(matches)
        end
      end
    end
  end
end


"""
  compete(players, n; active, kwargs...)

Simulate game results for a number of `n` matches between players at positions
`i` and `j` in the collection `players` if at least one of `i` or `j` is in
`active`.

The return value is of type [`Results`](@ref).

# Arguments
- `active`: Iterable of active players. Defaults to `players`.
- `instance`: Initial game state provider. Defaults to `() -> Game.instance(G)` \
if the game type `G` can be inferred automatically.
- `callback`: Function called after each of the `n` matches.
- `threads`: Whether to use threading.
- `draw_after`: Number of moves after which the game is stopped and counted as \
- `anneal`: Function applied to the move counter returning a temperature for \
  the next move.
a draw.
- `progress = true`: Whether to print progress information. Only meaningful if \
  `verbose = false`.
- `verbose = false`: Whether to print the current ranking after each match.
"""
function compete( players
                , n :: Integer
                ; active = players
                , instance = gametype(players...)
                , callback = () -> nothing
                , verbose = false
                , progress = true
                , threads = false
                , anneal = _ -> 1f0
                , draw_after = typemax(Int) )

  if instance isa Type{<: AbstractGame}
    G = instance
    instance = () -> Game.instance(G)
  end

  # Using only a single BLAS thread was beneficial for performance in tests
  t = BLAS.get_num_threads()
  BLAS.set_num_threads(1)

  lk = ReentrantLock()
  matches = planmatches(n, players, active)
  m = length(players)
  outcomes = zeros(Int, m, m, 3)

  if progress && !verbose
    step, finish = Util.stepper("# competing...", length(matches))
  end

  count = 0
  report = (p1, p2, k) -> begin
    count += 1
    p1, p2 = name(p1), name(p2)
    verb = k == 3 ? ">" : (k == 1 ? "<" : "~")
    println("Match $count of $n: $p1 $verb $p2")
    ranking = rank(Results(players, outcomes))
    println(string(ranking))
    println()
  end

  Util.pforeach(matches; threads, ntasks = n) do (p1, p2)
    status = pvp(p1, p2; instance, draw_after, anneal)
    k = Int(status) + 2 # convert -1, 0, 1 to indices 1, 2, 3
    i = findfirst(isequal(p1), players)
    j = findfirst(isequal(p2), players)
    lock(lk) do
      outcomes[i, j, k] += 1
      if progress && !verbose
        step()
      end
      if verbose
        report(p1, p2, k)
      end
      callback()
    end
  end

  if progress && !verbose
    finish()
  end

  BLAS.set_num_threads(t)
  Results(players, outcomes)
end


"""
Structure that contains ranking information of a group of competing players.

Based on the results of a competition, elo values (and guesses for their
standard deviation via the observed Fisher information) as well as the start
advantage and a draw bandwidth is determined.
"""
struct Ranking
  results :: Results

  elo  :: Vector{Float64}
  sadv :: Float64
  draw :: Float64

  elostd  :: Vector{Float64}
  sadvstd :: Float64
  drawstd :: Float64
end

@pack Ranking in MapFormat

"""
    rank(results; steps = 100)
    rank(players, n [, results...]; steps = 100, <keyword arguments>)

Get a ranking of `players` based on `results` from a previous call of the
function [`compete`](@ref).

Alternatively, results are generated on the fly (and merged with optional
`results`) via `compete` for `n` games with all corresponding keyword arguments.
The number of steps in the iterative maximum likelihood solver can be specified
via `steps`.
"""
function rank(r :: Results; steps = 100)
  elo, sadv, draw = mlestimate(r.outcomes; steps = steps)
  elostd, sadvstd, drawstd = stdestimate(r.outcomes, elo, sadv, draw)
  Ranking(r, elo, sadv, draw, elostd, sadvstd, drawstd)
end

rank(rs :: Results ...; kwargs...) = rank(Base.merge(rs...); kwargs...)

function rank(players, n :: Integer; steps = 100, kwargs...)
  results = compete(players, n; kwargs...)
  rank(results; steps)
end

function rank(players, n :: Integer, rs :: Results ...; steps = 100, kwargs...)
  results = compete(players, n; kwargs...)
  results = merge(results, rs...)
  rank(results; steps)
end

"""
    rankmodels(models, n; powers, draw_bias, opponents, kwargs...)

Produce a ranking that compares a list of `models` by conducting `n` matches.

For each model (with index `i`) and `power` value in `powers`, the following players are produced:
* `IntuitionPlayer(model; name = "int-\$i")` for `power = 0`,
* `MCTSPlayer(model; power, draw_bias, name = mcts\$power-\$i)` for `power > 0`.

Additional players that should be included in the ranking can be specified via
`opponents`. See `[rank](@ref)` and `[compete](@ref)` for the remaining keyword arguments.

"""
function rankmodels( models
                   , n :: Int = 100
                   ; powers = [0]
                   , draw_bias = 0.0
                   , opponents = []
                   , kwargs... )
  powers = powers |> unique |> sort
  players = mapreduce(vcat, enumerate(models)) do (i, model)
    map(powers) do power
      if power == 0
        name = "int-$i"
        IntuitionPlayer(model; name)
      else
        name = "mcts$power-$i"
        MCTSPlayer(model; power, draw_bias, name)
      end
    end
  end
  players = [players..., opponents...]
  rank(players, n; kwargs...)
end

function Base.show(io :: IO, r :: Ranking)
  m = length(r.results.players)
  n = sum(r.results.outcomes)
  print(io, "Ranking($m players, $n games)")
end

function Base.show(io :: IO, :: MIME"text/plain", r :: Ranking)
  m = length(r.results.players)
  n = sum(r.results.outcomes)
  println(io, "Ranking with $m players and $n games:")
  print(io, string(r, true))
end


# -------- Visualize Rankings ------------------------------------------------ #

function Base.string(rk :: Ranking, matrix = false)

  # Formatting values dynamically (which does *not* work with @printf...)
  ndigits = x -> begin
    if !isfinite(x)
      3
    else
      (round(Int, x) |> digits |> length) + (x < 0)
    end
  end
  sp = (x, n) -> repeat(" ", max(n - ndigits(x), 0))

  # The player with highest elo will come first
  perm = sortperm(rk.elo) |> reverse

  players = rk.results.players[perm]
  elo = rk.elo[perm]
  elostd = rk.elostd[perm]
  nv = maximum(ndigits, elo)
  ns = maximum(ndigits, elostd)

  rows = map(1:length(perm), players, elo, elostd) do i, p, v, s
    Printf.@sprintf "%2d. %s%.0f ± %s%.0f %15s" i sp(v, nv) v sp(s, ns) s p
  end

  if matrix

    res = rk.results.outcomes
    mat = res[perm, perm, 3] - res[perm, perm, 1]
    nm = maximum(ndigits, mat)

    for (i, row) in enumerate(rows)
      vals = map(mat[i,:], 1:length(perm)) do x, j
        if i == j
          repeat(" ", nm) * "•"
        else
          Printf.@sprintf(" %s%.0f", sp(x, nm), x)
        end
      end
      rows[i] = row * "  " * join(vals)
    end

  end

  sadv = Printf.@sprintf "start advantage %.0f ± %.0f" rk.sadv rk.sadvstd
  draw = Printf.@sprintf "draw bandwidth  %.0f ± %.0f" rk.draw rk.drawstd

  join([rows; ""; sadv; draw], "\n")
end

function visualize(io :: IO, rk :: Ranking, matrix = false)
  println(io, string(rk, matrix))
end

visualize(rk :: Ranking, args...) = visualize(stdout, rk, args...)

