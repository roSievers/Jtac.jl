
"""
    planmatches(n, nplayers, active)

Returns `n` evenly distributed tuples `(i, j)`, where `1 <= i, j <= nplayers`,
such that at least one of `i` or `j` is in `active`.
"""
function planmatches(nmatches, nplayers, active)
  stop = false
  matches = []
  while !stop
    for i in 1:nplayers, j in 1:nplayers
      if i != j && (i in active || j in active)
        push!(matches, (i, j))
        if length(matches) >= nmatches
          stop = true
          break
        end
      end
    end
  end
  Random.shuffle(matches)
end

"""
    parallelforeach(f, items; ntasks, threads)

Run `f(item)` for each `item` in `items`. If `threads = true`, threading via
`ntasks` tasks is used. If `threads = false`, `ntasks` async tasks are used.
"""
function parallelforeach(f, items :: AbstractVector; ntasks, threads)
  ch = Channel{eltype(items)}(length(items))
  foreach(item -> put!(ch, item), items)
  close(ch)
  if threads
    Threads.foreach(f, ch; ntasks)
  else
    asyncmap(f, ch; ntasks)
  end
  nothing
end

"""
  compete(players, n [, active]; <keyword arguments>)

Create game results for a number of `n` games between players at positions `i`
and `j` in the collection `players` if at least one of `i` or `j` is in
`active`. 

The matches start with `game`, which is infered automatically from
`players` if possible. `callback()` is called after each of the `n` matches.

The return value is a result array of dimensions `(l, l, 3)`, where `l
= length(players)`. The entry at indices `[i,j,k]` stands for the number of
outcomes `k` (`1`: loss, `2`: draw, `3`: win) when player `i` played against
`j`. When both `i` and `j` are inactive, no games are played. The entries at
`[i, j, :]` are `0` in this case.

# Arguments
- `instance`: Initial game state provider. Defaults to `() -> Game.instance(G)` \
if the game type `G` can be inferred automatically.
- `callback`: Function called after each of the `n` matches.
- `threads`: Whether to use threading.
- `draw_after`: Number of moves after which the game is stopped and counted as \
a draw.
- `verbose`: Print current ranking after each match.
"""
function compete( players
                , n :: Int
                , active = 1:length(players)
                ; instance = gametype(players...)
                , callback = () -> nothing
                , verbose = false
                , threads = false
                , draw_after = typemax(Int) )

  if instance isa Type{<: AbstractGame}
    G = instance
    instance = () -> Game.instance(G)
  end

  # Using only a single BLAS thread was beneficial for performance in tests
  t = BLAS.get_num_threads()
  BLAS.set_num_threads(1)

  lk = ReentrantLock()
  matches = planmatches(n, length(players), active)
  results = zeros(Int, length(players), length(players), 3)

  count = 0
  report = (p1, p2, k) -> begin
    count += 1
    p1, p2 = name(p1), name(p2)
    verb = k == 3 ? ">" : (k == 1 ? "<" : "~")
    println("Match $count: $p1 $verb $p2")
    ranking = rank(players, results)
    println(string(ranking))
    println()
  end

  parallelforeach(matches; threads, ntasks = n) do (i, j)
    p1, p2 = players[i], players[j]
    k = pvp(p1, p2; instance, draw_after) + 2 # convert -1, 0, 1 to indices 1, 2, 3
    lock(lk) do
      results[i, j, k] += 1
      verbose && report(p1, p2, k)
      callback()
    end
  end

  BLAS.set_num_threads(t)
  results
end


"""
Structure that contains ranking information of a group of competing players.

Based on the results of a competition, elo values (and guesses for their
standard deviation via the observed Fisher information) as well as the start
advantage and a draw bandwidth is determined.
"""
struct Ranking
  players :: Vector{String}
  results :: Array{Int, 3}

  elo  :: Vector{Float64}
  sadv :: Float64
  draw :: Float64

  elostd  :: Vector{Float64}
  sadvstd :: Float64
  drawstd :: Float64
end

Pack.@untyped Ranking
Pack.@binarray Ranking [:results]

"""
    rank(players, results; steps = 100)
    rank(players, n [, active]; steps = 100, <keyword arguments>)

Get a ranking of `players` based on `results` from a previous call of the
function `compete`. Alternatively, results are generated on the fly via
`compete` for `n` games with all corresponding keyword arguments.  The number of
steps in the iterative maximum likelihood solver can be specified via `steps`.
"""
function rank(players :: Vector{String}, results :: Array{Int, 3}; steps = 100)
  elo, sadv, draw = mlestimate(results; steps = steps)
  elostd, sadvstd, drawstd = stdestimate(results, elo, sadv, draw)
  Ranking(players, results, elo, sadv, draw, elostd, sadvstd, drawstd)
end

function rank(players, results :: Array{Int, 3}; steps = 100)
  rank(name.(players), results, steps = steps)
end

function rank(players, args...; steps = 100, kwargs...)
  results = compete(players, args...; kwargs...)
  rank(players, results; steps = steps)
end

function Base.show(io :: IO, r :: Ranking)
  p = length(r.players)
  n = sum(r.results)
  print(io, "Ranking($p players, $n games)")
end

function Base.show(io :: IO, :: MIME"text/plain", r :: Ranking)
  n = sum(r.results)
  p = length(r.players)
  println(io, "Ranking with $p players and $n games:")
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

  players, elo, elostd = rk.players[perm], rk.elo[perm], rk.elostd[perm]
  nv = maximum(ndigits, elo)
  ns = maximum(ndigits, elostd)

  rows = map(1:length(perm), players, elo, elostd) do i, p, v, s
    Printf.@sprintf "%2d. %s%.0f ± %s%.0f %15s" i sp(v, nv) v sp(s, ns) s p
  end

  if matrix

    res = rk.results
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
  println(string(rk, matrix))
end

visualize(rk :: Ranking, args...) = visualize(stdout, rk, args...)

