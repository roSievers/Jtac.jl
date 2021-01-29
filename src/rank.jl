
module Rank

using Printf
import Random
import ..Jtac

# -------- ELO Estimation ---------------------------------------------------- #

include("mlelo.jl")

# -------- Competing --------------------------------------------------------- #

function plan_matches(n, l, a) # matches, players total, players active

  pairings = l * l - (l-a)*(l-a) - a

  n < pairings && @info "Not every pair competes"

  matches = zeros(Int, pairings)
  matches .+= floor(Int, n / pairings)
  matches[1:(n % pairings)] .+= 1

  Random.shuffle(matches)
end


"""
  compete(players, n [, active]; <keyword arguments>)

Create game results for a number of `n` games between players at positions `i`
and `j` in the collection `players` if at least one of `i` or `j` is in
`active`. 

The playings start with `game`, which is infered automatically from
`players` if possible. `callback()` is called after each of the `n` matches.

The return value is a result array of dimensions `(l, l, 3)`, where `l
= length(players)`. The entry at indices `[i,j,k]` stands for the number of
outcomes `k` (`0`: loss, `1`: draw, `2`: win) when player `i` played against
`j`. When both `i` and `j` are inactive, no games are played. The entries at
`[i, j, :]` are `0` in this case.

# Arguments
- `game`: Initial game state. Inferred automatically if possible.
- `callback`: Function called after each of the `n` matches.
- `distributed = false`: Whether to conduct the self-playings on several
   processes in parallel. If `true`, all available workers are used. Alternatively,
   a list of worker ids can be passed.
- `tickets`: Number of chunks in which the workload is distributed if
   `distributed != false`. By default, it is set to the number of workers.
"""
function compete( players
                , n :: Int
                , active = 1:length(players)
                ; game = Jtac.derive_gametype(players)()
                , callback = () -> nothing
                , distributed = false
                , tickets = nothing )

  # If the competitions takes place in parallel, get the list of workers
  # and cede to the corresponding function in distributed.jl
  if distributed

    workers = distributed == true ? Distributed.workers() : distributed
    tickets = isnothing(tickets) ? length(workers) : tickets

    return compete_distributed( players, n, active
                              ; game = game
                              , callback = callback
                              , workers = workers
                              , tickets = tickets )
  end

  matches = plan_matches(n, length(players), length(active))
  results = zeros(Int, length(players), length(players), 3)
  players = enumerate(players)

  l = 1

  for (i, p1) in players, (j, p2) in players

    if i != j && (i in active || j in active)

      asyncmap(1:matches[l]) do _
        k = Jtac.pvp(p1, p2, game = game) + 2 # convert -1, 0, 1 to indices 1, 2, 3
        results[i, j, k] += 1
        callback()
      end

      l += 1
    end
  end

  results
end

function compete_distributed(args...; kwargs...) 
  sum(with_workers(compete, args...; kwargs...))
end

# -------- Rankings ---------------------------------------------------------- #

"""
Structure that contains the ranking information of a group of competing players.
Based on the results of a competition, elo values (and guesses for their
standard deviation) as well as the start advantage and a draw bandwidth is
determined.
"""
struct Ranking
  players :: Vector{String}
  results :: Array{Int, 3}

  elo  :: Array{Float64}
  sadv :: Float64
  draw :: Float64

  elostd  :: Array{Float64}
  sadvstd :: Float64
  drawstd :: Float64
end

"""
    Ranking(players, results; steps = 100)
    Ranking(players, n [, active]; steps = 100, <keyword arguments>)

Get a ranking of `players` based on `results` from a previous call of the
function `compete`. Alternatively, results are generated on the fly via
`compete` for `n` games with all corresponding keyword arguments.  The number of
steps in the iterative maximum likelihood solver can be specified via `steps`.
"""
function Ranking(players :: Vector{String}, results :: Array{Int, 3}; steps = 100)
  elo, sadv, draw = mlestimate(results; steps = steps)
  elostd, sadvstd, drawstd = stdestimate(results, elo, sadv, draw)
  Ranking(players, results, elo, sadv, draw, elostd, sadvstd, drawstd)
end

function Ranking(players, results :: Array{Int, 3}; steps = 100)
  Ranking(Jtac.name.(players), results, steps = steps)
end

function Ranking(players, args...; steps = 100, kwargs...)
  results = compete(players, args...; kwargs...)
  Ranking(players, results; steps = steps)
end


# -------- Visualize Rankings ------------------------------------------------ #

function Base.string(rk :: Ranking, matrix = false)

  # Formatting values dynamically (which does *not* work with @printf...)
  ndigits(x) = (round(Int, x) |> digits |> length) + (x < 0)
  sp(x, n) = repeat(" ", max(n - ndigits(round(Int, x)), 0))

  # The player with highest elo will come first
  perm = sortperm(rk.elo) |> reverse

  players, elo, elostd = rk.players[perm], rk.elo[perm], rk.elostd[perm]
  nv = maximum(ndigits, elo)
  ns = maximum(ndigits, elostd)

  rows = map(1:length(perm), players, elo, elostd) do i, p, v, s
    Printf.@sprintf "%2d. %s%.0f ± %s%.0f %15s" i sp(v, nv) v sp(s, ns) s p
  end

  if matrix

    mat = rk.results[perm, perm, 3] - rk.results[perm, perm, 1]
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

end # module Rank
