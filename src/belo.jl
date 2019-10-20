
# -------- Bayes ELO Objective ----------------------------------------------- #

elosigm(d) = 1 / (1 + exp(d/400))

function likelihood(elo1, elo2, res; draw, start)

  p1 = elosigm(elo2 - elo1 - start + draw)
  p2 = elosigm(elo1 - elo2 + start + draw)

  res == 1 ? p1 : (res == -1 ? p2 : 1 - p1 - p2)

end

function loglikelihood(elos, results; kwargs...)

  sum(CartesianIndices(results)) do c
    (i,j,k) = Tuple(c)
    results[i,j,k] * log(likelihood(elos[i], elos[j], k-2; kwargs...))
  end

end

logprior(elos; mean, std) = sum(elo -> -0.5 * ((elo - mean) / std)^2, elos)

function objective(elos, results; mean, std, kwargs...)

  loglikelihood(elos, results; kwargs...) + 
  logprior(elos; mean = mean, std = std)

end


# -------- ELO Estimation ---------------------------------------------------- #

function estimate( results  # result array (see function compete below)
                 ; start = nothing
                 , draw = nothing
                 , mean = 0.
                 , std = 1500. )

  # Get the number of players
  k = size(results, 1)

  # Upper and lower bounds and start values for (start, draw)
  lb = [-4std, 1]
  ub = [ 4std, 4std]
  st = [ 0., 1.]

  # If start or draw are given, we do not estimate them but set 
  # upper bound = lower bound = start value = given value
  !isnothing(start) && (st[1] = lb[1] = ub[1] = start)
  !isnothing(draw)  && (st[2] = lb[2] = ub[2] = draw)

  # Create the optimizer object
  opt = NLopt.Opt(:LN_BOBYQA, k+2)

  # Define the minimization objective function compatible to NLopt
  f = (x, _) -> -objective( x[1:k]
                          , results
                          ; draw = x[k+2]
                          , start = x[k+1]
                          , mean = mean
                          , std = std )

  # Set the objective and set reasonable lower/upper bounds
  NLopt.min_objective!(opt, f)
  NLopt.lower_bounds!(opt, [fill(mean - 4std, k); lb])
  NLopt.upper_bounds!(opt, [fill(mean + 4std, k); ub])

  # Conduct the optimization
  value, params, ret = NLopt.optimize(opt, [fill(mean, k); st] )

  # Return the found values
  params

end


# -------- Competitions ------------------------------------------------------ #


function match_repetitions(n, pairings)

  n < pairings && @warn "Not every pair competes"

  matches = zeros(Int, pairings)
  matches .+= floor(Int, n / pairings)
  matches[1:(n % pairings)] .+= 1

  shuffle(matches)

end

pairings(l, a) = l * l - (l-a)*(l-a)


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
                ; game = derive_gametype(players)()
                , callback = () -> nothing
                , distributed = false
                , tickets = nothing )

  # If the competitions takes place in parallel, get the list of workers
  # and cede to the corresponding function in distributed.jl
  if distributed != false

    workers = distributed == true ? Distributed.workers() : distributed
    tickets = isnothing(tickets) ? length(workers) : tickets

    return compete_distributed( players, n, active
                              ; game = game
                              , callback = callback
                              , workers = workers
                              , tickets = tickets )

  end

  # Get the number of pairings that compete
  m = pairings(length(players), length(active))
  
  matches = match_repetitions(n, m)

  results = zeros(Int, length(players), length(players), 3)
  players = enumerate(players)

  l = 1

  for (i, p1) in players, (j, p2) in players

    if i != j && (i in active || j in active)

      asyncmap(1:matches[l]) do _
        k = pvp(p1, p2, game) + 2 # convert -1, 0, 1 to indices 1, 2, 3
        results[i, j, k] += 1
        callback()
      end

      l += 1

    end

  end

  results

end


# -------- Rankings ---------------------------------------------------------- #

"""
Structure that contains the ranking information of a group of competing players.
Based on the results of a competition, elo values and the start advantage as
well as draw bandwidth is determined.
"""
struct Ranking

  players :: Vector{Player}
  results :: Array{Int, 3}

  elos :: Array{Float64}

  draw :: Float64
  start :: Float64
  
end

"""
    Ranking(players, results; <keyword arguments>)
    Ranking(players, n [, active]; <keyword arguments>)

Get a ranking of `players` based on `results` from a previous call of the
function `compete`. Alternatively, results are generated on the fly via
`compete` for `n` games with all corresponding keyword arguments.

# Arguments
The remaining arguments alter how the Bayesian elo values are calculated.
- `mean = 0`: Mean value of the prior elo distribution.
- `std = 1500`: Standard deviation of the prior elo distribution.
- `start = nothing`: If not `nothing`, the start advantage is fixed to this value.
- `draw = nothing`: If not `nothing`, the draw bandwidth is fixed to this value.
"""
function Ranking(players, results :: Array{Int, 3}; kwargs...)

  est = estimate(results; kwargs...)

  k = size(results, 1)
  elos  = est[1:k]
  start = est[k+1]
  draw  = est[k+2]

  Ranking(players, results, elos, draw, start)

end

function Ranking( players
                , args...
                ; game = derive_gametype(players)()
                , callback = () -> nothing
                , distributed = false
                , tickets = nothing
                , kwargs...)

  results = compete( players
                   , args...
                   ; game = game
                   , callback = callback
                   , distributed = distributed
                   , tickets = tickets )

  Ranking(players, results; kwargs...)

end


# -------- Visualize Rankings ------------------------------------------------ #


function Base.summary(rk :: Ranking, matrix = false)

  # The player with highest elo will come first
  perm = sortperm(rk.elos) |> reverse

  ranks = map(1:length(perm), rk.players[perm], rk.elos[perm]) do i, p, elo
    Printf.@sprintf "%2d. %10.2f  %-s" i elo name(p)
  end

  start = Printf.@sprintf "start advantage: %.2f" rk.start
  draw  = Printf.@sprintf "draw bandwidth:  %.2f" rk.draw

  join([ranks; ""; start; draw], "\n")

end

