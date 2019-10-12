
# -------- Bayes ELO Objective ----------------------------------------------- #

elosigm(d) = 1 / (1 + exp(d/400))

function likelihood(elo1, elo2, res; draw, adv)
  p1 = elosigm(elo2 - elo1 - adv + draw)
  p2 = elosigm(elo1 - elo2 + adv + draw)
  res == 1 ? p1 : (res == -1 ? p2 : 1 - p1 - p2)
end

function loglikelihood(elos, games; kwargs...)
  sum(games) do (p1, p2, res)
    log( likelihood(elos[p1], elos[p2], res; kwargs...) )
  end
end

function logprior(elos; mean, std)
  sum(elos) do elo
    -0.5 * ((elo - mean) / std)^2
  end
end

function objective(elos, games; mean, std, kwargs...)
  loglikelihood(elos, games; kwargs...) + 
  logprior(elos; mean = mean, std = std)
end


# -------- ELO Estimation ---------------------------------------------------- #

function estimate( k      # number of players
                 , games  # triples (player1, player2, result)
                 ; adv = nothing
                 , draw = nothing
                 , mean = 0.
                 , std = 1500. )

  # Upper and lower bounds and start values for (adv, draw)
  lb = [-4std, 1]
  ub = [ 4std, 4std]
  st = [ 0., 1.]

  # If adv or draw are given, we do not estimate them but set 
  # upper bound = lower bound = start value = given value
  !isnothing(adv)  && (st[1] = lb[1] = ub[1] = adv)
  !isnothing(draw) && (st[2] = lb[2] = ub[2] = draw)

  # Create the optimizer object
  opt = NLopt.Opt(:LN_BOBYQA, k+2)

  # Define the minimization objective function compatible to NLopt
  f(x, _) = -objective( x[1:k]
                      , games
                      ; draw = x[k+2]
                      , adv = x[k+1]
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


# -------- Tournaments ------------------------------------------------------- #

# Number of matches per active player when at most nmax games
# are to be played.
function match_repetitions(nmax, players, active)

  r = length(active)
  k = length(players) - r
  floor(Int, nmax / (r * (r-1) + 2k*r))

end

# Create vector of game results (p1, p2, result) for a number of players of type
# Player. We distinguish between active and inactive players. Two inactive
# players do not play against one another, but each active player plays against
# each player.
function playouts( players
                 , game :: Game
                 , nmax :: Int
                 ; active = 1:length(players)
                 , async = false
                 , callback = () -> nothing )

  n = match_repetitions(nmax, players, active)

  if n == 0
    @warn "nmax too small: every pair gets one game"
    n = 1
  end

  games = []
  players = enumerate(players)

  for (i, p1) in players, (j, p2) in players

    if i == j || !(i in active || j in active)
      # in this case, we do not play the game, because the players are the same
      # or none of the players is active
      continue
    end

    if async
      push!(games, asyncmap(_ -> (i, j, pvp(p1, p2, game)), 1:n))
    else
      push!(games, map(_ -> (i, j, pvp(p1, p2, game)), 1:n))
    end

    callback()

  end

  vcat(games...)

end

function playouts(players, nmax :: Int; kwargs...)
  playouts(players, derive_gametype(players)(), nmax; kwargs...)
end


# -------- Rankings ---------------------------------------------------------- #

"""
    ranking(players, [game,] nmax; <keyword arguments>)

Create a ranking of `players` competing at `game` with at most `nmax` plays.
If at least one of the players is specialized to a concrete game type, the
argument `game` may be left out. Returns a named tuple with entries `:elos`,
`:adv`, and `:draw`.

# Arguments
The function takes the following keyword arguments:
- `active`: A selection of indices of players that are regarded as active.
- `cache`: Results created by `playouts`, which are considered for the ranking.
- `async`: Whether to run the pvp playouts via `asyncmap`.
- `callback`: Function that is called after each match.
"""
function ranking(players, playouts :: Vector; kwargs...)

  k = length(players)
  res = estimate(k, playouts; kwargs...)
  (elos = res[1:k], adv = res[k+1], draw = res[k+2])

end

function ranking( players
                , game :: Game
                , nmax
                ; cache = [] # Vector of game results
                , active = 1:length(players)
                , async = false
                , callback = () -> nothing
                , kwargs...)

  k = length(players)
  games = playouts( players
                  , game
                  , nmax
                  ; active = active
                  , async = async
                  , callback = callback )

  ranking(players, [games; cache]; kwargs...)

end

function ranking(players, nmax; kwargs...)
  ranking(players, derive_gametype(players)(), nmax; kwargs...)
end


# -------- Visualize Rankings ------------------------------------------------ #

"""
    print_ranking(players, ranking [; prepend])

Print a ranking of `players` given by `ranking`. Optionally, the string
`prepend` can be prepended to each printed line.
"""
function print_ranking(players, rk :: NamedTuple; prepend = "")

  i = 1

  # The player with highest elo will come first
  perm = sortperm(rk.elos) |> reverse

  for (player, elo) in zip(players[perm], rk.elos[perm])
    Printf.@printf "%s %3d. %10.2f  %-s\n" prepend i elo name(player)
    i += 1
  end

  Printf.@printf( "%s\n%s start advantage: %.2f\n%s draw bandwidth:  %.2f\n"
                , prepend
                , prepend
                , rk.adv
                , prepend
                , rk.draw )

end

function print_ranking(players, args...; prepend = "", kwargs...)
  rk = ranking(players, args...; kwargs...)
  print_ranking(players, rk; prepend = prepend)
end

