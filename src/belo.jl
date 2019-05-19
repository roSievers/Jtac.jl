
import NLopt
import Printf

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
  try
    loglikelihood(elos, games; kwargs...) + logprior(elos; mean = mean, std = std)
  catch err
    show(err)
  end
end

function estimate(k, games; adv = nothing, draw = nothing, mean = 0., std = 1500.)

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
  f(x, _) = -objective(x[1:k], games; draw = x[k+2], adv = x[k+1], mean = mean, std = std)

  # Set the objective and set reasonable lower/upper bounds
  NLopt.min_objective!(opt, f)
  NLopt.lower_bounds!(opt, [fill(mean - 4std, k); lb])
  NLopt.upper_bounds!(opt, [fill(mean + 4std, k); ub])

  # Conduct the optimization
  value, params, ret = NLopt.optimize(opt, [fill(mean, k); st] )

  # Return the found values
  params

end


# Returns a named tuple with entries :elos, :draw, :adv
function ranking(players, game :: Game, nmax; async = false, kwargs...)
  
  k = length(players)
  n = floor(Int, nmax / binomial(k, 2) / 2)

  @assert n > 0 "nmax too small: not every match is assumed"

  games = []
  players = enumerate(players)

  for (i, p1) in players, (j, p2) in players

    i == j && continue

    if async
      push!(games, asyncmap(_ -> (i, j, pvp(p1, p2, game)), 1:n))
    else
      push!(games, map(_ -> (i, j, pvp(p1, p2, game)), 1:n))
    end

  end

  games = vcat(games...)

  res = estimate(k, games; kwargs...)

  (elos = res[1:k], adv = res[k+1], draw = res[k+2])

end

function print_ranking(players, rk)

  i = 1

  # The player with highest elo will come first
  perm = sortperm(rk.elos) |> reverse

  for (player, elo) in zip(players[perm], rk.elos[perm])
    Printf.@printf "%3d. %7.2f %10s\n" i elo name(player)
    i += 1
  end

  Printf.@printf "\nstart-advantage: %.2f\ndraw-bandwidth:  %.2f" rk.adv rk.draw

end

function print_ranking(players, game, nmax; kwargs...)
  rk = ranking(players, game, nmax; kwargs...)
  print_ranking(players, rk)
end

