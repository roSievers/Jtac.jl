
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

function estimate(k, games; adv = nothing, draw = nothing, mean = 0, std = 1500)
  if isnothing(draw) && isnothing(adv)
    opt = NLopt.Opt(:LN_BOBYQA, k+2)
    NLopt.min_objective!(opt, (x, _) -> -objective(x[1:k], games; draw = x[k+2], adv = x[k+1], mean = mean, std = std))
    NLopt.lower_bounds!(opt, [fill(mean - 3std, k+1); 1])
    NLopt.upper_bounds!(opt, [fill(mean + 3std, k+1); abs(mean) + 3std])
    value, params, ret = NLopt.optimize(opt, [[mean for _ in 1:k]; 0.; 1.] )
  else
    @show adv
    @show draw
    error("Fixed adv or draw values not yet implemented")
  end
  params
end


function ranking(players, game :: Game, nmax; kwargs...)
  
  k = length(players)
  n = floor(Int, nmax / binomial(k, 2) / 2)

  @assert n > 0 "nmax too small: not every match is assumed"

  games = []
  players = enumerate(players)

  for (i, p1) in players, (j, p2) in players
    i == j && continue

    g = [(i, j, pvp(p1, p2, game)) for _ in 1:n]
    push!(games, g)
  end

  games = vcat(games...)

  estimate(k, games; kwargs...)

end

function show_ranking(players, game, nmax; adv = nothing, draw = nothing, kwargs...)
  
  rk = ranking(players, game, nmax; adv = adv, draw = draw, kwargs...)
  elos = rk[1:length(players)]

  if isnothing(adv) && isnothing(draw)
    adv, draw = rk[end-1:end]
  else
    error("Fixed adv or draw values not yet implemented")
  end

  perm = sortperm(elos) |> reverse
  i = 1
  for (player, elo) in zip(players[perm], elos[perm])
    Printf.@printf "%3d. %7.2f %10s\n" i elo name(player)
    i += 1
  end

  Printf.@printf "\nstart-advantage: %.2f\ndraw_bandwidth:  %.2f" adv draw

end

function show_ranking(players, elos)
  perm = sortperm(elos) |> reverse
  for (player, elo) in zip(players[perm], elos[perm])
    println(name(player), ": ", elo)
  end
end


