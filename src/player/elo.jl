# Elo by maximum likelihood estimation
# Search for 'Bradley-Terry' models for more context

# We use an algorithm proposed by Hunter 2004 as reference, and trusted in the
# notes of Remi Coulom, who wrote about this idea here:
#
#   https://www.remi-coulom.fr/Bayesian-Elo/

using Statistics

# One MM update step

function mlstep(w, l, d, gamma, sadv, draw)
  gamma_ = map(1:length(gamma)) do i
    a = sum(w[i, :] .+ l[:, i] .+ d[i, :] .+ d[:, i])
    b1 = sadv * sum((d[i, :] .+ w[i, :]) ./ (sadv * gamma[i] .+ draw * gamma))
    b2 = sadv * draw * sum((d[i, :] .+ l[i, :]) ./ (sadv * draw * gamma[i] .+ gamma))
    b3 = draw * sum((d[:, i] .+ w[:, i]) ./ (sadv * gamma .+ draw * gamma[i]))
    b4 = sum((d[:, i] .+ l[:, i]) ./ (draw * sadv * gamma .+ gamma[i]))
    a / (b1 + b2 + b3 + b4)
  end
  sadv_ = begin
    a = sum(w .+ d)
    b1 = sum((w .+ d) .* gamma ./ (sadv .* gamma .+ draw .* gamma'))
    b2 = draw * sum((l .+ d) .* gamma ./ (sadv * draw * gamma .+ gamma'))
    a / (b1 + b2)
  end
  draw_ = begin
    a = sum(d)
    b1 = sum((w .+ d) .* gamma' ./ (sadv .* gamma .+ draw .* gamma'))
    b2 = sadv * sum((l .+ d) .* gamma ./ (draw * sadv * gamma .+ gamma'))
    b = b1 + b2
    (a + sqrt(a^2 + b^2)) / b
  end
  gamma_ / sum(gamma_), sadv_, draw_
end

function elotoscale(elo, sadv, draw)
  f(x) = exp10(x / 400)
  scale = f.(elo)
  scale / sum(scale), f(sadv), f(draw)
end

function scaletoelo(gamma, sadv, draw)
  f(x) = 400 * log10(x)
  elo = f.(gamma)
  elo .- mean(elo), f(sadv), f(draw)
end

function mlestimate(wins, losses, draws; steps = 100 )
  n = size(wins, 1)
  @assert size(wins) == size(losses) == size(draws)
  @assert size(wins) == (n, n)

  gamma = ones(n) / n
  sadv = 1.0
  draw = 1.5

  for i in 1:steps
    gamma, sadv, draw = mlstep(wins, losses, draws, gamma, sadv, draw)
  end
  
  scaletoelo(gamma, sadv, draw)
end

function mlestimate(res :: Array{T, 3}; kwargs...) where {T <: Real}
  mlestimate(res[:, :, 3], res[:, :, 1], res[:, :, 2]; kwargs...)
end

# Error estimates
#
# Two reasonable candidates for estimating the error of the MLE are the inverse
# Fisher information (see Cramer Rao bound) or the inverse *observed* Fisher
# information (even easier to calculate, has a Bayesian flavour, but is used
# less, as far as I can see). Bradley Efron wrote a favorable article about it
# in 1978:
#
#   https://academic.oup.com/biomet/article-abstract/65/3/457/233667
#
# Comment: we seem to run into problems when trying to invert the Fisher
# information matrix - probably, because the elo is translation-invariant
# and we thus do not have full dimensionality in the hessian
#
# We can still use diagonal entries of the information matrix for estimates
# of the deviation. The results seem reasonable.


# Calculate the loglikelihood of the elo model given data
# we can use this to check if our gradient / Hessian formulas are correct

safelog(x) = log(max(1e-50, x))
safediv(x, y) = sign(y) * x / (abs(y) + 1e-50)
safesqrt(x) = x >= 0 ? sqrt(x) : Inf

function logl(w, l, d, elo, sadv, draw)
  gamma, sadv, draw = elotoscale(elo, sadv, draw)
  p = sadv .* gamma ./ (sadv .* gamma .+ draw .* gamma')
  q = gamma' ./ (sadv .* draw .* gamma .+ gamma')
  sum(w .* safelog.(p) .+ l .* safelog.(q) .+ d .* safelog.(1 .- p .- q))
end

function logl(res :: Array{T, 3}, args...) where {T}
  logl(res[:, :, 3], res[:, :, 1], res[:, :, 2], args...)
end

# Actual estimate for the standard error as inverse observed information
# If something goes wrong (i.e., the diagonal of the Hessian of the
# loglikelihood is not negative), we return an error of Inf.
# Possible reasons: degenerate data or subpar mlestimate

function stdestimate(w, l, d, elo, sadv, draw)
  c = log(10) / 400.
  f(x) = exp10(x/400.)
  fp = f.(elo' .- elo .- sadv .+ draw)
  fq = f.(elo .- elo' .+ sadv .+ draw)
  p, q = 1 ./ (1 .+ fp), 1 ./ (1 .+ fq)
  pd = - c .* fp .* p.^2
  qd = - c .* fq .* q.^2
  pdd = c .* pd .* (1 .- 2 .* p .* fp)
  qdd = c .* qd .* (1 .- 2 .* q .* fq)

  h1 = safediv.(pdd, p) .- safediv.(pd, p).^2
  h2 = safediv.(qdd, q) .- safediv.(qd, q).^2
  h3 = safediv.(pdd .+ qdd, 1 .- p .- q) .+ safediv.(pd .- qd, 1 .- p .- q).^2
  h4 = safediv.(pdd .+ qdd, 1 .- p .- q) .+ safediv.(pd .+ qd, 1 .- p .- q).^2
  h = w .* h1 .+ l .* h2 .- d .* h3

  sadv_std = safesqrt(-1 / sum(h))
  draw_std = safesqrt(-1 / sum(w .* h1 .+ l .* h2 .- d .* h4))
  elo_std = safesqrt.(-1 ./ (sum(h, dims = 2) + sum(h, dims = 1)'))

  reshape(elo_std, :), sadv_std, draw_std
end

function stdestimate(res :: Array{T, 3}, args...) where {T}
  stdestimate(res[:, :, 3], res[:, :, 1], res[:, :, 2], args...)
end

