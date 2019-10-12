
# -------- Miscellaneous ----------------------------------------------------- #

second((a, b)) = b

unzip(a :: Array{<: Tuple}) = map(first, a), map(second, a)

function choose_index(probs :: Vector{Float32}) :: Int

  @assert all(probs .>= 0) && sum(probs) ≈ 1.0 "probability vector not proper"

  r = rand(Float32)
  index = findfirst(x -> r <= x, cumsum(probs))

  isnothing(index) ? length(probs) : index

end

function derive_gametype(players)

  gt = mapreduce(gametype, typeintersect, players, init = Game)

  @assert gt != Union{} "Players do not play compatible games"
  @assert !isabstracttype(gt) "Cannot infere game from abstract type"

  gt

end

isasync(m) = isa(m, Async) ? true : false


# -------- Symmetry ---------------------------------------------------------- #

cind(i, s) = Tuple(CartesianIndices(s)[i])
lind(c, s) = LinearIndices(s)[c...]

hmirror(matrix) = reshape(matrix[end:-1:1, :, :], size(matrix))
hmirror(a :: Tuple, s) = (s[1]+1-a[1], a[2:end]...)
hmirror(a :: Int, s) = lind(hmirror(cind(a, s), s), s)

vmirror(matrix) = reshape(matrix[:, end:-1:1, :], size(matrix))
vmirror(a :: Tuple, s) = (a[1], s[2]+1-a[2], a[3:end]...)
vmirror(a :: Int, s) = lind(vmirror(cind(a, s), s), s)

dmirror(matrix :: Matrix) = permutedims(matrix, (2, 1))
dmirror(matrix :: Array{T,3}) where {T} = permutedims(matrix, (2, 1, 3))
dmirror(a :: Tuple, s) = (a[2], a[1], a[3:end]...)
dmirror(a :: Int, s) = lind(dmirror(cind(a, s), s), s)


# Full symmetry group of square matrices

function apply_dihedral_group(matrix :: Array)
  [
    matrix |> copy,
    matrix |> hmirror,
    matrix |> dmirror,
    matrix |> vmirror,
    matrix |> dmirror |> hmirror,
    matrix |> hmirror |> dmirror,
    matrix |> vmirror |> hmirror,
    matrix |> vmirror |> hmirror |> dmirror
  ]
end

function apply_dihedral_group(action :: Int, s :: Tuple)
  [
    action,
    hmirror(action, s),
    dmirror(action, s),
    vmirror(action, s),
    hmirror(dmirror(action, s), s),
    dmirror(hmirror(action, s), s),
    hmirror(vmirror(action, s), s),
    dmirror(hmirror(vmirror(action, s), s), s)
  ]
end


# Symmetry group for non-square matrices

function apply_klein_four_group(matrix :: Array)
  [
    matrix |> copy,
    matrix |> hmirror,
    matrix |> vmirror,
    matrix |> vmirror |> hmirror
  ]
end

function apply_klein_four_group(action :: Int, s :: Tuple)
  [
    action,
    hmirror(action, s),
    vmirror(action, s),
    vmirror(hmirror(action, s), s)
  ]
end


# -------- Pretty Printing --------------------------------------------------- #

const gray_crayon = Crayons.Crayon(foreground = :dark_gray)
const default_crayon = Crayons.Crayon(reset = true)

function progressmeter( n, desc
                      ; dt = 0.5
                      , kwargs... )

  glyphs = ProgressMeter.BarGlyphs("[=>⋅]")
  ProgressMeter.Progress( n
                        , dt = dt
                        , desc = desc
                        , barglyphs = glyphs
                        , kwargs... )

end

progress! = ProgressMeter.next!

clear_output!(p) = ProgressMeter.printover(p.output, "")


# -------- Pretty Printing to Monitor Training ------------------------------- #

function print_loss_header(loss, use_features)

  names = use_features ? caption(loss) : caption(loss)[1:3]

  components = map(names) do c
    Printf.@sprintf("%10s", string(c)[1:min(end, 10)])
  end

  println(join(["#"; "   epoch"; components; "     total"; "    length"], " "))

end

function print_loss(l, p, epoch, train, test)

  for (set, cr) in [(train, gray_crayon), (test, default_crayon)]
  
    # Compute the losses and get them as strings
    ls = loss(l, training_model(p), set)
    losses = map(x -> @sprintf("%10.3f", x), ls)

    # Print everything in grey (for train) and white (for test)
    print(cr)
    @printf( "%10d %s %10.3f %10d\n"
           , epoch
           , join(losses, " ")
           , sum(ls)
           , length(set) )

  end
  
end

log_option(s, v) = Printf.@sprintf "# %-22s %s\n" string(s, ":") v

function print_contest(players, contest_length, async, active, cache)

  # Log that a contest comes next
  print(gray_crayon)
  println("#\n# Contest with $(length(players)) players:\n#")

  # Get the length of the progress bar
  r = length(active)
  k = length(players) - r
  n = (r * (r-1) + 2k*r)

  p = progressmeter(n + 1, "# Contest...")

  # Calculate the ranking and print it
  rk = ranking( players
              , contest_length
              , async = async
              , active = active
              , cache = cache
              , callback = () -> progress!(p) )

  clear_output!(p)

  print(gray_crayon)
  print_ranking(players, rk, prepend = "#")
  println("#")

end



# -------- Feature Compability ----------------------------------------------- #

function check_features(l, model)

  features(l) == features(model) && !isempty(features(model))

end

function check_features(l, model, dataset)

  fl = features(l)
  fm = features(model)
  fd = features(dataset)

  if isempty(fm)        return false
  elseif fm == fl == fd return true
  elseif fm == fd       @warn "Loss does not support model features"
  elseif fm == fl       @warn "Dataset does not support model features"
  else                  @warn "Loss and dataset do not support model features"
  end

  false

end

