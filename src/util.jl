
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


# -------- Pretty Printing to Monitor Training ------------------------------- #

const gray_crayon = Crayons.Crayon(foreground = :dark_gray)
const default_crayon = Crayons.Crayon(reset = true)

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

function print_ranking(rk)

  # Log that a contest comes next
  print(gray_crayon)
  println("#\n# Contest with $(length(rk.players)) players:\n#")

  # Get the summary of the contest and print it
  s = "# " * replace(summary(rk, true), "\n" => "\n# ") * "\n#"
  println(s)

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


# ------- Switch Knet Allocator ---------------------------------------------- #

# TODO: This is an ugly hack that should be discussed with the Knet people.
a = true
Knet.cuallocator() = a
switch_knet_allocator() = (global a; a = !a)


# -------- Parallel-Stable Progress Maps ------------------------------------- #

"""
    stepper(description, length; <keyword arguments>)

Return a pair of functions `(step, finish)` that control a progress bar with
title `description` and `length` number of steps. For the keyword arguments, see
`ProgressMeter.Progress`.

The progress bar is running in its own thread and is communicated with through
a `RemoteChannel`. This allows calling the `step` function safely under any
condition. However, you **must** make sure that `step()` is not used after
`finish()` anymore. This means that all tasks or futures have to be fetched
before calling `finish`.

# Examples

```julia
step, finish = stepper("Progress", 15)

# Do some calculations in a parallel fashion
pmap(1:15) do i 
  sleep(0.5)
  step()
  i
end

# Since pmap fetches all tasks/futures it creates, we can safely finish
finish()
```
"""
function stepper(description, n :: Int, kwargs...)

  glyphs = ProgressMeter.BarGlyphs("[=>⋅]")
  progress = ProgressMeter.Progress( n + 1
                                   ; dt = 0.1
                                   , desc = description
                                   , barglyphs = glyphs
                                   , kwargs... )

  # Remote channel that can be used to signal a step on any process
  channel = RemoteChannel(() -> Channel{Bool}(0))

  # Thread that manages progressing the progress bar
  thread = @async begin

    while take!(channel)
      ProgressMeter.next!(progress)
    end

    ProgressMeter.printover(progress.output, "")

  end

  step = () -> (put!(channel, true); yield())
  finish = () -> (put!(channel, false); fetch(thread))

  step, finish

end

