
cind(i, s) = Tuple(CartesianIndices(s)[i])
lind(c, s) = LinearIndices(s)[c...]

# Symmetry operations on matrix-shaped data representations of games

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


function apply_dihedral_group(matrix :: Array)
  [
    matrix |> copy,
    matrix |> hmirror,
    matrix |> dmirror,
    matrix |> vmirror,
    matrix |> dmirror |> hmirror,
    matrix |> hmirror |> dmirror,
    matrix |> vmirror |> hmirror,
    matrix |> vmirror |> hmirror |> vmirror
  ]
end

function apply_dihedral_group(action :: Int, s :: Tuple)
  [
    action,
    hmirror(action, s),
    dmirror(action, s),
    vmirror(action, s),
    dmirror(hmirror(action, s), s),
    hmirror(dmirror(action, s), s),
    vmirror(hmirror(action, s), s),
    vmirror(hmirror(vmirror(action, s), s), s)
  ]
end


# Choose an index from a proper probability vector

function choose_index(probs)
  r = rand(Float32)
  index = findfirst(x -> r <= x, cumsum(probs))
  @assert index != nothing "probability vector is not proper!"
  index
end


# Manipulation of Console output and progress bars

const gray_crayon = Crayons.Crayon(foreground = :dark_gray)

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


