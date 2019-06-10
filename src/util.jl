

# Symmetry operations on matrix-shaped data representations of games

hmirror(matrix) = matrix[end:-1:1, :]
vmirror(matrix) = matrix[:, end:-1:1]

function apply_dihedral_group(matrix)
  [
    matrix |> copy,
    matrix |> hmirror,
    matrix |> transpose,
    matrix |> vmirror,
    matrix |> transpose |> hmirror,
    matrix |> hmirror |> transpose,
    matrix |> vmirror |> hmirror,
    matrix |> vmirror |> hmirror |> vmirror
  ]
end


# Choose an index from a proper probability vector

function choose_index(probs)
  r = rand(Float32)
  index = findfirst(x -> r <= x, cumsum(probs))
  @assert index != nothing "probability vector is not proper!"
  index
end

