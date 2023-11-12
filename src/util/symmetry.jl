
cind(i, s) = Tuple(CartesianIndices(s)[i])
lind(c, s) = LinearIndices(s)[c...]

nomirror(matrix) = copy(matrix)
nomirror(a :: Tuple, s) = a
nomirror(a :: Int, s) = a

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


"""
Symmmetry group of a board game.
"""
abstract type SymmetryGroup end

"""
    groupoperations(group)

Return the group operations of `group`.
"""
groupoperations(grp :: SymmetryGroup) = error("Not implemented")

"""
    applygroup(group, matrix)

Apply all group operations of the symmetry group `group` to `matrix`.

See also [`groupoperations`](@ref).
"""
function applygroup(group :: SymmetryGroup, matrix :: Array)
  map(groupoperations(group)) do ops
    for op in ops
      matrix = op(matrix)
    end
    matrix
  end
end

"""
    applygroup(group, action, size)

Apply all group operations transforming the action index `action` on a board
of size `size`.
"""
function applygroup(group :: SymmetryGroup, action :: Int, s :: Tuple)
  map(groupoperations(group)) do ops
    for op in ops
      action = op(action, s)
    end
    action
  end
end

"""
The full symmetry group of square matrices with 8 operations.
"""
struct DihedralGroup <: SymmetryGroup end

function groupoperations(:: DihedralGroup)
  [
    [nomirror],
    [hmirror],
    [dmirror],
    [vmirror],
    [hmirror, dmirror],
    [dmirror, hmirror],
    [hmirror, vmirror],
    [dmirror, hmirror, vmirror],
  ]
end

"""
The full symmetry group of non-square matrices with 4 operations.
"""
struct KleinFourGroup <: SymmetryGroup end

function groupoperations(:: KleinFourGroup)
  [
    [nomirror],
    [hmirror],
    [vmirror],
    [hmirror, vmirror],
  ]
end

