# Returns an index between 1 and 81
function combined_index(o_row, o_col, i_row, i_col)
    27 * (o_row - 1) + 9 * (o_col - 1) + 3 * (i_row - 1) + i_col
end

# Takes a single row index and a single col index
# Returns an index between 1 and 9
function single_index(row, col)
    3 * (row - 1) + col
end

function outer_index(combined)
    div(combined - 1, 9) + 1
end

function inner_index(combined)
    mod1(combined, 9)
end