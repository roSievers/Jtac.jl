function print_char(board, index)
    entry = board[index]
    if entry == 0
        print(".")
    elseif entry == 1
        print("X")
    elseif entry == 2
        print("O")
    else
        print("?")
    end
end

function print_row(board, o_row, i_row)
    print(" ")
    for o_col = 1:3, i_col = 1:3
        print_char(board, combined_index(o_row, o_col, i_row, i_col))
        print(" ")
        if i_col == 3 && o_col != 3
            print("| ")
        end
    end
    println("")
end

function print_board(board)
    for o_row = 1:3, i_row = 1:3
        print_row(board, o_row, i_row)
        if i_row == 3 && o_row != 3
            println(" ---------------------")
        end
    end
end