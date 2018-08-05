function print_board(game)
    for o_row = 1:3, i_row = 1:3
        print_row(game, o_row, i_row)
        if i_row == 3 && o_row != 3
            println(" ---------------------")
        end
    end
end

function print_row(game, o_row, i_row)
    for o_col = 1:3, i_col = 1:3
        cell_index = combined_index(o_row, o_col)
        if game.focus == cell_index && i_col == 1
            print("#")
        else
            print(" ")
        end
        print_char(game.board, combined_index(o_row, o_col, i_row, i_col))
        if i_col == 3 && o_col != 3
            print(" |")
        end
    end
    print(" ")
    println("")
end

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