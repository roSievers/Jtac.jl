
# This is the main file of the library, where all include and export all
# functionality

# Interface that games must satisfy and some convenience functions
include("game.jl")

export Game, Status, ActionIndex


# Game implementations
include("games/metatac.jl")
#include("games/tac.jl")
#include("games/four3d.jl")
#include("games/chess.jl")

export MetaTac #, Tac, Four3d, Chess

