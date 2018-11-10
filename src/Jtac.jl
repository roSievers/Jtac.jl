# Jtac.jl
# A julia implementation of the Alpha zero learning design

# Interface that games must satisfy and some convenience functions
include("game.jl")
export Game, Status, ActionIndex

# Interface for models
include("model.jl")
export Model

# Model implementations
include("models/layers.jl")
include("models/toymodels.jl")
export DummyModel, RandomModel, RolloutModel, LinearModel

# Markov chain tree search with model predictions
include("mc.jl")
export mctree_turn!, mctree_vs_random

# Game implementations
include("games/metatac.jl")
include("games/tictactoe.jl")
#include("games/four3d.jl")
#include("games/chess.jl")
export MetaTac, TicTacToe #, Four3d, Chess

# Loss for learning
include("learning.jl")
export loss, record_selfplay

# Players
include("player.jl")
export RandomPlayer, MCTPlayer, 
       PolicyPlayer, SoftPolicyPlayer, 
       HumanPlayer

