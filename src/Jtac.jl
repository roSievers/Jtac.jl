# Jtac.jl
# A julia implementation of the Alpha zero learning design

module Jtac

# We use machine learning capabilities of the package Knet.jl

using AutoGrad, Knet
export AutoGrad, Knet

# We save models/games via BSON.jl

using BSON

# Interface that games must satisfy and some convenience functions

include("game.jl")
export Game, Status, ActionIndex,
       status, current_player, legal_actions, apply_action!, 
       is_action_legal, representation, policy_length, random_playout, 
       augment, draw


# Interface for models

include("model.jl")
export Model, apply, save_model, load_model

# Building blocks for models

include("layers.jl")
export Dense, Conv, Chain, Dropout, Batchnorm

# Model implementations

include("models/toymodels.jl")
export DummyModel, RandomModel, RolloutModel

include("models/basemodels.jl")
export BaseModel, Shallow, MLP, ShallowConv

# Markov chain tree search with model predictions

include("mc.jl")
export mctree_turn!

# Game implementations

include("games/tictactoe.jl")
include("games/metatac.jl")
export TicTacToe, MetaTac

# Loss and update steps for learning

include("learning.jl")
export DataSet, loss, record_selfplay, 
       set_optimizer!, train_step!

# Players

include("player.jl")
export RandomPlayer, MCTPlayer, 
       IntuitionPlayer, HumanPlayer, 
       pvp

end # module JTac
