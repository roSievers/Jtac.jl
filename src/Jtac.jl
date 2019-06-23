# Jtac.jl
# A julia implementation of the Alpha zero learning design

module Jtac

using Statistics, LinearAlgebra, Random

# We use machine learning capabilities of the package Knet.jl

import AutoGrad
import Knet
import Knet: identity, relu, softmax, minibatch

export AutoGrad, Knet, 
       minibatch,
       identity, relu, elu, softmax, tanh, sigm

# We save models/games via BSON.jl

import BSON

# We need functions in Random

import Random

# Auxiliary functions
include("util.jl")

# Interface that games must satisfy and some convenience functions

include("game.jl")
export Game, Status, ActionIndex,
       status, current_player, legal_actions, apply_action!, 
       is_action_legal, representation, policy_length, random_playout, 
       augment, draw, is_over, random_turn!


# Interface for models

include("model.jl")
export Model, apply, save_model, load_model, 
       to_gpu, to_cpu, swap, on_gpu, training_model

# Building blocks for models

include("layers.jl")
export Pointwise, Dense, Conv, Deconv, Pool, Dropout, Batchnorm, 
       Chain, Stack,
       valid_insize, outsize, layers,
       @chain, @stack

# Model implementations

include("models/toy.jl")
export DummyModel, RandomModel, RolloutModel

include("models/neural.jl")
export NeuralModel, Shallow, MLP, ShallowConv

include("models/async.jl")
export Async

# Markov chain tree search with model predictions

include("mc.jl")
export mctree_turn!

# Game implementations

include("games/tictactoe.jl")
include("games/tictac554.jl")
include("games/metatac.jl")
export TicTacToe, MetaTac, TicTac554

# Loss and update steps for learning

include("learning.jl")
export DataSet, loss, loss_components, record_selfplay, 
       set_optimizer!, train_step!,
       minibatch # exports Knet.minibatch

# Players

include("player.jl")
export RandomPlayer, MCTSPlayer, 
       IntuitionPlayer, HumanPlayer, 
       pvp, name

include("belo.jl")
export ranking, print_ranking

end # module JTac
