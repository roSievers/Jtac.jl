
"""
Module that implements the Alpha Zero learning design.

The module exports (i) implementations of various two-player board games, like
tic-tac-toe and generalizations thereof, (ii) functions for creating neural
networks that act on the states of games, and (iii) functionality to generate
datasets via the selfplay of a Monte-Carlo Tree Search player assisted by
a neural model. Aspects (ii) and (iii) of Jtac are kept orthogonal to (i), such
that it is easy to extend the library with new games.
"""
module Jtac

# Standard Libraries
using Random, Statistics, LinearAlgebra, Printf

# Pretty Printing
import ProgressMeter, Crayons

# Neural Networks
import AutoGrad, Knet
import Knet: identity, relu, elu, softmax, tanh, sigm, minibatch

export AutoGrad, Knet, 
       minibatch, identity, relu, elu, softmax, tanh, sigm

# Saving models and datasets
import BSON

# Optimizing the Bayesian ELO likelihood
import NLopt

# Utilities
include("util.jl")

# Games
include("game.jl")

export Game, Status, ActionIndex,
       status, current_player, legal_actions, apply_action!, 
       is_action_legal, representation, policy_length, random_playout, 
       augment, draw, is_over, random_turn!

# Models
include("feature.jl")
include("element.jl")
include("layer.jl")
include("model.jl")

export Feature, ConstantFeature
export Model, apply, 
       to_gpu, to_cpu, swap, on_gpu, training_model
export Pointwise, Dense, Conv, Deconv, Pool, Dropout, Batchnorm, 
       Chain, Stack,
       valid_insize, outsize, layers,
       @chain, @stack

# Model implementations
include("models/toy.jl")
include("models/neural.jl")
include("models/async.jl")

export DummyModel, RandomModel, RolloutModel,
       NeuralModel, Shallow, MLP, ShallowConv,
       Async

# MCTS, Players, and Bayesian ELO rankings
include("mc.jl")
include("player.jl")
include("belo.jl")

export RandomPlayer, MCTSPlayer, 
       IntuitionPlayer, HumanPlayer, 
       pvp, name, think, decide, turn!,
       playouts, ranking, print_ranking


# Training
include("dataset.jl")
include("loss.jl")
include("learning.jl")

export DataSet, augment, minibatch, 
       record_self, record_against,
       Loss, loss, caption,
       set_optimizer!, train_step!, train!,
       train_self!, train_against!, train_from!,
       with_contest

# Saving and loading games and datasets
include("io.jl")

export save_model, load_model 

# Game implementations
include("games/mnkgame.jl")
include("games/metatac.jl")
include("games/nim.jl")
include("games/morris.jl")

export TicTacToe, MNKGame, MetaTac, Nim, Morris

end # module Jtac
