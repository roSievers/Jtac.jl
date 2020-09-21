
"""
Module that implements the Alpha Zero learning design.

The module exports (i) implementations of various two-player board games, like
tic-tac-toe, (ii) functions for creating neural networks that act on the states
of games, and (iii) functionality to generate datasets via the selfplay of
a Monte-Carlo Tree Search player assisted by a neural model. Aspects (ii) and
(iii) of Jtac are kept orthogonal to (i), such that it is easy to extend the
library with new games.
"""
module Jtac

# -------- Packages ---------------------------------------------------------- #

using Random, Statistics, LinearAlgebra, Printf, Distributed

import ProgressMeter, Crayons, BSON, NLopt

import AutoGrad, Knet
import Knet: identity,
             relu,
             elu,
             softmax,
             tanh,
             sigm

export AutoGrad, Knet

export identity,
       relu,
       elu,
       softmax,
       tanh,
       sigm

# -------- Utilities --------------------------------------------------------- #

include("util.jl")

export prepare,
       branch

# -------- Games ------------------------------------------------------------- #

include("game.jl")

export Game, 
       Status,
       ActionIndex

export status,
       current_player,
       legal_actions,
       apply_action!, 
       is_action_legal,
       representation,
       policy_length,
       random_playout, 
       augment,
       draw,
       is_over,
       random_turn!

# -------- CPU/GPU Elements and NN Models ------------------------------------ #

include("feature.jl")
include("element.jl")
include("layer.jl")
include("model.jl")

export Feature,
       ConstantFeature

export Model

export apply,
       apply_features,
       to_gpu,
       to_cpu,
       swap,
       on_gpu,
       base_model,
       playing_model,
       training_model,
       gametype

export Pointwise,
       Dense,
       Conv,
       Deconv,
       Pool,
       Dropout,
       Batchnorm, 
       Chain,
       Stack

export valid_insize,
       outsize,
       layers,
       @chain,
       @stack,
       @residual

# -------- Specific models implementations ----------------------------------- #

include("models/toy.jl")
include("models/neural.jl")
include("models/async.jl")

export DummyModel,
       RandomModel,
       RolloutModel,
       NeuralModel,
       Shallow,
       MLP,
       ShallowConv,
       Async

# -------- MCTS, Players, and Bayesian ELO rankings -------------------------- #

include("mc.jl")
include("player.jl")
include("belo.jl")

export RandomPlayer,
       MCTSPlayer, 
       IntuitionPlayer,
       HumanPlayer

export pvp,
       name,
       think,
       decide,
       turn!,
       compete

export Ranking
export summary

# -------- Training ---------------------------------------------------------- #

include("dataset.jl")
include("loss.jl")
include("learning.jl")

export DataSet

export augment,
       minibatch, 
       record_self,
       record_against,
       Loss,
       loss,
       caption,
       set_optimizer!,
       train_step!,
       train!,
       train_self!,
       train_against!,
       train_from_model!,
       with_contest

# -------- Distributed creation of datasets ---------------------------------- #

include("distributed.jl")

export record_self_distributed,
       record_against_distributed,
       compete_distributed

# -------- Saving and loading games and datasets ----------------------------- #

include("io.jl")

export save_model,
       load_model,
       save_dataset,
       load_dataset

# -------- Game implementations ---------------------------------------------- #

include("games/mnkgame.jl")
include("games/metatac.jl")
include("games/nim.jl")
include("games/morris.jl")

export TicTacToe,
       MNKGame,
       MetaTac,
       Nim,
       Morris


end # module Jtac
