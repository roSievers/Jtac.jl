
"""
Julia module that implements the Alpha Zero learning design.

The module exports implementations of various two-player board games (like
tic-tac-toe), functions for creating neural networks that evaluate the states of
games, and functionality to generate datasets through selfplay via a Monte-Carlo
Tree Search player assisted by a neural model.
"""
module Jtac

const _version = v"0.1"

# -------- Packages ---------------------------------------------------------- #

using Random, Statistics, LinearAlgebra

import AutoGrad, Knet, CUDA
import Knet: identity,
             relu,
             elu,
             softmax,
             tanh,
             sigm

export Knet, CUDA, AutoGrad

export identity,
       relu,
       elu,
       softmax,
       tanh,
       sigm

# -------- MsgPack Configuration --------------------------------------------- #

import MsgPack
include("msgpack.jl")

# -------- Utilities --------------------------------------------------------- #

module Util

  using Distributed
  import ProgressMeter

  using ..Jtac

  include("util.jl")

  export one_hot,
         choose_index,
         apply_dihedral_group,
         apply_klein_four_group,
         prepare,
         branch,
         stepper

end # module Util

# -------- Games ------------------------------------------------------------- #

module Game

  using Random, Statistics, LinearAlgebra
  import MsgPack

  using ..Jtac
  using ..Util

  include("game.jl")

  export AbstractGame, 
         Status,
         ActionIndex

  export status,
         current_player,
         legal_actions,
         apply_action!, 
         apply_actions!,
         is_action_legal,
         array,
         policy_length,
         random_playout, 
         augment,
         draw,
         is_over,
         random_turn!,
         random_turns!,
         instance


  # -------- Game implementations ---------------------------------------------- #

  include("games/mnkgame.jl")
  include("games/metatac.jl")
  include("games/nim.jl")
  include("games/nim2.jl")
  include("games/morris.jl")

  export TicTacToe,
         MNKGame,
         MetaTac,
         Nim,
         Nim2,
         Morris


end #module Game

# -------- CPU/GPU Elements and NN Models ------------------------------------ #

module Model

  using Random, Statistics, LinearAlgebra
  using CUDA
  import MsgPack

  using ..Jtac
  using ..Util
  using ..Game

  include("feature.jl")
  include("element.jl")
  include("layer.jl")
  include("model.jl")

  export Feature,
         ConstantFeature

  export AbstractModel

  export features,
         feature_length,
         feature_name,
         feature_compatibility,
         apply,
         apply_features,
         to_gpu,
         to_cpu,
         swap,
         on_gpu,
         tune,
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
         Stack,
         Residual

  export valid_insize,
         outsize,
         layers,
         @chain,
         @stack,
         @residual

  # -------- Fundamental model implementations ------------------------------- #

  include("models/basic.jl")
  include("models/neural.jl")
  include("models/async.jl")
  include("models/caching.jl")

  export AbstractModel,
         DummyModel,
         RandomModel,
         RolloutModel,
         NeuralModel,
         Async,
         Caching

  # -------- Predefined NeuralModel architectures ---------------------------- #

  module Zoo

    using ...Jtac
    using ...Game
    using ..Model

    include("models/zoo.jl")

    export Shallow,
           MLP,
           ShallowConv,
           ZeroConv,
           ZeroRes

  end # module Zoo

  # -------- Saving and loading models --------------------------------------- #

  include("modelio.jl")

  export save,
         load

end # module Model


# -------- MCTS, Player, and ML ELO rankings -------------------------------- #

module Player

  using Random, Statistics, LinearAlgebra
  using Printf, Distributed
  import CUDA

  using ..Jtac
  using ..Util
  using ..Game
  using ..Model

  include("mc.jl")
  include("player.jl")
  include("rank.jl")
  include("distributed.jl")

  export AbstractPlayer,
         RandomPlayer,
         MCTSPlayer, 
         IntuitionPlayer,
         HumanPlayer

  export pvp,
         name,
         think,
         decide,
         turn!,
         compete,
         switch_model

  export Ranking

end # module Player

# -------- Datasets ---------------------------------------------------------- #

module Data

  using Random, Statistics, LinearAlgebra
  import MsgPack, TranscodingStreams, CodecZstd

  using ..Jtac
  using ..Util
  using ..Game
  using ..Model
  using ..Player

  include("dataset.jl")
  include("pool.jl")

  export Dataset, Datacache, Batches, Pool

  export save,
         load,
         augment,
         record_self,
         record_against

end # Data

# -------- Training ---------------------------------------------------------- #

module Training

  using Random, Statistics, LinearAlgebra
  using Printf, Distributed

  using ..Jtac
  using ..Util
  using ..Game
  using ..Model
  using ..Player
  using ..Data

  include("loss.jl")
  include("learning.jl")

  export Dataset, Loss

  export loss,
         caption,
         set_optimizer!,
         train_step!,
         train!,
         train_self!,
         train_against!,
         train_from_model!,
         with_contest

end # module Training

module Bench

  include("bench.jl")

end

export Util,
       Bench,
       Game,
       Model,
       Player,
       Data,
       Training

end # module Jtac
