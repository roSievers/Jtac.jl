
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


# -------- Utilities --------------------------------------------------------- #

module Util

  using Distributed
  import ProgressMeter

  using ..Jtac

  include("util/util.jl")


  export apply_dihedral_group,
         apply_klein_four_group,
         prepare,
         branch,
         stepper


  module Bench

    using Distributed
    using Statistics
    using Printf

    using ..Jtac

    include("util/bench.jl")

  end # module Bench

  export Bench

end # module Util

# -------- Serialization ----------------------------------------------------- #

module Pack

  import MsgPack

  using ..Jtac
  import ..Util

  include("pack.jl")

end

# -------- Games ------------------------------------------------------------- #

module Game

  using Random, Statistics, LinearAlgebra
  import MsgPack

  using ..Jtac
  using ..Util
  import ..Pack

  include("game/game.jl")

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
         is_augmentable,
         random_turn!,
         random_turns!,
         instance


  # -------- Game implementations ---------------------------------------------- #

  include("game/mnkgame.jl")
  include("game/metatac.jl")
  include("game/nim.jl")
  include("game/nim2.jl")
  include("game/morris.jl")

  export TicTacToe,
         MNKGame,
         MetaTac,
         Nim,
         Nim2,
         Morris


end #module Game

# -------- Training targets -------------------------------------------------- #

module Target

  using Random, Statistics, LinearAlgebra
  using CUDA

  using ..Jtac
  using ..Util
  using ..Game
  import ..Pack

  include("target.jl")

  export AbstractTarget,
         PredictionTarget,
         RegularizationTarget

  export ValueTarget,
         PolicyTarget,
         DummyTarget,
         L1Reg,
         L2Reg

  export targets,
         adapt

end # module Target

# -------- CPU/GPU Elements and NN Models ------------------------------------ #

module Model

  using Random, Statistics, LinearAlgebra
  using CUDA
  import MsgPack

  using ..Jtac
  using ..Util
  using ..Game
  using ..Target
  import ..Pack

  include("model/element.jl")
  include("model/layer.jl")
  include("model/model.jl")

  export AbstractModel

  export targets,
         opt_targets,
         apply,
         to_gpu,
         to_cpu,
         swap,
         on_gpu,
         tune,
         is_async,
         base_model,
         playing_model,
         training_model,
         gametype

  export LayerWeight,
         LayerActivation,
         Layer,
         PrimitiveLayer,
         CompositeLayer,
         Pointwise,
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

  include("model/basic.jl")
  include("model/neural.jl")
  include("model/async.jl")
  include("model/caching.jl")

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

    include("model/zoo.jl")

    export Shallow,
           MLP,
           ShallowConv,
           ZeroConv,
           ZeroRes

  end # module Zoo

end # module Model

# -------- Datasets ---------------------------------------------------------- #

module Data

  using Random, Statistics, LinearAlgebra, Distributed
  import MsgPack, TranscodingStreams, CodecZstd

  using ..Jtac
  using ..Util
  using ..Game
  using ..Model
  using ..Target
  import ..Pack

  include("data/dataset.jl")
  include("data/cache.jl")
  include("data/batch.jl")
  include("data/pool.jl")

  export DataSet, Cache, Batches, Pool

  export save,
         load,
         augment

end # Data


# -------- MCTS, Player, and ML ELO rankings -------------------------------- #

module Player

  using Random, Statistics, LinearAlgebra
  using Printf, Distributed
  import CUDA
  import ThreadPools # employ background threads

  using ..Jtac
  using ..Util
  using ..Game
  using ..Target
  using ..Model
  using ..Data
  import ..Pack

  include("player/mc.jl")
  include("player/player.jl")
  include("player/elo.jl")    # outsource to Util or rank.jl?
  include("player/rank.jl")
  include("player/record.jl")

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
         switch_model,
         record,
         record_against,
         record_model

  export Ranking

end # module Player


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
  using ..Target

  include("training.jl")

  export loss,
         set_optimizer!,
         train_step!,
         train!,
         train_contest!

end # module Training


export Util,
       Pack,
       Game,
       Model,
       Player,
       Data,
       Target,
       Training

end # module Jtac
