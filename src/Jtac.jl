
# TODO: rewrite this
"""
Julia package that implements the Alpha Zero learning design in a modular
manner.

The package exports implementations of various two-player board games (like
tic-tac-toe), functions for creating neural networks that evaluate the states of
games, and functionality to generate datasets through selfplay via a Monte-Carlo
Tree Search player assisted by a neural model.
"""
module Jtac

const _version = v"0.2"

using Random,
      Statistics,
      LinearAlgebra

import NNlib

"""
Jtac serialization module.

Provides fast serialization and deserialization of basic struct types via the
msgpack format.
"""
module Pack
  import TranscodingStreams: TOKEN_END
  import CodecZstd: ZstdCompressorStream, ZstdDecompressorStream

  include("pack/pack.jl")
  include("pack/macro.jl")

  export @pack
end


"""
Jtac utility module.

Contains various utility functions for the rest of the library, ranging
from symmetry operations and named values to benchmarking tools.
"""
module Util
  # TODO: remove ProgressMeter!
  import ProgressMeter
  import NNlib
  using ..Jtac
  using ..Pack

  include("util/util.jl")

  export parallelforeach,
         showindented,
         stepper

  include("util/symmetry.jl")

  export applygroup

  export DihedralGroup,
         KleinFourGroup

  include("util/named.jl")

  export register!,
         isregistered,
         lookup,
         lookupname,
         resolve,
         NamedValueFormat

  module Bench
    using Statistics, Printf
    using ..Jtac

    # TODO: refactor / repair bench.jl
    # include("util/bench.jl")
  end # module Bench

  export Bench
end # module Util


"""
Jtac game module.

Defines the interface that any game type `G <: Game.AbstractGame` has
to implement. Also provides proof-of-concept implementations of the game
Tic-Tac-Toe ([`Game.TicTacToe`](@ref)), its generalization to general grids
([`Game.MNKGame`](@ref)), as well as the (much more interesting) variant
meta Tic-Tac-Toe ([`Game.MetaTac`](@ref)).
"""
module Game
  using Random, Statistics, LinearAlgebra

  using ..Jtac
  using ..Util
  import ..Pack
  import ..Pack: @pack

  include("game/game.jl")

  export AbstractGame, 
         Status,
         ActionIndex

  export status,
         instance,
         activeplayer,
         isactionlegal,
         isover,
         isaugmentable,
         legalactions,
         move!, 
         policylength,
         augment,
         randommove!,
         randomturn!,
         randommatch!, 
         randommatch, 
         array,
         branch,
         draw

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

end


"""
Jtac target module.    

Defines prediction targets that Jtac models can be trained on (see
[`Target.AbstractTarget`](@ref)).
"""
module Target

  using Random, Statistics, LinearAlgebra

  using ..Jtac
  using ..Util
  using ..Game
  import ..Pack
  import ..Pack: @pack

  include("target.jl")

  export AbstractTarget,
         DefaultValueTarget,
         DefaultPolicyTarget,
         DummyTarget

  export LabelContext

end

"""
Jtac model module.

In Jtac, models are responsible for game state evaluations. Given a game state,
models predict a scalar value and a policy vector to assess the current state
and the available options for action.

Models are the driving engines behind players (see `Player.AbstractPlayer`),
which live at a higher level of abstraction and which implement additional logic
like Monte-Carlo-Tree search (see [`Player.MCTSPlayer`](@ref)).

This module defines the interface for abstract Jtac models (see
[`Model.AbstractModel`](@ref)) and provides the following concrete model
implementations:
- [`Model.RolloutModel`](@ref): A model that always proposes a uniform policy \
  and a value obtained by simulating the game outcome via random actions. \
  If plugged into an [`MCTSPlayer`](@ref), this model leads to the classical
  rollout-based Monte-Carlo tree search algorithm.
- [`Model.NeuralModel`](@ref): A neural network based model. This model type \
  is special in at least two ways: First, it can be trained on recorded data \
  (see the module `Training` for more information). Second, it can also learn \
  to predict other targets than the value and policy for a game state.
- [`Model.AsyncModel`](@ref): Wrapper model that makes batched evaluation
  available to [`Model.NeuralModel`](@ref)s in asynchronous contexts.
- [`Model.AssistedModel`](@ref): Wrapper model that equipps a given model with
  an assistant (like an analytical solver for certain states of a game).
"""
module Model

  using Random, Statistics, LinearAlgebra
  import NNlib

  using ..Jtac
  using ..Util
  using ..Game
  using ..Target
  import ..Pack
  import ..Pack: @pack

  include("model/model.jl")

  export AbstractModel,
         Format,
         DefaultFormat

  export apply,
         assist,
         gametype,
         targets,
         targetnames,
         ntasks,
         isasync,
         basemodel,
         childmodel,
         trainingmodel,
         playingmodel,
         configure,
         adapt,
         save,
         load

  include("model/layer.jl")

  export Backend,
         DefaultBackend,
         Activation

  export Dense,
         Conv,
         Batchnorm,
         Chain,
         Residual

  export getbackend,
         isvalidinputsize,
         isvalidinput,
         outputsize,
         parameters,
         parametercount,
         layers,
         @chain,
         @residual

  include("model/dummy.jl")
  include("model/random.jl")
  include("model/rollout.jl")

  export DummyModel,
         RandomModel,
         RolloutModel

  include("model/neural.jl")
  include("model/async.jl")
  include("model/threaded.jl")
  include("model/caching.jl")
  include("model/assisted.jl")

  export NeuralModel,
         AsyncModel,
         CachingModel,
         AssistedModel

  """
  Predefined neural model architectures.

  Most relevant are [`Zoo.ZeroConv`](@ref) and [`Zoo.ZeroRes`](@ref), which
  are modeled after the convolutional and residual architectures of the Alpha
  Zero publications.
  """
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

"""
TODO: document this module    
"""
module Player

  using Random, Statistics, LinearAlgebra
  using Printf
  import NNlib: softmax

  using ..Jtac
  using ..Util
  using ..Game
  using ..Target
  using ..Model
  import ..Pack
  import ..Pack: @pack

  include("player/mcts.jl")
  include("player/player.jl")

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
         switchmodel

  include("player/elo.jl")    # outsource to Util or rank.jl?
  include("player/rank.jl")

  export Ranking
  export rank

end


"""
TODO: document this module 
"""
module Training

  using Random, Statistics, LinearAlgebra
  using Printf

  using ..Jtac
  using ..Util
  using ..Game
  using ..Target
  using ..Model
  using ..Player

  import ..Pack
  import ..Pack: @pack

  include("training/dataset.jl")

  export save, load

  export DataSet, DataCache, DataBatches

  include("training/record.jl")

  export record

  include("training/learn.jl")

  export LossFunction, LossContext

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
       Target,
       Training

using .Util, .Model, .Training

function __init__()
  register!(Activation, identity, :id, :identity)
  register!(Activation, NNlib.relu, :relu)
  register!(Activation, NNlib.selu, :selu)
  register!(Activation, NNlib.elu, :elu)
  register!(Activation, NNlib.tanh_fast, :tanh)
  register!(Activation, NNlib.sigmoid_fast, :sigmoid)
  register!(Activation, Activation(NNlib.softmax, broadcast = false), :softmax)

  register!(Backend, DefaultBackend{Array{Float32}}(), :default, :default32)
  register!(Backend, DefaultBackend{Array{Float16}}(), :default16)
  register!(Backend, DefaultBackend{Array{Float64}}(), :default64)

  register!(Format, DefaultFormat(), :jtm)

  register!(LossFunction, (x, y) -> sum(abs, x .- y), :sumabs)
  register!(LossFunction, (x, y) -> sum(abs2, x .- y), :sumabs2)
  register!(LossFunction, (x, y) -> -sum(y .* log.(x .+ 1f-7)), :crossentropy)
end


end # module Jtac
