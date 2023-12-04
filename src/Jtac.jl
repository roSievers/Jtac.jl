
"""
Julia package that implements the Alpha Zero learning paradigm for two-player
boardgames in a modular manner. It exports the following components:

* [`Pack`](@ref): Generic serialization module on top of the msgpack format.
* [`Game`](@ref): Module that defines interfaces for two-player boardgames and \
  provides  reference implementations for tic-tac-toe and meta tic-tac-toe.
* [`Model`](@ref): Module that defines interfaces for value and policy predictors that \
  operate on game states.
* [`Player`](@ref): Module that defines interfaces for and implements boardgame agents \
  on top of models.
* [`Target`](@ref): Module that adds support for model predictions other than the \
  value and policy of a given game state.
* [`Training`](@ref): Module that implements the generation of and training on labeled \
  data that is created during matches between players.
"""
module Jtac

const _version = v"0.2"

using Random,
      Statistics,
      LinearAlgebra

import NNlib

"""
Generic serialization module based on the msgpack format.
"""
module Pack
  import TranscodingStreams: TOKEN_END

  include("pack/pack.jl")
  include("pack/macro.jl")

  export @pack
end


"""
Jtac utility module.

Contains utility functions for the library. For example, implements symmetry
operations on matrices, named values (used for activation names or neural
network backends), and various benchmarking tools.
"""
module Util
  # TODO: remove ProgressMeter!
  import ProgressMeter
  import NNlib
  using ..Jtac
  using ..Pack

  include("util/util.jl")

  export pforeach,
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
         resolve

  export NamedValueFormat

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

Defines the interface that any game type `G <: Game.AbstractGame` must implement
to be supported by Jtac.

Also provides reference implementations of the game Tic-Tac-Toe
([`Game.TicTacToe`](@ref)), its generalization to general grids
([`Game.MNKGame`](@ref)), as well as the more complex variant meta Tic-Tac-Toe
([`Game.MetaTac`](@ref)).
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
         visualize

  include("game/mnkgame.jl")
  include("game/metatac.jl")
  # include("game/nim.jl")
  # include("game/nim2.jl")
  # include("game/morris.jl")

  export TicTacToe,
         MNKGame,
         MetaTac,
         Nim,
         Nim2,
         Morris

end


"""
Jtac target module.

Defines prediction targets that Jtac models can be trained for. See
[`Target.AbstractTarget`](@ref) for more information.
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

Models are responsible for game state evaluations. Given a game state, models
predict a scalar value and a policy vector to assess the quality of the current
state and the available options for action.

This module defines the interface for abstract Jtac models (see
[`Model.AbstractModel`](@ref)) and provides the following concrete model
implementations:
- [`Model.RolloutModel`](@ref): A model that always proposes a uniform policy \
  and a value obtained by simulating the game outcome via random actions. \
  If plugged into an [`MCTSPlayer`](@ref), this model leads to the classical
  rollout-based Monte-Carlo tree search algorithm.
- [`Model.NeuralModel`](@ref): A neural network based model. This model type \
  is special in two ways: it can be trained on recorded data \
  (see the module `Training` for more information), and it can also learn \
  to predict other targets than the value and policy for a game state.
- [`Model.AsyncModel`](@ref): Wrapper model that makes batched evaluation
  available to [`Model.NeuralModel`](@ref)s in asynchronous contexts.
- [`Model.AssistedModel`](@ref): Wrapper model that equipps a given model with
  an assistant (like an analytical solver for certain states of a game).

Models are the "intutitive brain" of players ([`Player.AbstractPlayer`](@ref)),
which live at a higher level of abstraction and can implement additional logic,
like Monte-Carlo tree search in case of the [`Player.MCTSPlayer`](@ref).
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
  follow the convolutional and residual architectures of the Alpha Zero
  publications.
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
Jtac player module

Players, subtypes of [`Player.AbstractPlayer`](@ref), are agents in a boardgame.
This module defines an interface for players and provides several player
implementations.

The most important players are [`Player.IntuitionPlayer`](@ref), which lets
an [`Model.AbstractModel`](@ref) directly decide on the policy, and the
[`Player.MCTSPlayer`](@ref), which combines model "intuition" with a classical
Monte-Carlo tree search (inspired by the Alpha Zero paradigm).
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
Jtac training module.

Provides tools to generate prediction labels via observing matches between
[`Player.AbstractPlayer`](@ref)s, and to then train [`Model.NeuralModel`](@ref)s
on the recorded data.
"""
module Training

  using Random, Statistics, LinearAlgebra
  using Printf

  import CodecZstd: ZstdCompressorStream, ZstdDecompressorStream

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
         setup,
         step!,
         train!,
         train_contest!

end # module Training


export Pack,
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
