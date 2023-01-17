
"""
Program that makes a player play against itself. The produced datasets are
returned.

#### Config
* `gpu :: Int = -1`: GPU device used by the model
* `atype :: String = "knet"`: GPU array type
* `batchsize :: Int = 64`: Number of parallel evaluations in the model
* `ntasks :: Int = 2batchsize`: Number of async selfplays
* `spawn :: Bool = true`: Whether to spawn a new process for the model

#### Input
* `player`: Tuple of an `AbstractModel`, context information, and a
  selfplay configuration.

#### Output
* `data`: Tuple of a `DataSet`, corresponding to the states from one selfplay,
  together with the context in which the selfplay took place.
"""
mutable struct SelfplayProgram <: AbstractProgram
  state :: ProgramState
  config :: Dict{Symbol, Any}

  data :: Channel{Tuple{DataSet, Any}}
  player :: Channel{Tuple{AbstractModel, Any, Any}}
end

function SelfplayProgram(config, data = nothing)
  state = ProgramState()
  if isnothing(data)
    data = Channel{Tuple{DataSet, Any}}(100)
  end
  player = Channel{Tuple{AbstractModel, Any, Any}}(1)
  SelfplayProgram(state, config, data, player)
end

function init_storage(:: SelfplayProgram)
  player = Channel{MCTSPlayer}(1)
  ctx = Channel(1)
  config = Channel(1)
  counter = Ref(0)
  (; player, ctx, config, counter)
end

function on_stop(k :: SelfplayProgram, storage)
  close(k.player) # TODO: this should not be necessary...
  close(storage.player)
  close(storage.ctx)
  close(storage.config)
end

on_exit(k :: SelfplayProgram, storage) = on_stop(k, storage)

function check_config(k :: SelfplayProgram)
  batchsize = get(k.config, :batchsize, 64)
  ntasks = get(k.config, :ntasks, 2batchsize)
  @assert 1 <= batchsize <= 4096 "invalid batchsize $batchsize"
  @assert 1 <= ntasks <= 4096 "invalid value of :ntasks ($ntasks)"
end

function run(k :: SelfplayProgram, storage)
  batchsize = get(k.config, :batchsize, 64)
  ntasks = get(k.config, :ntasks, 2batchsize)
  @sync begin
    @logdebug k "initializing storage handling"
    @async handle_error(k) do 
      update_storage(k, storage)
    end
    @logdebug k "initializing $ntasks selfplay cycles"
    @async begin
      asyncmap(1:ntasks; ntasks) do i
        handle_error(k) do
          cycle_selfplays(k, storage, i)
        end
      end
    end
  end
end

function setdevice!(k :: SelfplayProgram)
  gpu = get(k.config, :gpu, -1) :: Integer
  if gpu >= 0
    @assert gpu < length(CUDA.devices())
    CUDA.device!(gpu)
    @log k "setting gpu device to $gpu"
    atype = get(k.config, :atype, "knet")
    Model.atype_gpu!(atype)
    @log k "setting gpu array type to '$atype'"
    true
  else
    false
  end
end

function build_player(k :: SelfplayProgram, model, config)

  # some selfplay-related options are externally provided to the program
  power = get(config, :power, 250)
  temperature = get(config, :temperature, 1.)
  exploration = get(config, :exploration, 1.41)
  dilution = get(config, :dilution, 0.0)

  @logdebug k "player option :power = $power"
  @logdebug k "player option :temperature = $temperature"
  @logdebug k "player option :exploration = $exploration"
  @logdebug k "player option :dilution = $dilution"

  if model isa NeuralModel
    # gpu and batchsize-options are decided by the program config
    gpu = setdevice!(k)
    spawn = get(k.config, :spawn, true)
    batchsize = get(k.config, :batchsize, 64)
    buffersize = get(k.config, :ntasks, 2batchsize)

    @logdebug k "player option :spawn = $spawn"
    @logdebug k "player option :batchsize = $batchsize"
    @logdebug k "player option :buffersize = $buffersize"

    if gpu
      model = Model.to_gpu(model)
    end
    model = Async(model; batchsize, buffersize, spawn, dynamic = false)
  end

  MCTSPlayer(model; power, temperature, exploration, dilution)
end

function update_storage(k :: SelfplayProgram, storage)
  player = player_old = nothing
  while getstatus(k) == :running
    model, ctx, config = @take_or! k.player break
    @log k "received new model and selfplay config"
    if !isnothing(player_old)
      Model.dynamic_mode!(player_old.model, true)
      @logdebug k "enabled dynamic mode for previous async model"
    end
    player = build_player(k, model, config)
    modify!(storage.player, player)
    modify!(storage.ctx, ctx)
    modify!(storage.config, config)
    player_old = player
  end
  if !isnothing(player)
    # TODO: this seems to slow down the player greatly as it begins
    # to use effective batchsize 1...
    # For this reason, we currently interrupt the selfplays even
    # when calling :stop on the program
    Model.dynamic_mode!(player.model, true)
    @logdebug k "enabled dynamic mode for current async model"
  end
  @logdebug k "returning"
end

function cycle_selfplays(k :: SelfplayProgram, storage, i)
  player = @fetch_or storage.player return
  G = Model.gametype(player)

  while getstatus(k) == :running && isopen(k.data)
    ctx = @fetch_or storage.ctx break
    config = @fetch_or storage.config break

    min = get(config, :branch_step_min, 1)
    max = get(config, :branch_step_max, 5)
    prob = get(config, :branch_probability, 0.0)
    randomize = get(config, :randomize_instance, 0.0)
    augment = get(config, :augment, false)

#    @logdebug k "selfplay option :branch_step_min = $min"
#    @logdebug k "selfplay option :branch_step_max = $max"
#    @logdebug k "selfplay option :branch_probability = $prob"
#    @logdebug k "selfplay option :randomize_instance = $random"
#    @logdebug k "selfplay option :augment = $augment"

    instance = () -> Game.instance(G; randomize)
    branch = game -> Game.branch(game; prob, steps = min:max)

    time = @elapsed begin
      # updates of storage.player immediately affect selfplays :)
      ds = Player.record( storage.player, 1
                        ; merge = true
                        , branch
                        , instance
                        , augment
                        , callback_move = (_) -> checkpoint(k) )

      put_maybe!(k, k.data, (ds, ctx))
    end

    count = storage.counter[]
    @log k "dataset $count generated in $time seconds (task $i)"
    storage.counter[] += 1
    checkpoint(k)
  end

  @logdebug k "returning (task $i)"
end 

