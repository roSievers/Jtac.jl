
"""
Program that consumes data sets to train a neural model.

#### Input
* `players`: Tuple of an `AbstractPlayer` and context information, which
  may include the fields `:branch_step_[min|max]`, `branch_prob`, and
  `randomize_instance`.

#### Output
* `data`: Tuple of a `DataSet`, corresponding to the states from one selfplay,
  together with the context in which the selfplay took place.

#### Config
* `gpu :: Int = -1`: GPU device used by the model
* `async :: Int = 50`: Number of async selfplays
"""
mutable struct TrainingProgram{G <: AbstractGame} <: AbstractProgram
  state :: ProgramState
  config :: Dict{Symbol, Any}

  config_update :: Channel{Dict{Symbol, Any}}
  data :: Channel{Tuple{DataSet, Any}}
  model :: Channel{Tuple{NeuralModel{G}, Any}}
  steps :: Channel{NamedTuple}

  _model ::  NeuralModel{G}
  _pool :: NamedTuple{(:train, :test), Tuple{Pool{G}, Pool{G}}}
end

# TODO: loss targets and target weights!

function TrainingProgram(_model :: NeuralModel{G, false}, config) where {G}
  state = ProgramState()
  config_update = Channel{Dict{Symbol, Any}}(1) # TO PROGRAM
  data = Channel{Tuple{DataSet, Any}}(100)      # TO PROGRAM
  model = Channel{Tuple{NeuralModel{G}, Any}}(10)  # TO HOST
  steps = Channel{NamedTuple}(100)              # TO HOST

  meta = (; generation = Int, usage = Int, age = Int)
  trainpool = Pool(G, meta, targets = Target.targets(_model))
  testpool = Pool(G, meta, targets = Target.targets(_model))

  _pool = (; train = trainpool, test = testpool)

  TrainingProgram{G}(state, config, config_update, data, model, steps, _model, _pool)
end

function init_storage(:: TrainingProgram{G}) where {G}
  lock_pool = Threads.Condition()
  lock_model = ReentrantLock() 
  lock_config = ReentrantLock()
  generation = Channel{Int}(1)
  gpu = Channel{Bool}(1)

  put!(generation, 1)
  put!(gpu, false)

  (; lock_pool, lock_model, lock_config, generation, gpu )
end

function on_stop(k :: TrainingProgram, storage)
  close(k.data)
  close(k.config_update)
  locked_notify(storage.lock_pool)
end

on_exit(k :: TrainingProgram, storage) = on_stop(k, storage)

function run(k :: TrainingProgram, storage)
  @sync begin
    @logdebug k "initializing pool updates"
    @async handle_error(k) do
      update_pool(k, storage)
    end
    @logdebug k "initializing config updates"
    @async handle_error(k) do
      update_config(k, storage)
    end
    @logdebug k "initializing model training loop"
    @async handle_error(k) do
      train_model(k, storage)
    end
  end
end

function update_pool(k :: TrainingProgram, storage)
  configure_pool!(k, storage)
  while true
    ds, ctx = @take_or! k.data break

    # info on the data
    len = length(ds)
    gen = ctx.generation
    age = fetch(storage.generation) - gen

    test = test_pool_starves(k, storage)

    Base.lock(storage.lock_pool) do
      if test
        append!(k._pool.test, ds, (; generation = gen, usage = 0, age))
        log(k, 2, :update_pool, "new dataset for test pool (length $len, gen $gen)")
        Data.trim!(k._pool.test)
      else
        append!(k._pool.train, ds, (; generation = gen, usage = 0, age))
        log(k, 2, :update_pool, "new dataset for train pool (length $len, gen $gen)")
        Data.trim!(k._pool.train)
      end
      notify(storage.lock_pool)
    end
  end
end

function getconfig(k :: TrainingProgram, storage, key, default)
  Base.lock(storage.lock_config) do
    get(k.config, key, default)
  end
end

function test_pool_starves(k, storage)
  size_min_test = getconfig(k, storage, :size_min_test, 1_000)
  if length(k._pool.test) < size_min_test
    true
  else
    otest = Data.occupation(k._pool.test)
    otrain = Data.occupation(k._pool.train)
    otrain > otest
  end
end

function update_config(k :: TrainingProgram, storage)
  while true
    config = @take_or! k.config_update break
    Base.lock(storage.lock_config) do
      merge!(k.config, config)
    end
    # reconfigure pool
    configure_pool!(k, storage)

    # maybe reconfigure optimizer, too
    if intersect(keys(config), [:optimizer, :lr]) |> !isempty
      configure_optimizer!(k, storage)
    end
  end
end

function configure_pool!(k, storage)
  size_max = getconfig(k, storage, :size_max, 1_000_000)
  size_max_test = getconfig(k, storage, :size_max_test, 10_000)

  keep_iterations = getconfig(k, storage, :keep_iterations, 10)
  keep_generations = getconfig(k, storage, :keep_generations, 3)

  criterion = meta -> begin
    age_weight = (keep_generations - meta.age) / keep_generations
    use_weight = (keep_iterations - meta.usage) / keep_iterations
    max(0., age_weight)^2 * max(0., use_weight)
  end

  lock(storage.lock_pool) do
    Data.capacity!(k._pool.train, size_max)
    Data.capacity!(k._pool.test, size_max_test)

    Data.criterion!(k._pool.train, criterion)
    Data.criterion!(k._pool.test, criterion)
  end
end

function configure_optimizer!(k :: TrainingProgram, storage)

  # get optimizer and options
  opt, kwargs = Base.lock(storage.lock_config) do
    opt = get(k.config, :optimizer, "momentum")
    if opt == "sgd"
      lr = get(k.config, :lr, 0.1)
      Knet.SGD, (; lr)
    elseif opt == "momentum"
      lr = get(k.config, :lr, 0.05)
      gamma = get(k.config, :gamma, 0.95)
      Knet.Momentum, (; lr, gamma)
    elseif opt == "adam"
      lr = get(k.config, :lr, 0.001)
      beta1 = get(k.config, :beta1, 0.9)
      beta2 = get(k.config, :beta2, 0.999)
      Knet.Adam, (; lr, gamma, beta1, beta2)
    elseif opt == "rmsprop"
      lr = get(k.config, :lr, 0.01)
      rho = get(k.config, :rho, 0.9)
      Knet.Rmsprop, (; lr, rho)
    else
      @logwarn k "ignoring unknown optimizer '$opt'"
      nothing, nothing
    end
  end

  # apply optimizer
  if !isnothing(opt)
    Base.lock(storage.lock_model) do
      for param in Knet.params(k._model)
        param.opt = opt(; kwargs...)
      end
    end
    @log k "configured optimizer: $opt($kwargs)"
  end
end

function train_model(k :: TrainingProgram, storage)
  gpu = setdevice!(k)
  modify!(storage.gpu, gpu)

  Base.lock(storage.lock_model) do
    k._model = Model.tune(copy(k._model); gpu)
  end
  configure_optimizer!(k, storage)

  generation = 1
  while true

    # train for one generation
    train_model_generation(k, storage)
    @log k "generation $generation finished"

    generation += 1
    modify!(storage.generation, generation)

    # send model to host
    Base.lock(storage.lock_model) do
      model = copy(Model.to_cpu(k._model))
      ctx = (; generation)
      put!(k.model, (model, ctx)) # this put! is too important, don't use put_maybe!
    end
    @logdebug k "current model sent to host"

    # update pool age metadata
    Base.lock(storage.lock_pool) do
      Data.update!(k._pool.test) do meta
        (; meta..., age = generation - meta.generation)
      end
      Data.update!(k._pool.train) do meta
        (; meta..., age = generation - meta.generation)
      end
      Data.trim!(k._pool.train)
      Data.trim!(k._pool.test)
    end
    @logdebug k "pool metadata updated"

    # check if we should pause or exit
    checkpoint(k, exit_on_stop = true)
  end
end

function train_model_generation(k :: TrainingProgram, storage)
  for step in 1:getconfig(k, storage, :gensize, 100)

    # get training data for one training step
    data, testdata, stepsize, batchsize = Base.lock(storage.lock_pool) do

      l = length(k._pool.train)
      stepsize = getconfig(k, storage, :stepsize, 10)
      batchsize = getconfig(k, storage, :batchsize, 512)
      size_pool_min = getconfig(k, storage, :size_min, 0)

      while l < stepsize * batchsize || l < size_pool_min 
        log(k, 2, :train_model_generation, "train pool not large enough (size $l)")
        wait(storage.lock_pool)
        checkpoint(k, exit_on_stop = true)
        l = length(k._pool.train)
        stepsize = getconfig(k, storage, :stepsize, 10)
        batchsize = getconfig(k, storage, :batchsize, 512)
      end

      data, sel = Data.sample(k._pool.train, stepsize * batchsize)
      Data.update!(k._pool.train, sel) do meta
          (; meta..., usage = meta.usage + 1)
      end
      Data.trim!(k._pool.train, sel)

      testdata = k._pool.test.data[1:end]
      data, testdata, stepsize, batchsize
    end

    @logdebug k "training data sampled (stepsize $stepsize, batchsize $batchsize)"

    # divide the data in batches
    gpu = fetch(storage.gpu)
    batches = Batches(data, batchsize, shuffle = false, gpu = gpu, store_on_gpu = false)

    # iterate batches
    time = @elapsed for cache in batches
      checkpoint(k)
      Base.lock(storage.lock_model) do
        Training.train_step!(k._model, cache)
      end
    end

    checkpoint(k)
    # TODO: this probably requires an unacceptable amount of compute power, we
    # have to report losses less frequently or with smaller data sizes
    loss = Base.lock(storage.lock_model) do
      train = Training.loss(k._model, data)
      test = Training.loss(k._model, testdata)
      (; train, test)
    end

    # stats about the pool
    pool = Base.lock(storage.lock_pool) do
      age = mean(m -> m.age, k._pool.train.meta)
      usage = mean(m -> m.usage, k._pool.train.meta)
      occupation = Data.occupation(k._pool.train)
      capacity = Data.capacity(k._pool.train)
      (; age, usage, occupation, capacity)
    end

    generation = fetch(storage.generation)
    put_maybe!(k.steps, (; time, step, generation, loss, pool))

    @logdebug k "step results sent to host (step $step, generation $generation)"

    checkpoint(k, exit_on_stop = true)
  end
end
