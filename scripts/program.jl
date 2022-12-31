
module Log

export log, @log, @logwarn, @logerror, @logdebug

# ------- Logging ------------------------------------------------------------ #

function log end

macro log(k, str)
  quote
    _self_ = Base.nameof(var"#self#")
    log($k, 2, _self_, $str)
  end |> esc
end

macro logerror(k, str)
  quote
    _self_ = Base.nameof(var"#self#")
    log($k, 0, _self_, $str)
  end |> esc
end

macro logwarn(k, str)
  quote
    _self_ = Base.nameof(var"#self#")
    log($k, 1, _self_, $str)
  end |> esc
end

macro logdebug(k, str)
  quote
    _self_ = Base.nameof(var"#self#")
    log($k, 3, _self_, $str)
  end |> esc
end

end # module Log


module Program

using Statistics
using ThreadPools
using Jtac

import .Game: AbstractGame
import .Data: DataSet, Pool, Batches
import .Model: NeuralModel
import .Player: AbstractPlayer

import ..Log: log, @log, @logwarn, @logerror, @logdebug

"""
Exception that is used to exit programs in case of interruption by the host.
"""
struct InterruptProgram <: Exception end

"""
Exception that is used to exit programs in case of stop signals by the host.
"""
struct StopProgram <: Exception end

"""
    AbstractProgram

A program is an program intended to be executed in its own thread by some host.
It can be paused, stopped, and interrupted.

#### Api

On the application level, the host controls the basic behavior of the program via
the functions

* `run`
* `pause`
* `resume
* `stop`
* `interrupt`

#### Implementation

Some guidelines when implementing programs.

0) Implementing a program means defining a concrete subtype of `AbstractProgram`
   plus the definition of several methods:

   * `init_storage`
   * `on_stop`
   * `on_exit`
   * `run`

   In the following, the term 'program code' refers to these four methods. Any
   concrete subtype must additionally hold the field `state`, which is of type
   `ProgramState`. The state is used for logging, basic logic, and error
   handling.

1) Program code should distinguish between `host_to_program` and `program_to_host` channels.

   `host_to_program`:

   The program may call `fetch` and `take!`
   The program takes ownership of the channel and the internal logic closes
   it on shutdown. For this reason, any attempt to `take!` or `fetch` must be
   guarded in the `run(program)` portion of the Program. Convenience macros like
   `@take_or!` may be used for this purpose.
   The channel should be closed in the `on_stop` and `on_exit` routines
   The channel must not be closed by the host.
   The program must not call `put!` on these channels, since this can easily
   lead to blocking.

   `program_to_host`:

   The program may call `put!`. 
   The host has ownership over the channel and the program is forbidden to
   close it.
   If the host cannot keep up or errors, putting should not block the program.
   Helper functions like `put_maybe!` can be used in this case
   A closed channel `program_to_host` channel should usually be treated like
   a `stop(program)` signal, but depending on the circumstances, it does not
   have to be.

2) Stopping a program does not imply that it immediately has to return. If
   if it is currently generating a batch of data, it may decide to finish and
   `put!` the data into a `program_to_host` channel before it shuts down. Note
   that `program_to_host` channel will remain open even after stopping, while
   `host_to_program` channels are supposed to close immediately (via `on_stop`
   and `on_exit`).

3) When implementing a program, pausing is enforced in the function
   `checkpoint`. Therefore, even after pausing, the program may be active as
   long as `checkpoint` has not been reached.
"""
abstract type AbstractProgram end

"""
    log(program, level, fn, str)

Put the message `str` into the `log` channel of the program state. Also store
`level` information and the name `fn` of the function where the log originated.
This channel can be used by the host in order to display logged events in the
program. The loglevels are interpreted as follows:

    0 => :error
    1 => :warning
    2 => :info
    3 => :debug

Since this function aims to be non-blocking, the message is discarded if the log
channel is full.
"""
function log(k, level :: Int, fn, msg :: String)
  time = Base.time()
  s = getstate(k)
  put_maybe!(s.log, (; time, level, fn, msg))
end

function Base.show(io :: IO, k :: K) where {K <: AbstractProgram}
  sym = Base.nameof(K)
  status = getstatus(k)
  print(io, "$sym(:$status)")
end

function Base.show(io :: IO, ::MIME"text/plain", k :: AbstractProgram)
  Base.show(io, k)
end

"""
Type to keep track of low-level implementation details that all programs of type
`AbstractProgram` share. It possesses the following fields:

* `status :: Channel{Symbol}`: A one-element channel that contains the status of
  the program. Valid statuses are `:inactive`, `:running`, `:paused`, `:stopped`,
  `:interrupted`, and `:failed`.

* `signal :: Channel{Symbol}`: A multi-element channel that is used for basic
  communication with the host. Valid signals are `:stop`, `:pause`, `:resume`,
  and `:interrupt`.

* `log :: Channel{Tuple{Float64,String}}`: A multi-element channel that is used
  for reporting logging information back to the host.

* `pause :: Condition`: Condition that is used to implement pausing.

* `err :: Union{Nothing, Exception}`: The possible exception that caused the
  program to fail.
"""
mutable struct ProgramState
  status :: Channel{Symbol}
  signal :: Channel{Symbol}
  log :: Channel{NamedTuple}
  pause :: Threads.Condition
  err :: Union{Nothing, Exception}
end

function ProgramState()
  status = Channel{Symbol}(1)
  signal = Channel{Symbol}(100)
  log = Channel{NamedTuple}(10000)
  pause = Threads.Condition()
  put!(status, :inactive)
  ProgramState(status, signal, log, pause, nothing)
end

# ------- Host side interface ------------------------------------------------ #

"""
    start(program)

Start an inactive `program`. Executes the function `run(program, storage)` with
additional handling of logic and errors. The local variable storage `storage` of
the program is created by `init_storage(program)`.
"""
function start(k :: AbstractProgram)
  @assert getstatus(k) == :inactive "Program has been started before"
  @log k "initializing program on thread $(Threads.threadid())"
  s = getstate(k)
  try
    modify_status!(k, :running)
    @logdebug k "initializing storage"
    storage = init_storage(k)
    @sync begin
      @logdebug k "initializing signal handling"
      on_stop() = Program.on_stop(k, storage)
      on_exit() = Program.on_exit(k, storage)
      @async handle_signals(k; on_stop, on_exit)
      @logdebug k "executing program"
      handle_error(k) do
        run(k, storage)
      end
      close(s.signal)
    end
  catch err
    s.err = err
    @logerror k "encountered error: $err"
    modify_status!(k, :failed)
  finally
    getstatus(k) == :stopping && modify_status!(k, :stopped)
    @log k "program exited with status :$(getstatus(k))"
    close(s.signal)
    close(s.log)
  end
end

"""
    pause(program)

Ask an active `program` to pause computation.
"""
function pause(k :: AbstractProgram)
  s = getstate(k)
  if fetch(s.status) != :inactive
    put!(s.signal, :pause)
  end
end

"""
    resume(program)

Ask a paused `program` to resume computation
"""
function resume(k :: AbstractProgram)
  s = getstate(k)
  if fetch(s.status) != :inactive
    put!(s.signal, :resume)
  end
end

"""
    stop(program)

Ask an active `program` to gracefully stop computation. Ongoing computations are
finished.
"""
function stop(k :: AbstractProgram)
  s = getstate(k)
  if fetch(s.status) != :inactive
    put!(s.signal, :stop)
  end
end

"""
    interrupt(program)

Interrupt an active `program`. Partial results of ongoing computations are
discarded.
"""
function interrupt(k :: AbstractProgram)
  s = getstate(k)
  if fetch(s.status) != :inactive
    put!(s.signal, :interrupt)
  end
end


"""
    getstatus(program)

Obtain the current status of the program. Can be one of

    [:inactive, :running, :paused, :stopped, :interrupted, :failed]

In case the status is :failed, the causing error can be accessed via
`geterror(program)`.
"""
function getstatus(k :: AbstractProgram)
  s = getstate(k)
  fetch(s.status) # this should never actually block
end

"""
    geterror(program)

If the `program` failed, return the causing exception. Otherwise, return
`nothing`.
"""
function geterror(k :: AbstractProgram)
  if getstatus(k) == :failed
    s = getstate(k)
    s.err
  end
end

"""
    getstate(program)

Get the state of a program.
"""
getstate(k :: AbstractProgram) = k.state

"""
    getlog(program)

Get all entries currently logged by the program. Note that this removes the
messages from the `log` channel of `program.state`.
"""
function getlog(program)
  s = getstate(program)
  logs = Vector{NamedTuple}()
  while isready(s.log)
    push!(logs, take!(s.log))
  end
  logs
end


# ------- Program-side interface --------------------------------------------- #

"""
    modify!(channel, value)

Exchange the current value in `channel` by `value`. Expects that `channel` is
not empty (otherwise it blocks).
"""
function modify!(ch, val)
  isready(ch) && take!(ch)
  put!(ch, val)
end

"""
    modify_status!(program, value)

Exchange the current status of the program. Should only be used by program code
that handles signals.
"""
function modify_status!(k :: AbstractProgram, val :: Symbol)
  s = getstate(k)
  ch = s.status
  modify!(ch, val)
end

function locked_wait(condition)
  lock(condition) do
    wait(condition)
  end
end

function locked_notify(condition)
  lock(condition) do
    notify(condition)
  end
end

"""
    checkpoint(program; exit_on_stop)

Function that can be used to check the status of the program. It is intended for
usage from within program code.

If the status is either `:running` or `:stopped`, `checkpoint` returns `true` or
`false`. If the status is `:interrupted` or `:failed`, `checkpoint` throws an
`InterruptProgram` exception. If the status is `:paused`, `checkpoint` waits until
`pause_condition` is notified before checking the status again.
"""
function checkpoint(k :: AbstractProgram; exit_on_stop = false)
  status = getstatus(k)
  if status in [:interrupted, :failed]
    @logdebug k "status :$status"
    throw(InterruptProgram())
  elseif exit_on_stop && status in [:stopping, :stopped]
    @logdebug k "status :$status"
    throw(StopProgram())
  elseif status == :paused
    @logdebug k "status :$status"
    s = getstate(k)
    locked_wait(s.pause)
    checkpoint(k)
  end
end

"""
    handle_error(program, err)
    handle_error(f, program)

Properly handles an exception `err` from within a catch block in program code.

If `err` is of type `InterruptProgram` or `InvalidStateException` and the program
status is `:interrupted`, the exception is understood to come from proper program
interruption. If the program status is `:failed`, an original error is understood
to have been handled already. If none of the above applies, a `:fail` signal is
emitted to `program` and `err` is attached to the program.

The second method calls `f()` and takes care of handling exceptions.
"""
function handle_error(k :: AbstractProgram, err)
  status = getstatus(k)
  if err isa InterruptProgram && status == :interrupted
    nothing # proper interruption, don't report
  elseif err isa StopProgram && status in [:stopping, :stopped]
    nothing # proper interruption, don't report
  elseif err isa InvalidStateException && status in [:interrupted, :stopping]
    nothing # proper interruption or stop, don't report
  elseif status == :failed
    nothing # some exception was already handled, don't report
  else
    @logerror k "encountered error ($status): $err"
    s = getstate(k)
    put!(s.signal, :fail)
    s.err = err
  end
end

function handle_error(f :: Function, k :: AbstractProgram)
  try f()
  catch err
    handle_error(k, err)
  end
end

"""
    handle_signals(program; on_stop, on_exit)

Listens to the `signal` channel of `program` and modifies the status accordingly.
Also notifies `program.pause` on `:resume` signals.

If the signal is one of `:interrupt` or `:fail`, the function closes the
`program.signal` channel and returns. Before doing so, the corresponding
`on_[signal]` routine is called, which should clean up the local storage of the
program.

If the signal is `:stop`, the function sets the status to `:stopped` but
continues to handle signals. Only when `k.signal` is closed by code of the
program (which should have reacted to `:stopped`) the function returns.
"""
function handle_signals(k :: AbstractProgram; on_stop, on_exit)
  try
    s = getstate(k)
    while true
      signal = try
        take!(s.signal)
      catch err
        # The function only exits in a normal way if the channel k.signal is
        # closed. This is intended to be triggered by the other program functions
        # that react on the `:stopped` status.
        if err isa InvalidStateException
          nothing # this is a sign that the rest of the program finished :)
        else
          s.err = err
          modify_status!(k, :failed)
          locked_notify(s.pause)
          @logerror k "error that should not be possible: $err"
        end
        break
      end
      @logdebug k "received signal :$signal"
      if signal == :interrupt
        modify_status!(k, :interrupted)
        locked_notify(s.pause)
        break
      elseif signal == :fail
        modify_status!(k, :failed)
        locked_notify(s.pause)
        break
      elseif signal == :stop
        modify_status!(k, :stopping)
        locked_notify(s.pause)
        on_stop() # this should cause close(k.signal) from somewhere in the program
      elseif signal == :pause
        modify_status!(k, :paused)
      elseif signal == :resume
        modify_status!(k, :running)
        locked_notify(s.pause)
      end
    end
  catch err
    @logerror k "encountered error: $err"
    handle_error(k, err)
  finally
    on_exit()
  end
end

"""
    isfull(ch)

Returns true if the channel `ch` is currently fully occupied.
"""
function isfull(ch)
  if ch.sz_max == 0
    isready(ch)
  else
    length(ch.data) >= ch.sz_max
  end
end

"""
    put_maybe!(ch, value)
    put_maybe!(program, ch, value)

Non-blocking put!. If the channel is full, return immediately.
If `program` is provided, a log message is left if the channels is full, and
a stop signal is emitted to `program` if the channel is full.

Note that this function is susceptible to race conditions.
"""
function put_maybe!(ch, value)
  !isfull(ch) && put!(ch, value)
end

function put_maybe!(k, ch, value)
  if isfull(ch)
    @logwarn k "discarding value of type $(typeof(value))"
  else
    try put!(ch, value)
    catch err
      if err isa InvalidStateException
        @logdebug k "channel closed"
        stop(k)
      else
        @logerror k "encountered error: $err"
        rethrow()
      end
    end
  end
end

"""
    @take_or! ch [action]

Try to take from `ch`. If this results in an error due to `ch` being closed,
conduct `action`, which may, e.g., be one of `return`, `break`, or `continue`.
"""
macro take_or!(ch, action)
  quote
    try Base.take!($ch)
    catch err
      if err isa InvalidStateException
        $action
      else
        @logerror k "encountered_error: $err"
        rethrow()
      end
    end
  end |> esc
end

"""
    @fetch_or ch [action]

Try to fetch from `ch`. If this results in an error due to `ch` being closed,
conduct `action`, which may, e.g., be one of `return`, `break`, or `continue`.
"""
macro fetch_or(ch, action)
  quote
    try Base.fetch($ch)
    catch err
      if err isa InvalidStateException
        $action
      else
        @logerror k "encountered_error: $err"
        rethrow()
      end
    end
  end |> esc
end


# ------- Dummy Program ------------------------------------------------------- #

"""
Proof of concept program that can be used as template for more meaningful
implementations.
"""
mutable struct DummyProgram <: AbstractProgram
  state :: ProgramState

  host_to_program :: Channel{String}
  program_to_host :: Channel{String}
end

function DummyProgram(program_to_host = Channel{String}(100))
  state = ProgramState()
  host_to_program = Channel{String}(100)

  DummyProgram(state, host_to_program, program_to_host)
end

function init_storage(k :: DummyProgram)
  (; )
end

function on_stop(k :: DummyProgram, _)
  close(k.host_to_program)
end

function on_exit(k :: DummyProgram, _)
  close(k.host_to_program)
end

function run(k :: DummyProgram, storage)
  i = 0
  while getstatus(k) == :running
    msg = @take_or! k.host_to_program break           # break if stop signal arrives
    @log k "received message $i"
    put_maybe!(k, k.program_to_host, "msg $i: $msg")  # don't put if channel is full
    checkpoint(k)                                   # causes pauses if paused, raises
                                                     # exceptions if interrupted or failed
    i += 1
  end
end


# ------- Selfplay Program ---------------------------------------------------- #

"""
Program that lets a player play against itself, returning the datasets produced.

#### Config
* `gpu :: Int = -1`: GPU device used by the model
* `async :: Int = 50`: Number of async selfplays

#### Input
* `players`: Tuple of an `AbstractPlayer` and context information, which
  may include the fields `:branch_step_[min|max]`, `branch_prob`, and
  `instance_randomization`.

#### Output
* `data`: Tuple of a `DataSet`, corresponding to the states from one selfplay,
  together with the context in which the selfplay took place.
"""
mutable struct SelfplayProgram <: AbstractProgram
  state :: ProgramState
  config :: Dict{Symbol, Any}

  data :: Channel{Tuple{DataSet, Any}}
  player :: Channel{Tuple{AbstractPlayer, Any}}
end

function SelfplayProgram(config, data = nothing)
  state = ProgramState()

  if isnothing(data)
    data = Channel{Tuple{DataSet, Any}}(100)
  end
  player = Channel{Tuple{AbstractPlayer, Any}}(1)

  SelfplayProgram(state, config, data, player)
end

function init_storage(k :: SelfplayProgram)
  player = Channel{AbstractPlayer}(1)
  ctx = Channel(1)
  counter = Ref(0)
  (; player, ctx, counter)
end

function on_stop(k :: SelfplayProgram, storage)
  close(k.player)
  close(storage.player)
  close(storage.ctx)
end

on_exit(k :: SelfplayProgram, storage) = on_stop(k, storage)

function run(k :: SelfplayProgram, storage)
  async = getasync(k)
  @sync begin
    @logdebug k "initializing storage handling"
    @async handle_error(k) do 
      update_storage(k, storage)
    end
    @logdebug k "initializing selfplay cycles"
    @async begin
      asyncmap(1:async; ntasks = async) do i
        handle_error(k) do
          cycle_selfplays(k, storage, i)
        end
      end
    end
  end
end

function getasync(k)
  async = get(k.config, :async, 50) :: Integer
  @assert 1 <= async <= 8192
  async
end

function setdevice!(k)
  gpu = get(k.config, :gpu, -1) :: Integer
  if gpu >= 0
    @assert gpu < length(CUDA.devices())
    CUDA.device!(gpu)
    @log k "setting gpu device to $gpu"
    true
  else
    false
  end
end

function update_storage(k :: SelfplayProgram, storage)
  gpu = setdevice!(k)
  async = getasync(k)

  while getstatus(k) == :running
    player, ctx = @take_or! k.player break
    @log k "received new player / context"
    player = Player.tune(player; gpu, async)
    modify!(storage.player, player)
    modify!(storage.ctx, ctx)
  end
end

function cycle_selfplays(k, storage, i)
  G = Model.gametype(fetch(storage.player))

  while getstatus(k) == :running && isopen(k.data)
    ctx = @fetch_or storage.ctx break

    min = get(ctx, :branch_step_min, 1)
    max = get(ctx, :branch_step_max, 5)
    prob = get(ctx, :branch_prob, 0.0)
    random = get(ctx, :instance_randomization, 0.0)

    instance() = Game.instance(G; random)
    branch(game) = Game.branch(game; prob, steps = min:max)

    time = @elapsed begin
      # updates of storage.player immediately affect selfplays :)
      ds = Player.record( storage.player, 1
                        ; threads = false
                        , merge = true
                        , branch
                        , instance
                        , callback_move = (_) -> checkpoint(k) )

      put_maybe!(k, k.data, (ds, ctx))
    end

    count = storage.counter[]
    @log k "dataset $count generated in $time seconds (task $i)"
    storage.counter[] += 1
  end
end 

# ------- Training Program --------------------------------------------------- #

"""
Program that takes in data and trains a model on it.

#### Input
* `players`: Tuple of an `AbstractPlayer` and context information, which
  may include the fields `:branch_step_[min|max]`, `branch_prob`, and
  `instance_randomization`.

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

function init_storage(k :: TrainingProgram{G}) where {G}
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

    test = test_pool_starves(k, storage, ds)

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

function test_pool_starves(k, storage, ds)
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

function configure_optimizer!(k, storage)

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
    time = @elapsed for (i, cache) in enumerate(batches)
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

end # module Program

