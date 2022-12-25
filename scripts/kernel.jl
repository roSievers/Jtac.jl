
module Kernel

using ThreadPools
using Jtac

import .Data: DataSet
import .Player: AbstractPlayer

"""
Exception that is used to exit kernels in case of interruption by the host.
"""
struct InterruptKernel <: Exception end

"""
    AbstractKernel

A kernel is an program intended to be executed in its own thread by some host.
It can be paused, stopped, and interrupted.

#### Api

On the application level, the host controls the basic behavior of the kernel via
the functions

* `run`
* `pause`
* `resume
* `stop`
* `interrupt`

#### Implementation

Some guidelines when implementing kernels.

0) Implementing a kernel means defining a concrete subtype of `AbstractKernel`
   plus the definition of several methods:

   * `init_storage`
   * `on_stop`
   * `on_exit`
   * `run`

   In the following, the term 'kernel code' refers to these four methods. Any
   concrete subtype must additionally hold the field `state`, which is of type
   `KernelState`. The state is used for logging, basic logic, and error
   handling.

1) Kernel code should distinguish between `host_to_kernel` and `kernel_to_host` channels.

   `host_to_kernel`:

   The kernel may call `fetch` and `take!`
   The kernel takes ownership of the channel and the internal logic closes
   it on shutdown. For this reason, any attempt to `take!` or `fetch` must be
   guarded in the `run(kernel)` portion of the Kernel. Convenience macros like
   `@take_or!` may be used for this purpose.
   The channel should be closed in the `on_stop` and `on_exit` routines
   The channel must not be closed by the host.
   The kernel must not call `put!` on these channels, since this can easily
   lead to blocking.

   `kernel_to_host`:

   The kernel may call `put!`. 
   The host has ownership over the channel and the kernel is forbidden to
   close it.
   If the host cannot keep up or errors, putting should not block the kernel.
   Helper functions like `put_maybe!` can be used in this case
   A closed channel `kernel_to_host` channel should usually be treated like
   a `stop(kernel)` signal, but depending on the circumstances, it does not
   have to be.

2) Stopping a kernel does not imply that it immediately has to return. If
   if it is currently generating a batch of data, it may decide to finish and
   `put!` the data into a `kernel_to_host` channel before it shuts down. Note
   that `kernel_to_host` channel will remain open even after stopping, while
   `host_to_kernel` channels are supposed to close immediately (via `on_stop`
   and `on_exit`).

3) When implementing a kernel, pausing is enforced in the function
   `checkstatus`. Therefore, even after pausing, the kernel may be active as
   long as `checkstatus` has not been reached.
"""
abstract type AbstractKernel end

function Base.show(io :: IO, k :: K) where {K <: AbstractKernel}
  sym = Base.nameof(K)
  status = getstatus(k)
  print(io, "$sym(:$status)")
end

function Base.show(io :: IO, ::MIME"text/plain", k :: AbstractKernel)
  Base.show(io, k)
end

"""
Type to keep track of low-level implementation details that all kernels of type
`AbstractKernel` share. It possesses the following fields:

* `status :: Channel{Symbol}`: A one-element channel that contains the status of
  the kernel. Valid statuses are `:inactive`, `:running`, `:paused`, `:stopped`,
  `:interrupted`, and `:failed`.

* `signal :: Channel{Symbol}`: A multi-element channel that is used for basic
  communication with the host. Valid signals are `:stop`, `:pause`, `:resume`,
  and `:interrupt`.

* `log :: Channel{Tuple{Float64,String}}`: A multi-element channel that is used
  for reporting logging information back to the host.

* `pause :: Condition`: Condition that is used to implement pausing.

* `err :: Union{Nothing, Exception}`: The possible exception that caused the
  kernel to fail.
"""
mutable struct KernelState
  status :: Channel{Symbol}
  signal :: Channel{Symbol}
  log :: Channel{Tuple{Float64,String}}
  pause :: Threads.Condition
  err :: Union{Nothing, Exception}
end

function KernelState()
  status = Channel{Symbol}(1)
  signal = Channel{Symbol}(100)
  log = Channel{Tuple{Float64, String}}(10000)
  pause = Threads.Condition()
  put!(status, :inactive)
  KernelState(status, signal, log, pause, nothing)
end

# ------- Host side interface ------------------------------------------------ #

"""
    start(kernel)

Start an inactive `kernel`. Executes the function `run(kernel, storage)` with
additional handling of logic and errors. The local variable storage `storage` of
the kernel is created by `init_storage(kernel)`.
"""
function start(k :: AbstractKernel)
  @assert getstatus(k) == :inactive "Kernel has been started before"
  log(k, "start: initializing kernel on thread $(Threads.threadid())")
  s = getstate(k)
  try
    modify_status!(k, :running)
    log(k, "start: initializing storage")
    storage = init_storage(k)
    @sync begin
      log(k, "start: initializing signal handling")
      on_stop() = Kernel.on_stop(k, storage)
      on_exit() = Kernel.on_exit(k, storage)
      @async handle_signals(k; on_stop, on_exit)
      log(k, "start: executing kernel")
      handle_error(k) do
        run(k, storage)
      end
      close(s.signal)
    end
  catch err
    s.err = err
    log(k, "start: encountered error: $err")
    modify_status!(k, :failed)
  finally
    getstatus(k) == :stopping && modify_status!(k, :stopped)
    log(k, "start: kernel exited with status :$(getstatus(k))")
    close(s.signal)
    close(s.log)
  end
end

"""
    pause(kernel)

Ask an active `kernel` to pause computation.
"""
function pause(k :: AbstractKernel)
  s = getstate(k)
  if fetch(s.status) != :inactive
    put!(s.signal, :pause)
  end
end

"""
    resume(kernel)

Ask a paused `kernel` to resume computation
"""
function resume(k :: AbstractKernel)
  s = getstate(k)
  if fetch(s.status) != :inactive
    put!(s.signal, :resume)
  end
end

"""
    stop(kernel)

Ask an active `kernel` to gracefully stop computation. Ongoing computations are
finished.
"""
function stop(k :: AbstractKernel)
  s = getstate(k)
  if fetch(s.status) != :inactive
    put!(s.signal, :stop)
  end
end

"""
    interrupt(kernel)

Interrupt an active `kernel`. Partial results of ongoing computations are
discarded.
"""
function interrupt(k :: AbstractKernel)
  s = getstate(k)
  if fetch(s.status) != :inactive
    put!(s.signal, :interrupt)
  end
end


"""
    getstatus(kernel)

Obtain the current status of the kernel. Can be one of

    [:inactive, :running, :paused, :stopped, :interrupted, :failed]

In case the status is :failed, the causing error can be accessed via
`geterror(kernel)`.
"""
function getstatus(k :: AbstractKernel)
  s = getstate(k)
  fetch(s.status) # this should never actually block
end

"""
    geterror(kernel)

If the `kernel` failed, return the causing exception. Otherwise, return
`nothing`.
"""
function geterror(k :: AbstractKernel)
  if getstatus(k) == :failed
    s = getstate(k)
    s.err
  end
end

"""
    getstate(kernel)

Get the state of a kernel.
"""
getstate(k :: AbstractKernel) = k.state

"""
    getlog(kernel)

Get all entries currently logged by the kernel. Note that this removes the
messages from the `log` channel of `kernel.state`.
"""
function getlog(kernel)
  s = getstate(kernel)
  logs = Vector{Tuple{Float64, String}}()
  while isready(s.log)
    push!(logs, take!(s.log))
  end
  logs
end


# ------- Kernel side interface ---------------------------------------------- #

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
    modify_status!(kernel, value)

Exchange the current status of the kernel. Should only be used by kernel code
that handles signals.
"""
function modify_status!(k :: AbstractKernel, val :: Symbol)
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
    checkstatus(kernel)

Function that can be used to check the status of the kernel. It is intended for
usage from within kernel code.

If the status is either `:running` or `:stopped`, `checkstatus` returns `true` or
`false`. If the status is `:interrupted` or `:failed`, `checkstatus` throws an
`InterruptKernel` exception. If the status is `:paused`, `checkstatus` waits until
`pause_condition` is notified before checking the status again.
"""
function checkstatus(k :: AbstractKernel)
  status = getstatus(k)
  if status in [:interrupted, :failed]
    throw(InterruptKernel())
  elseif status == :paused
    s = getstate(k)
    locked_wait(s.pause)
    checkstatus(k)
  end
end

"""
    handle_error(kernel, err)
    handle_error(f, kernel)

Properly handles an exception `err` from within a catch block in kernel code.

If `err` is of type `InterruptKernel` or `InvalidStateException` and the kernel
status is `:interrupted`, the exception is understood to come from proper kernel
interruption. If the kernel status is `:failed`, an original error is understood
to have been handled already. If none of the above applies, a `:fail` signal is
emitted to `kernel.signal` and `err` is attached to the kernel.

The second method calls `f()` and takes care of handling exceptions.
"""
function handle_error(k :: AbstractKernel, err)
  status = getstatus(k)
  if err isa InterruptKernel && status == :interrupted
    nothing # proper interruption, don't report
  elseif err isa InvalidStateException && status == :interrupted
    nothing # proper interruption, don't report
  elseif status == :failed
    nothing # some exception was already handled, don't report
  else
    log(k, "handle_error: encountered error: $err")
    s = getstate(k)
    put!(s.signal, :fail)
    s.err = err
  end
end

function handle_error(f :: Function, k :: AbstractKernel)
  try f()
  catch err
    handle_error(k, err)
  end
end

"""
    handle_signals(kernel; on_stop, on_exit)

Listens to the `signal` channel of `kernel` and modifies the status accordingly.
Also notifies `kernel.pause` on `:resume` signals.

If the signal is one of `:interrupt` or `:fail`, the function closes the
`kernel.signal` channel and returns. Before doing so, the corresponding
`on_[signal]` routine is called, which should clean up the local storage of the
kernel.

If the signal is `:stop`, the function sets the status to `:stopped` but
continues to handle signals. Only when `k.signal` is closed by code of the
kernel (which should have reacted to `:stopped`) the function returns.
"""
function handle_signals(k :: AbstractKernel; on_stop, on_exit)
  try
    s = getstate(k)
    while true
      signal = try
        take!(s.signal)
      catch err
        # The function only exits in a normal way if the channel k.signal is
        # closed. This is intended to be triggered by the other kernel functions
        # that react on the `:stopped` status.
        if err isa InvalidStateException
          nothing # this is a sign that the rest of the kernel finished :)
        else
          s.err = err
          modify_status!(k, :failed)
          locked_notify(s.pause)
        end
        break
      end
      log(k, "handle_signals: received signal :$signal")
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
        on_stop() # this should cause close(k.signal) from somewhere in the kernel
      elseif signal == :pause
        modify_status!(k, :paused)
      elseif signal == :resume
        modify_status!(k, :running)
        locked_notify(s.pause)
      end
    end
  catch err
    @show err
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
    put_maybe!(kernel, ch, value)

Non-blocking put!. If the channel is full, return immediately.
If `kernel` is provided, a log message is left if the channels is full, and
a stop signal is emitted to `kernel` if the channel is full.

Note that this function is susceptible to race conditions.
"""
function put_maybe!(ch, value)
  !isfull(ch) && put!(ch, value)
end

function put_maybe!(k, ch, value)
  if isfull(ch)
    log(k, "put_maybe!: channel full, discarding value")
  else
    try put!(ch, value)
    catch err
      if err isa InvalidStateException
        stop(k)
      else
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
        rethrow()
      end
    end
  end |> esc
end

"""
    log(kernel, str)

Put the message `str` into the `log` channel of the kernel state. This channel
can be used by the host in order to display logged events in the kernel.

Since this function aims to be non-blocking, the message is discarded if the
log channel is full.
"""
function log(k :: AbstractKernel, str)
  t = time()
  s = getstate(k)
  put_maybe!(s.log, (t, str))
end

# ------- Dummy Kernel ------------------------------------------------------- #

"""
Proof of concept kernel that can be used as template for more meaningful
implementations.
"""
mutable struct DummyKernel <: AbstractKernel
  state :: KernelState

  host_to_kernel :: Channel{String}
  kernel_to_host :: Channel{String}
end

function DummyKernel(kernel_to_host = Channel{String}(100))
  state = KernelState()
  host_to_kernel = Channel{String}(100)

  DummyKernel(state, host_to_kernel, kernel_to_host)
end

function init_storage(k :: DummyKernel)
  (; )
end

function on_stop(k :: DummyKernel, _)
  close(k.host_to_kernel)
end

function on_exit(k :: DummyKernel, _)
  close(k.host_to_kernel)
end

function run(k :: DummyKernel, storage)
  i = 0
  while getstatus(k) == :running
    msg = @take_or! k.host_to_kernel break           # break if stop signal arrives
    log(k, "run: received message $i")
    put_maybe!(k, k.kernel_to_host, "msg $i: $msg")  # don't put if channel is full
    checkstatus(k)                                   # causes pauses if paused, raises
                                                     # exceptions if interrupted or failed
    i += 1
  end
end


# ------- Selfplay Kernel ---------------------------------------------------- #

"""
Kernel that lets a player play against itself, returning the datasets produced.
"""
mutable struct SelfplayKernel <: AbstractKernel
  state :: KernelState
  config :: Dict{Symbol, Any}

  data :: Channel{Tuple{DataSet, Any}}
  player :: Channel{Tuple{AbstractPlayer, Any}}
end

function SelfplayKernel(config, data = nothing)
  state = KernelState()

  if isnothing(data)
    data_size = get(config, :data_channel_size, 100)
    data = Channel{Tuple{DataSet, Any}}(data_size)
  end
  player = Channel{Tuple{AbstractPlayer, Any}}(1)

  SelfplayKernel(state, config, data, player)
end

function on_stop(k :: SelfplayKernel, storage)
  close(k.player)
  close(storage.player)
end

on_exit(k :: SelfplayKernel, storage) = on_stop(k, storage)

init_storage(k :: SelfplayKernel) = (; player = Channel(1))

function run(k :: SelfplayKernel, storage)
  async = getasync(k)
  # t_update, t_play = get(k.config, :threads, [2,3])
  @sync begin
    @async handle_error(k) do 
      update_storage(k, storage)
    end
    @async begin
      asyncmap(1:async; ntasks = async) do _
        handle_error(k) do
          cycle_selfplays(k, storage)
        end
      end
    end
  end
end

function getasync(k)
  async = get(k.config, :async, 50) :: Integer
  @assert 1 <= async <= 1000
  async
end

function setdevice!(k)
  gpu = get(k.config, :gpu, -1) :: Integer
  if gpu >= 0
    @assert gpu < length(CUDA.devices())
    CUDA.device!(gpu)
    true
  else
    false
  end
end

function update_storage(k :: SelfplayKernel, storage)
  gpu = setdevice!(k)
  async = getasync(k)

  while getstatus(k) == :running
    player, ctx = @take_or! k.player break
    log(k, "received new context")
    player = Player.tune(player; gpu, async)
    modify!(storage.player, (player, ctx))
  end
  log(k, "update_storage: returning")
end

function cycle_selfplays(k, storage)
  player = ctx = nothing

  while getstatus(k) == :running && isopen(k.data)
    player, ctx = @fetch_or storage.player break

    min = get(ctx, :branch_step_min, 1)
    max = get(ctx, :branch_step_max, 5)
    prob = get(ctx, :branch_prob, 0.0)
    random = get(ctx, :instance_randomization, 0.0)

    G = Model.gametype(player)
    instance() = Game.instance(G; random)
    branch(game) = Game.branch(game; prob, steps = min:max)

    ds = Player.record( player, 1
                      ; threads = false
                      , merge = true
                      , branch
                      , instance
                      , callback_move = (_) -> checkstatus(k) )

    log(k, "cycle_selfplays: dataset generated")
    put_maybe!(k, k.data, (ds, ctx))
    GC.gc()
  end
end 


end # module Kernel
