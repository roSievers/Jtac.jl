
#
# jtac train
#

# This is important for the shutdown of the train function to work properly.
# Base.@sync has the disadvantage that tasks can fail silently and the exception
# will never reach the outside code for error handling.
import Base.Experimental: @sync

"""
Auxiliary type that collects information about connected clients.
"""
struct ClientInfo{R}
  login :: Login{R}
  sock :: Sockets.TCPSocket
  ip :: Sockets.IPAddr
  port :: UInt16
end

"""
    remote_channel(type, n)

Create an remote channel with `n` slots of `type`.
"""
remote_channel(type, n) = Distributed.RemoteChannel(() -> Channel{type}(n))

function set_token(tok)
  tok = isnothing(tok) ? join(rand(0:9, 5)) : tok
  Log.info("the authentification token for this session is '$tok'")
  tok
end

function parse_history(file)
  try
    map(readlines(file)) do line
      parse_json(line, Event)
    end
  catch err
    if err isa SystemError
      Log.error("history file '$file' cannot be accessed")
      []
    else
      rethrow(err)
    end
  end
end

init_model(m :: String, folder) = Jtac.load_model(joinpath(folder, m))
init_model(m :: Jtac.Model, _) = m

init_history(:: Nothing, _) = []
init_history(h :: String, folder) = parse_history(joinpath(folder, h))

#function init_history(resume :: Bool, m :: String)
#  if resume
#    base = Filesystem.splitext(m)[1]
#    parse_history(base * ".jth")
#  else
#    []
#  end
#end

optional(:: Nothing, b) = b
optional(a, b) = a

ipv4(ip :: Sockets.IPv4) = ip
ipv4(str :: String) = Sockets.IPv4(str)

function initialize(context, model, folder, name, info, base, history)
  if !Filesystem.isdir(folder)
    Log.info("creating data folder '$folder'")
    Filesystem.mkpath(folder)
  else
    Log.info("data folder '$folder' already exists")
  end
  if model isa String && isnothing(base)
    model_base = Filesystem.basename(model) |> Filesystem.splitext |> first
  end
  model = init_model(model, folder)
  events = init_history(history, folder)
  push!(events, Event(Model(model, name, info, base)), Event(context))
  context, model, base, events
end

"""
    train(model, name, context; <keyword arguments>)

Start a jtac train session with initial model `model` and context `context`.
"""
function train( model :: Union{String, Jtac.NeuralModel{<: Jtac.Game, false}}
              , model_name :: String
              , context :: Context
              ; ip          = "127.0.0.1"
              , port        = 7788
              , token       = nothing
              , user        = ENV["USER"]
              , model_info  = ""
              , model_base  = nothing
              , use_gpu     = false
              , worker      = 2 # process id of train worker
              , max_clients = 50
              , folder      = "data"  # data folder used for model files
              , history     = nothing )

  # TODO: what about contests? <-- Do that when everything else works, should be
  # modular

  Log.info("initializing jtac training session")

  channels = Dict{String, Distributed.RemoteChannel}(
      "context"          => remote_channel( Context, 1 )
    , "context-update"   => remote_channel( Context, 1 )
    , "ref-model"        => remote_channel( Jtac.NeuralModel, 1 )
    , "ref-model-update" => remote_channel( Jtac.NeuralModel, 1 )
    , "data-request"     => remote_channel( TrainDataServe, 1 )
    , "data"             => remote_channel( Tuple{String, ServeData}, 100 )
    , "contest-request"  => remote_channel( TrainContestServe, 100 )
    , "contest"          => remote_channel( ServeContest, 100 )
    , "history"          => remote_channel( Vector{Event}, 100 )
    , "shutdown"         => remote_channel( Bool, 1 ) )

  meta = (folder, model_name, model_info, model_base)
  context, model, model_base, events = initialize(context, model, meta..., history)
  token = set_token(token)

  put!(channels["context"], context)
  put!(channels["ref-model"], model)
  put!(channels["history"], events)

  com = nothing
  trn = nothing

  # TODO: have to treat interrupt exceptions for this block, too, since
  # interrupt could happen during execution / compilation

  try 
    @sync begin
      com = @async comtask(channels, ipv4(ip), port, token, user, max_clients)
      trn = @async trntask(channels, folder, model_name, use_gpu, worker)
    end
  catch err
    ok  = err isa TaskFailedException && shutdown_fail(err.task)
    ok |= shutdown_exn(err)
    ok |= err isa LoadError && shutdown_exn(err.error)
    if ok
      Log.info("received shutdown signal in toplevel")
    else
      Log.error("uncaught error reached toplevel: $err")
      soft_shutdown!(channels)
      rethrow(err)
    end
    soft_shutdown!(channels)
    wait_tasks([com, trn])
  end

  # closing all channels
  hard_shutdown!(channels)

  # TODO: Better options to choose jth and jtm file name / path
  jth = joinpath(folder, "$model_name.jth")
  write(jth, join(json.(com.result), "\n"))
  Log.info("saved the history of the training session as '$jth'")

  if isnothing(trn.result)
    Log.warn("could not save model: nothing returned by training task")
  elseif trn.result isa Exception
    Log.warn("could not save model: training task returned exception")
  elseif trn.result isa Jtac.NeuralModel
    jtm = joinpath(folder, model_name)
    Jtac.save_model(jtm, trn.result)
    Log.info("saved model returned by the training task as '$jtm.jtm'")
  else
    Log.warn("could not save model")
  end
end


# Error handling

"""
Exception that sub-elements of the program may throw in order to indicate that
the session should be terminated gracefully. Like `InterruptException`, this
exception is treated specially and leads to the activation of a global shutdown
signal.
"""
struct Shutdown <: Exception end

"""
Exception that is thrown if a tasks within a `wrap_shutdown` block exits as
intended. This is used to distinguish normal task output from real exceptions
/ shutdown exceptions.
"""
struct Return <: Exception
  result
end

shutdown_exn(exn) = isa(exn, Union{InterruptException, Shutdown})
shutdown_fail(t) = shutdown_exn(t.exception)

function wait_tasks(tasks, report_internal = true)
  for t in tasks
    try wait(t)
    catch err
      if report_internal && !shutdown_fail(t) && !(t.exception isa Return)
        Log.warn("internal task failed: $(t.exception)")
      end
    end
  end
end

wait_shutdown(ch)  = (fetch(ch["shutdown"]); throw(Shutdown()))
soft_shutdown!(ch) = (!isready(ch["shutdown"]) && put!(ch["shutdown"], true))
hard_shutdown!(ch) = for (_,c) in ch close(c) end
check_shutdown(ch) = isready(ch["shutdown"])
throw_shutdown()   = throw(Shutdown())

"""
    wrap_shutdown(f, error_handling, channels)

Executes `f()` while listening for `Shutdown` and `InterruptException` errors
asynchronously. Both of these exceptions cause `error_handling()` to be called,
while other exceptions are re-thrown. Soft shutdowns are also handled by waiting
for `channels["shutdown"]` and are treated like `Shutdown` exceptions.

Returns the value of `f()` when it completes without exceptions and is not
interrupted by shutdown signals.
"""
function wrap_shutdown(f :: Function, error_handling, channels, on_return = x -> x)
  wrap_shutdown([f], error_handling, channels)
end

"""
    wrap_shutdown(fs, error_handling, channels)

Execute all functions in the iterable `fs`. Acts like `wrap_shutdown` with
a single function as argument. Note, however, that the functions in `fs` should
never return without exception (otherwise, the tasks for the other threads are
left hanging uninterrupted).
"""
function wrap_shutdown(fs, on_shutdown, channels, on_return = x -> x)
  tasks = []
  returned = false
  try
    @sync begin
      stask = @async while !returned
        check_shutdown(channels) && throw(Shutdown())
        sleep(0.5)
      end
      push!(tasks, stask)
      for f in fs
        push!(tasks, @async throw(Return(f())))
      end
    end
  catch err
    ok  = shutdown_exn(err) 
    ok |= err isa TaskFailedException && shutdown_fail(err.task)
    ok |= err isa Distributed.RemoteException && shutdown_exn(err.captured.ex)
    ok |= err isa LoadError && shutdown_exn(err.error)
    if ok
      soft_shutdown!(channels)
      on_shutdown()
      wait_tasks(tasks, false)
    elseif err isa Distributed.ProcessExitedException
      Log.error("received process exited exception: $err")
      soft_shutdown!(channels)
      on_shutdown()
      wait_tasks(tasks, false)
    elseif check_shutdown(channels)
      on_shutdown()
      wait_tasks(tasks, false)
    elseif err isa TaskFailedException
      err = err.task.exception
      if err isa Return
        res = on_return(err.result)
        returned = true
        wait_tasks(tasks, true)
        res
      else
        rethrow(err)
      end
    else
      rethrow(err)
    end
  end
end


function sleep_shutdown(channels, delay)
  s = 0
  while s < delay
    check_shutdown(channels) && throw_shutdown()
    s += @elapsed sleep(0.5)
  end
end

"""
    change!(channel, value)

Remove the content of `channel` and replace it by `value`. Intended to be
applied on channels of length 1.
"""
function change!(channel, value)
  if isready(channel)
    take!(channel)
  end
  put!(channel, value)
end


# Client authentication

"""
    authenticate(login, sock, token, username)

Check if the `login` credentials provided by the client connected to `sock` are
valid. For authentication, only a simple test for the correct `token` is
performed. The variable `username` is used to formulate a welcoming message.

Returns either `nothing` (in case of failed authentication) or a value of type
`ClientInfo`.
"""
function authenticate(login, sock, ip, port, token, user, sess)
  # TODO: DANGEROUS: getpeername can cause a segfault if called when not
  # connected !!!
  reject(msg) = (send(sock, LoginAuth(false, msg, sess)); close(sock))

  if login.token != token
    Log.info("authentication of $ip:$port failed: wrong token '$(login.token)'")
    reject("token '$(login.token)' incorrect.")

  elseif login.version != VERSION
    Log.info("authentication of $ip:$port failed: wrong version '$(login.version)'")
    msg = "jtac server version $(login.version) " *
          "is not supported (need $(VERSION))."
    reject(msg)

  else
    msg1 = "Welcome, $(login.name)! "
    msg2 = "Thanks for supporting the jtac training session of $(user)."
    send(sock, LoginAuth(true, msg1 * msg2, sess))
    ClientInfo(login, sock, ip, port)
  end
end

authenticate(:: Nothing, sock, _, _, _) = close(sock) 


# Communicating with clients


"""
    handle_client!(channels, history, client)

This function governs the communication with `client` after a TCP connection has
been established and authentication was successful.

Information obtained from the client / information intended to be sent to the
client is communicated through the program via `channels` and `history`.
"""
function handle_client!(channels, _, client :: ClientInfo{Serve})

  name = client.login.name
  contest = Channel{ServeContest}(1)
  history = channels["history"]

  body = Client(name, string(client.ip), client.port, false)
  put!(history, [Event(body)])
  Log.info("connection to serve client $(client.ip):$(client.port) ($name) established")

  send_request(req, type) = try
    send(client.sock, req)
  catch err
    if !shutdown_exn(err)
      Log.error("sending $type request to client $name failed: $err")
    end
    rethrow(err)
  end

  data_loop() = begin
    req = nothing
    cdr = channels["data-request"]
    while isopen(cdr) && isopen(client.sock)
      if isready(cdr)
        r = fetch(cdr)
        if isnothing(req) || r.reqid > req.reqid
          req = r
          Log.info("sending new data request D$(r.reqid) to client $name")
          send_request(req, "data")
        end
      end
      sleep(1)
    end
  end

  contest_loop() = begin
    ccr = channels["contest-request"]
    while isopen(ccr) && isopen(client.sock)
      if isready(ccr)
        r = take!(ccr)
        Log.info("sending contest request to client $(client.login.name)")
        send_request(r, "contest")

        # after some time, a contest should arrive
        data = take!(contest)
        put!(channels["contests"], data)
        send(client.sock, TrainContestConfirmServe(data.id))
        body = Contest(client.login.name, r, data)
        put!(history, [Event(body)])
      end
      sleep(1)
    end
  end

  receive_message() = try
    !eof(client.sock)
    receive(client.sock, Message{Serve, Train})
  catch err
    if err isa Base.IOError
      Log.error("receiving data from client $name failed: connection closed")
    elseif !shutdown_exn(err) && !(err isa EOFError)
      Log.error("receiving data from client $name failed: $err")
      rethrow(err)
    end
  end

  receive_loop() = while isopen(client.sock)
    data = receive_message()
    if data isa ServeData
      put!(channels["data"], (name, data))
      body = Data(name, data)
      put!(history, [Event(body)])
      send(client.sock, TrainDataConfirmServe(data.id))
    elseif data isa ServeContest
      if length(contest.data) < contest.sz_max
        put!(contest, data)
      else
        Log.warn("contest queue for client $name is full")
        take!(contest, data)
        put!(contest, data)
      end
    elseif data isa ServeLogout
      close(client.sock)
      body = Client(name, client.ip, client.port, true)
      put!(history, [Event(body)])
    elseif isnothing(data)
      break
    end
  end

  disconnect() = begin
    body = Client(name, string(client.ip), client.port, true)
    put!(history, [Event(body)])
    Log.info("disconnected from client $name")
  end

  tasks = Any[receive_loop]
  client.login.accept_data && push!(tasks, data_loop)
  client.login.accept_contest && push!(tasks, contest_loop)

  shutdown = false
  on_shutdown() = (shutdown = true; close(client.sock))
  on_return(_)  = close(client.sock)

  try
    wrap_shutdown(tasks, on_shutdown, channels, on_return)
  catch err
    close(client.sock)
    Log.error("communication with client $name failed: $err")
    disconnect()
    rethrow(err)
  end

  disconnect()
  if shutdown throw(Shutdown()) end

end

function handle_client!(channels, hist :: Vector{Event}, client :: ClientInfo{Monitor})

  count = length(hist)
  name  = client.login.name
  Log.info("connection to monitor $(client.ip):$(client.port) ($name) established")

  send_events(events) = try
    for ev in events send(client.sock, ev) end
  catch err
    if !shutdown_exn(err)
      Log.error("failed to send event to monitor $name: $err")
    end
    rethrow(err)
  end

  event_loop() = while isopen(client.sock)
    if count < length(hist)
      send_events(hist[count+1:end])
      count = length(hist)
    else
      sleep(1)
    end
  end

  receive_context() = try
    if !eof(client.sock)
      receive(client.sock, Event)
    end
  catch err
    if err isa Union{Base.IOError, EOFError}
      Log.error("receiving context from monitor $name failed: connection closed")
    elseif !shutdown_exn(err)
      Log.error("receiving context from monitor $name failed: $err")
      rethrow(err)
    end
  end

  context_loop() = while isopen(client.sock)
    ev = receive_context()
    if !isnothing(ev) && ev.body isa Context
      i = findlast(e -> e.body isa Context, hist)
      ev.body.id = isnothing(i) ? 0 : hist[i].body.id + 1
      put!(channels["history"], [Event(ev.body)])
      put!(channels["context-update"], ev.body)
      Log.info("received context $(ev.body.id) from monitor $name")
      # TODO: When we quickly add two contests, they get the same id, since
      # update of hist is slower than receiving second context
    end
  end

  shutdown = false
  on_shutdown() = (shutdown = true; close(client.sock))
  on_return(_) = close(client.sock)

  try
    wrap_shutdown(() -> send_events(hist), on_shutdown, channels, on_return)
    wrap_shutdown([event_loop, context_loop], on_shutdown, channels, on_return)
  catch err
    close(client.sock)
    Log.error("communication with monitor $name failed: $err")
    Log.info("disconnected from monitor $name")
    rethrow(err)
  end

  Log.info("disconnected from monitor $name")
  if shutdown throw(Shutdown()) end

end

"""
    update_history!(channels, history)

Watches the "history" channel in `channels` and appends new events to `history`
as they arise.
"""
function update_history!(channels, history)
  hist = channels["history"]
  while isopen(hist) && !check_shutdown(channels)
    if isready(hist)
      events = take!(hist)
      for event in events
        id = length(history)
        push!(history, Event(id, event))
      end
    else
      sleep(1)
    end
  end
end


"""
    handle_clients!(channels, server, history, tasks, max_clients)

Listen to `server` and handle incoming client connections.
"""
function handle_clients!(channels, server, history, token, user, max_clients)
  sess = Random.randstring(10) # session identification string
  ctasks = []

  on_connection(sock, ip, port) = try
    login = receive(sock, Login)
    client = authenticate(login, sock, ip, port, token, user, sess)
    if isnothing(client)
      Log.info("connection request from $ip:$port rejected (login failed)")
    else
      task = @async handle_client!(channels, history, client)
      push!(ctasks, (sock, task))
    end
  catch err
    if !check_shutdown(channels) && !shutdown_exn(err)
      Log.warn("client $ip:$port disconnected during login")
    else
      rethrow(err)
    end
  end

  accept_connection() = begin
    sock = Sockets.accept(server)
    # TODO: getpeername sometimes segfaults in other parts of the code. Observe
    # if it also makes problems here. Maybe reproduce it and report bug?
    ip, port = Sockets.getpeername(sock)
    Log.info("connection request from $ip:$port")
    filter!(x -> !istaskdone(x[2]), ctasks)
    if length(ctasks) >= max_clients
      close(sock) # we reject
      Log.info("connection request from $ip:$port rejected (too many clients)")
    else
      wrap_shutdown(() -> on_connection(sock, ip, port), () -> close(sock), channels)
    end
  end

  on_shutdown() = begin
    Log.info("waiting for client tasks to return...")
    for (sock, _) in ctasks close(sock) end
    close(server)
    wait_tasks([t for (_, t) in ctasks])
  end

  while isopen(server)
    try
      wrap_shutdown(accept_connection, on_shutdown, channels)
    catch err
      if !check_shutdown(channels)
        Log.error("client task failed: $err")
      end
      on_shutdown()
    end
  end
end

"""
    comtask(channels, ip, port token, user, max_clients)

Starts the main communication tasks of a jtac train instance.
The functions include
- collect history information from throughout the jtac train session
- register monitor / serve clients
- send them requests / history information
- receive data sets and contest results / context changes and propagate them
  through the session
"""
function comtask(channels, ip, port, token, user, max_clients)
  Log.info("listening for client connections at $ip:$port")
  server = Sockets.listen(ip, port)
  history = Event[]

  htask() = update_history!(channels, history)
  ctask() = handle_clients!(channels, server, history, token, user, max_clients)

  try
    wrap_shutdown([htask, ctask], () -> close(server), channels)
  catch err
    Log.error("communication task failed: $err")
    rethrow(err)
  end

  Log.info("shutting down communication task")
  history
end

#-----------------------------------------------------------------------#
# Train task

function refresh_data!(train, test, channel, reqid, ctx, msg)
  reqid = fetch(reqid)
  while isready(channel)
    name, rec = take!(channel)
    rid = rec.reqid
    id  = rec.id

    if rid > reqid
      Log.warn(msg, "discarding dataset d$id:D$rid ($name) with impossible request id $rid")
      continue
    end

    ds = decompress(rec._data)
    Log.info(msg, "decompressed d$id:D$rid ($name) ($(length(ds)) datasets)")

    for d in ds
      length(test) 
      pool = rand() < ctx.test_frac ? test : train
      add!(pool, d, rid, reqid)
    end

    tr, te = length(train), length(test)
    Log.info(msg, "data d$id:D$rid ($name) assigned to training ($tr states) and testing ($te states)")
  end

  for (name, pool) in [("train", train), ("test", test)]
    len = length(pool)
    if len == 0
      Log.info(msg, "$name pool is empty")
    else
      update_age!(pool, reqid)
      update_capacity!(pool, ctx.capacity)
      r, rp = cleanse!(pool, ctx)
      Log.info(msg, "cleaning $name pool: $len -> $r states ($rp of positive quality)")
    end
  end

  len = length(train)
  if len > 0
    qavg = Statistics.mean(quality(train, ctx))
    Log.info(msg, "new average quality of train pool: $qavg")
  else
    qavg = 0.
  end

  if len == 0
    false
  elseif len < ctx.epoch_size
    Log.info(msg, "train pool too small for next epoch ($len < $(ctx.epoch_size))")
    false
  elseif qavg < ctx.min_quality
    Log.info(msg, "quality of train pool too low ($qavg < $(ctx.min_quality)")
    false
  else
    true
  end
end

function adapt_optimizer!(model, c, cold)
  lr = c.learning_rate
  gamma = c.momentum
  if isnothing(cold) || gamma != cold.momentum || lr != cold.learning_rate
    Jtac.set_optimizer!(model, Knet.Momentum, lr = lr, gamma = gamma)
  end
end

function train_epoch!(model, train, test, ctx, epoch, era, channels, msg)
  # Callback to halt training
  cb(_) = if check_shutdown(channels)
    Log.info(msg, "model received shutdown signal during training")
    throw_shutdown()
  end

  trainset, idx, qavg = random_selection(train, ctx, ctx.epoch_size)
  update_use!(train, idx)

  # information for logging / events
  qavg = round(qavg, digits = 3)
  cap = length(train) / train.capacity
  Log.info(msg, "selected training data of quality $qavg for epoch $epoch")

  l = ctx.loss_weights
  loss = Jtac.Loss(value = l[1], policy = l[2], reg = l[3])
  time = @elapsed Jtac.train!( model, trainset
                             ; loss = loss
                             , callback_step = cb
                             , batchsize = ctx.batch_size
                             , epochs = ctx.iterations )

  losses = map([trainset, test.data]) do ds
    names = Jtac.loss_names(loss)
    # TODO: make maxbatch adaptable when starting jtac train instance
    if length(ds) > 0
      ls = Jtac.loss(loss, model, ds, maxbatch = 1024)
    else
      ls = [0. for _ in names]
    end
    [collect(zip(names, ls)); ("total", sum(ls[1:3]))]
  end

  len = length(trainset)
  Log.info(msg, "finished epoch $epoch in $time seconds")
  ep = Epoch(epoch, losses..., qavg, cap, length(trainset), ctx.id, era)
  put!(channels["history"], [Event(ep)])
  len 
end

function conclude_era(model, folder, name, epoch, era, epochcount, channels, msg)
  ctx = fetch(channels["context"])

  if epochcount < ctx.era_size
    return false
  end

  Log.info(msg, "era $era finished after training on $epochcount states")

  try
    if ctx.backups != 0
      # save the model (with possibly updated context)
      path = joinpath(folder, "$(name)-$era")
      Log.info(msg, "saving reference model as '$path.jtm'")
      Jtac.save_model(path, model |> Jtac.to_cpu)

      # remove backup files that are too old
      if ctx.backups > 0 && era - ctx.backups >= 1
        path = joinpath(folder, "$(name)-$(era - ctx.backups)")
        if Base.Filesystem.isfile(path)
          Base.Filesystem.rm(path)
        end
      end
    end
  catch err
    Log.error(msg, "unexpected error when trying to save model: $err")
  end

  # initiate new era
  change!(channels["ref-model-update"], copy(model |> Jtac.to_cpu))
  put!(channels["history"], [Event(Era(era+1, epoch, []))])
  Log.info(msg, "era $(era+1) starts")

  true
end

function await_data(channels)
  while true
    if check_shutdown(channels)
      throw_shutdown()
    elseif isready(channels["data"])
      break
    else
      sleep(1.)
    end
  end
end

function wrap_train_worker(proc, args...)
  @sync Distributed.@spawnat proc train_worker(args...)
end

function precompile_trainstep(msg, model)
  Log.info(msg, "simulating training step to trigger precompilation...")
  m = copy(model)
  gt = Jtac.gametype(m)
  len = Jtac.policy_length(gt)
  games = [gt() for _ in 1:100]
  label = [rand(Float32, 1+len) for _ in 1:100]
  flabel = Vector{Float32}[]
  ds = Jtac.DataSet(games, label, flabel)
  Jtac.train!(m, ds, epochs = 1, batchsize = 20)
  Log.info(msg, "precompilation done")
end

function train_worker(channels, folder, name, use_gpu, reqid, return_model, msg)
  Log.info(msg, "training worker initialized")
  data = channels["data"]

  # get reference model and start era
  model = fetch(channels["ref-model"])
  change!(channels["ref-model-update"], model)
  put!(channels["history"], [Event(Era(0, 0, []))]) # TODO: era metrics

  # actual model that is used for training
  model = use_gpu ? Jtac.to_gpu(model) : copy(model)

  trainpool = DataPool{Jtac.gametype(model)}(0)
  testpool = DataPool{Jtac.gametype(model)}(0)

  era   = 0
  epoch = 0
  epochcount = 0

  ctx     = nothing
  ctx_old = nothing

  precompile_trainstep(msg, model)

  # each loop is one epoch
  try
    while true
      ctx = fetch(channels["context"])
      ok = refresh_data!(trainpool, testpool, data, reqid, ctx, msg)

      if !ok
        Log.info(msg, "waiting for fresh data before training can continue")
        await_data(channels)
        continue
      end

      adapt_optimizer!(model, ctx, ctx_old)
      len = train_epoch!(model, trainpool, testpool, ctx, epoch, era, channels, msg)
      epoch += 1
      ctx_old = ctx

      # begin new era?
      epochcount += len
      if conclude_era(model, folder, name, epoch, era, epochcount, channels, msg)
        epochcount = 0
        era += 1
      end
    end
  catch err
    if shutdown_exn(err) || check_shutdown(channels)
      change!(return_model, model |> Jtac.to_cpu)
      Log.info(msg, "received shutdown signal")
      throw_shutdown()
    else
      change!(return_model, model |> Jtac.to_cpu)
      Log.error(msg, "unexpected error: $err")
      rethrow(err)
    end
  end
end

function trntask(channels, folder, name, use_gpu, proc)
  Log.info("starting training task for model '$name'")

  msg = remote_channel(String, 100)
  reqid = remote_channel(Int, 1)
  change!(reqid, 0)

  handle_msg() = while true
    Log.info("worker: " * take!(msg))
  end

  commit_request(ctx, model, msg) = begin
    rid = fetch(reqid) + 1
    change!(channels["data-request"], TrainDataServe(rid, ctx, model))
    put!(channels["history"], [Event(Datareq(rid, ctx.id))])
    Log.info("prepared new data request D$rid ($msg)")
    change!(reqid, rid)
  end

  handle_context_changes() = while !check_shutdown(channels)
    if isready(channels["context-update"])
      ctx = take!(channels["context-update"])
      change!(channels["context"], ctx)
      model = fetch(channels["ref-model"])
      Log.info("training task noticed that the context changed")
      commit_request(ctx, model, "context update")
    else
      sleep(1.)
    end
  end

  handle_ref_changes() = while !check_shutdown(channels)
    if isready(channels["ref-model-update"])
      model = take!(channels["ref-model-update"])
      change!(channels["ref-model"], model)
      ctx = fetch(channels["context"])
      Log.info("training task noticed that the reference model changed")
      commit_request(ctx, model, "model update")
    else
      sleep(.25)
    end
  end

  return_model = remote_channel(Jtac.NeuralModel, 1)
  start_worker() = wrap_train_worker(proc, channels, folder, name, use_gpu, reqid, return_model, msg)

  tasks = [handle_msg, handle_context_changes, handle_ref_changes, start_worker]

  on_shutdown() = (sleep(0.5); close(msg))

  try
    wrap_shutdown(tasks, on_shutdown, channels)
  catch err
    Log.error("training task failed: $err")
    soft_shutdown!(channels)
    on_shutdown()
    # wait_tasks(tasks) <-- have to get tasks from wrap_shutdown!
    rethrow(err)
  end

  Log.info("shutting down training task")
  if isready(return_model)
    fetch(return_model)
  else
    Log.error("worker did not return trained model")
  end
end

