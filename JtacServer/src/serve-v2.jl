
#
# jtac-serve
#

# if we do not want an exception to end the program, and only print a warning
# / message if user exit was not issued
function catch_recoverable(f, ch, on_catch, msg :: Function, exts; warn = Log.warn)
  try f()
  catch err
    if err isa Union{exts...} 
      isready(ch["exit"]) || warn(msg(err))
      on_catch(err)
    else
      rethrow(err)
    end
  end
end

function catch_recoverable(f, ch, on_catch, msg, args...; kw...)
  catch_recoverable(f, ch, on_catch, _ -> msg, args...; kw...)
end

# stolen from https://github.com/JuliaLang/julia/issues/36217
# unfortunately, it does not seem to be possible to wait for
# ch["exit"] directly (since we have to close the resource c
# in the code below at the end of timeout)
function wait_or_exit(c, timeout :: Real) 
  timer = Timer(timeout) do t
    isready(c) || close(c)
  end
  try wait(c)
  catch nothing
  finally close(timer)
  end
end

# typing this function is inconvenient, since RemoteChannels are not
# AbstractChannels...
function replace!(ch, value)
  isready(ch) && take!(ch)
  put!(ch, value)
end

function rchannel(:: Type{T}, n) where {T}
  Distributed.RemoteChannel(() -> Channel{T}(n))
end

function close_data_channels!(ch)
  for (key, c) in ch
    if key != "exit" close(c) end
  end
end

function valid_workers(workers)
  if 1 in workers
    Log.error("using the main process as worker is not supported")
    false
  elseif !issubset(workers, Distributed.workers())
    diff = setdiff(workers, Distributed.workers())
    s = Stl.faulty("$(Tuple(diff))")
    Log.error("requested workers $s are not available")
    false
  else
    true
  end
end

# Only rethrow an error if "exit" is not set
# If the debug mode is activated, show that an error
# is not rethrown because of "exit"
function rethrow_or_exit(ch, err)
  if !isready(ch["exit"])
    rethrow(err)
  elseif DEBUG[]
    e = Stl.error(string(err))
    Log.debug("caught exception absorbed by exit signal: $e")
    throw(err)
  end
end

"""
    serve(<keyword arguments>)

Start a jtac-serve instance that connects to a jtac-train instance to serve it.
The train instance sends all model and player specific data to the serve
instance, and tasks the serve instance to either generate datasets for training
(via self plays) or to simulate contests.

!!! This function will abort if no worker processes are available. See option
`workers` below.

# Arguments
- `ip` : ip address of the jtac-train instance (default: `127.0.0.1`)
- `port` : port of the jtac-train instance (default: `7788`)
- `user` : user name. (default: `USER` enviroment variable)
- `token` : token to authenticate login at jtac-train instance (default: `""`)
- `playings` : number of playings per sent dataset (default: `50`)
- `use_gpu` : use GPU-support via Knet (default: `false`)
- `async` : use async mcts runs (default: `true`, recommended if `use_gpu`)
- `delay` : delay in seconds before next login attempt after failure (default: `10`)
- `gc_interval` : interval in seconds for manual calls to `GC.gc()` on workers (default: `10`)
- `accept_data` : data generation requests are accepted (default: `true`)
- `accept_contest` : contest requests are accepted (default: `true`)
- `workers` : list of worker ids that are used for computationally intensive tasks. must not contain the main process (default: all available workers)
"""
function serve(
              ; ip    = "127.0.0.1"
              , port  = 7788
              , user  = ENV["USER"]
              , token = ""
              , playings = 50
              , use_gpu  = false
              , async    = true
              , workers  = Distributed.workers()
              , delay    = 10
              , gc_interval    = 10
              , accept_data    = true
              , accept_contest = true)

  pname = Stl.name("jtac-serve")
  Log.info("initializing $pname")

  # leave early if the specified workers are not valid
  # TODO: should probably also do a range of other sanity checks
  if !valid_workers(workers) return end

  # initialize channels that will be used for communication between tasks
  # / workers
  ch = Dict{String, Distributed.RemoteChannel}(
      "exit"            => rchannel(Bool, 1)
    , "session"         => rchannel(String, 1)
    , "data-request"    => rchannel(TrainDataServe, 1)
    , "contest-request" => rchannel(TrainContestServe, 100)
    , "upload"          => rchannel(Union{ServeData, ServeContest}, 100))

  # TODO: Session channel is *empty* if we are not connected
  # -> we have to fetch in worker, not poll!

  # options that affect the communication
  cos = ( ip = ipv4(ip)
        , port = port
        , user = user
        , token = token
        , delay = delay
        , accept_data = accept_data
        , accept_contest = accept_contest )

  # options that affect the computation / workers
  wos = ( playings = playings
        , use_gpu = use_gpu
        , async = async
        , gc_interval = gc_interval )

  # when the user issues ctrl-d, fill the exit channel and close other channels.
  # all tasks and their children should see this and shut down gracefully, i.e.,
  # without exceptions
  on_exit() = begin
    put!(ch["exit"], true)
    close_data_channels!(ch)
  end

  # start the communication and compute tasks
  try
    with_gentle_exit(on_exit, name = "jtac-serve", log = Log.info) do
      @sync begin
        @async serve_communication(ch, cos)
        @async serve_computation(ch, wos, workers)
      end
    end
  finally
    close(ch["exit"])
    Log.debug("finally reached toplevel. closing exit channel")
  end

  Log.info("exiting $pname")

end

# communication task
# ---------------------------------------------------------------------------- #

function serve_communication(ch, os)
  Log.debug("entered serve_communication")

  last_session = ""    # store past jtac-train session id (to resume)

  id_data     = Ref(1) # counter for the dataset id
  id_contest  = Ref(1) # counter for the contest id

  c_data    = nothing  # confirmation id of successful data upload
  c_contest = nothing  # confirmation id of successful contest upload

  sock = Ref{Any}(nothing)  # socket for communication
  stop = nothing            # stop signal for wait_or_exit below

  # if the session changes, we reset the counters above and clear data channels
  reset_session(sess) = begin
    id_data[] = id_contest[] = 1
    for key in ["data-request", "contest-request", "upload"]
      while isready(ch[key]) take!(ch[key]) end
    end
  end

  # cleaning up all resources that are introduced in this scope
  cleanup() = begin
    isnothing(stop)      || close(stop)
    isnothing(sock[])    || close(sock[])
    isnothing(c_data)    || close(c_data)
    isnothing(c_contest) || close(c_contest)
  end

  # cleaning up on user induced exit
  @async if fetch(ch["exit"]) cleanup() end

  # try to connect. retry if the connection is lost
  while isopen(ch["exit"]) && !isready(ch["exit"])
    # communicate confirmation ids of data and contest uploads
    c_data = Channel{Int}(1)
    c_contest = Channel{Int}(1)

    # this channel only exists so that it can be closed in case of user exit,
    # which notifies wait_or_exit below
    stop = Channel() 

    # serve_connect! waits for a login confirmation on the socket
    # it creates. to make this responsive to user exit, we have
    # to pass a reference to the socket that is modified by serve_connect!
    # session is nothing if the connection attempt failed
    session = serve_connect!(ch, sock, os)

    # if something is not right, wait and clean up, then try again
    if isnothing(sock[]) || isnothing(session) || isready(ch["exit"])
      sdelay = Stl.quant("$(os.delay)")
      Log.warn("trying again in $sdelay seconds...")
      wait_or_exit(stop, os.delay)
      cleanup()
    
    # the connection is established, run the upload and download tasks
    else
      try
        # check if we connected to the same session as before. if we did,
        # notify the user
        put!(ch["session"], session)
        if session != last_session 
          reset_session(session)
        else
          s = Stl.keyword(session)
          log_info("will resume previous session $s")
        end
        @sync begin
          # these two functions are blocking through sock, c_data, and
          # c_contest. they shall only throw exceptions if they encounter a hard
          # bug. in case of connection problems (sock closes), they simply
          # return and the loop begins anew
          @async serve_download(ch, sock[], c_data, c_contest)
          @async serve_upload(ch, sock[], c_data, c_contest, id_data, id_contest)
        end
      catch err
        rethrow_or_exit(ch, err)
      finally
        cleanup()
        take!(ch["session"])
        last_session = session
      end
    end
  end

  Log.info("shutting down communications task")
end

function serve_connect!(ch, sock, os)
  Log.debug("entered serve_connect")

  dest = Stl.name("$(os.ip):$(os.port)")
  Log.debug("trying to log in to $dest")

  # when we hit io based errors in one of the steps of connecting + login, we do
  # not want to rethrow them, but to cancel serve_connect! gracefully.
  # throwing an error here means that it reaches the toplevel (except if
  # user exit has been specified)
  catch_io_exn(f, msg) = begin
    on_catch(_) = isnothing(sock[]) || (close(sock[]); sock[] = nothing)
    exns = [EOFError, Base.IOError, Sockets.DNSError]
    catch_recoverable(f, ch, on_catch, msg, exns)
  end

  # try to connect to the jtac-train instance via tcp
  catch_io_exn("cannot connect to $dest") do
    sock[] = Sockets.connect(os.ip, os.port)
  end

  if !isnothing(sock[])
    # if connecting was successful, send login credentials
    catch_io_exn("sending login request to $dest failed") do
      login = ServeLogin(os.user, os.token, os.accept_data, os.accept_contest)
      send(sock[], login)
    end
  end

  if !isnothing(sock[])
    # wait for the train instance to reply
    reply = catch_io_exn("receiving auth from $dest failed") do
      receive(sock[], LoginAuth)
    end

    # the reply could not be understood
    if isnothing(reply)
      Log.warn("receiving auth from $dest failed: could not understand reply")

    # the reply tells us that we were refused
    elseif !reply.accept
      msg = Stl.string(reply.msg)
      Log.warn("connection to $dest refused: $msg")

    # the reply tells us that we were accepted
    else
      Log.info("connection to $dest established")
      reply.session
    end
  end
end

function serve_download(ch, sock, c_data, c_contest)
  Log.debug("entered serve_download")

  # when something goes wrong, we want to return from this function and also
  # cause serve_upload to return
  cleanup() = (close(sock); close(c_data); close(c_contest))

  catch_com_exn(f) = begin
    exts = [InvalidStateException, Base.IOError, EOFError]
    msg(exn) = "download routine failed: $exn"
    catch_recoverable(f, ch, _ -> cleanup(), msg, exts)
  end

  # the train instance wants us to disconnect
  handle(msg :: TrainDisconnectServe) = begin
    Log.info("received request to disconnect")
    cleanup()
  end

  # we receive a new data generation request
  handle(msg :: TrainDataServe) = begin
    sreq = Stl.quant(msg)
    Log.info("received new request $sreq")
    replace!(ch["data-request"], msg)
  end

  # we receive a new contest request
  # note that contest requests can stack, while there is always only one current
  # data request
  handle(msg :: TrainContestServe) = begin
    sreq = Stl.quant(msg)
    Log.info("received new request $sreq")
    put!(ch["contest-request"], msg)
  end

  # we receive the confirmation that a data or contest package was
  # uploaded sucessfully. this information is passed to serve_upload via
  # c_data and c_contest
  handle(msg :: TrainDataConfirmServe)    = put!(c_data, msg.id)
  handle(msg :: TrainContestConfirmServe) = put!(c_contest, msg.id)

  # we received input that we could not parse
  # handle(:: Nothing) =

  # loop until the socket is closed
  while isopen(sock)
    catch_com_exn() do
      handle(receive(sock, Message{Train, Serve}))
    end
  end
end

function serve_upload(ch, sock, c_data, c_contest, id_data, id_contest)
  Log.debug("entered serve_upload")

  # when something goes wrong, we want to return from this function and also
  # notify serve_download to return
  cleanup() = (close(sock); close(c_data); close(c_contest))

  # handling exceptions related to sock, c_data, or c_contest
  catch_com_exn(f) = begin
    exts = [InvalidStateException, Base.IOError, EOFError]
    msg(exn) = "upload routine failed: $exn"
    catch_recoverable(f, ch, _ -> cleanup(), msg, exts)
  end

  # stylized printing of data and contest packages, e.g., d5:D2 for the second
  # dataset this client uploaded in the session (for data request 2)
  pp(r, i)     = Stl.quant("$r$i")
  pp(r, i, ri) = Stl.quant("$r$i:$(uppercase(r))$ri")

  # receive datasets from the upload channel and transfer them to the jtac-train
  # instance as long as the socket is open
  while isopen(sock)
    catch_com_exn() do
      # this channel is populated by the compute task, and it can contain data
      # sets or contest results
      data = take!(ch["upload"])
      if data isa ServeData
        r, c, id = "d", c_data, id_data
      else
        r, c, id = "c", c_contest, id_contest
      end

      # determine the worker id and sign the data set (= give it the correct id)
      worker = data.id
      data.id = id[]
      sworker = Stl.keyword("worker $worker")

      # try to upload the package and wait for a confirmation to be put in c
      spackage = pp(r, id[], data.reqid)
      Log.info("uploading $spackage from $sworker")
      time = @elapsed begin
        send(sock, data)
        i = take!(c)
      end

      # as a consistency check, make sure that the confirmation carries the same
      # id as the sent package
      if i == id[]
        stime = Stl.quant(round(time, digits = 3))
        Log.info("upload of $spackage took $stime seconds")
        id[] += 1

      # if the consistency check fails, we disconnect and try to reconnect
      else
        Log.error("upload inconsistency: $(pp(r, i)) instead of $(pp(r, id[]))")
        cleanup()
      end
    end
  end
end


# computation task
# ---------------------------------------------------------------------------- #

function serve_computation(ch, os, workers)
  Log.debug("entered serve_computation")
  
  # each worker gets its own channel to be used for logging
  mcs = [rchannel(String, 100) for _ in workers]

  # function to print messages from different workers. gracefully exits if the
  # corresponding channel is closed
  log_msg(i) = begin
    sworker = Stl.keyword("worker $i:")
    try
      while true Log.info("$sworker " * take!(mcs[i])) end
    catch
      nothing
    end
  end

  # each arriving contest request is split up and distributed to the workers.
  # for this reason, each worker also gets a channel to fetch contests from. the
  # results are collected in a single channels and combined in by the main
  # process.
  req_contest = [rchannel(TrainContestServe, 100) for _ in workers]
  res_contest = rchannel(ServeContest, 3 * length(workers))

  # a cleanup function to close the resources created in this scope
  cleanup() = begin
    for mc in mcs close(mc) end
    for rc in req_contest close(rc) end
    close(res_contest)
  end

  # start listening to the user exit signal
  @async if fetch(ch["exit"]) cleanup() end

  try
    @sync begin
      # logging by the workers
      for i in 1:length(workers) @async log_msg(i) end

      # register new contest requests, split them up, and distribute
      # them to the channels in req_contest
      @async serve_contests(ch, os, req_contest, res_contest)

      # start all workers 
      @async serve_workers(ch, os, workers, req_contest, res_contest, mcs)
    end
  catch err
    rethrow_or_exit(ch, err)
  finally
    cleanup()
  end

  Log.info("shutting down computation task")
end

function serve_contests(ch, os, req_contest, res_contest)
  Log.debug("entered serve_contests")

  # each loop corresponds to the processing of one received contest request
  while isopen(res_contest)
    # we wait for the next request. take it and divide it into smaller
    # pieces that are sent to the workers
    req = take!(ch["contest-request"])
    subreqs = serve_subdivide_contests(req, playings) #TODO
    k = length(subreqs)

    # some logging
    sl, sk, sreq = Stl.quant(req.length), Stl.quant(k), Stl.quant(req)
    Log.info("dividing $sreq of length $sl in $sk sub-requests")

    # place the subcontests in the respective worker queues.
    # the workers conduct the simulations place the results in
    # res_contest
    for (i, subreq) in enumerate(subreqs)
      wid = (i % m) + 1 
      put!(req_contest[wid], subreq)
    end

    # after some time, we will receive k result contests to take 
    res = map(_ -> take!(res_contest), 1:k)
    time = sum(x -> x.time, res)

    # all of the results are combined to a single contest result
    # and put in the gobal upload channel
    Log.info("simulation of $sreq finished")
    data = combine(res, time)
    put!(ch["upload"], data)
  end
end

function serve_workers(ch, os, workers, req_contest, res_contest, mcs)
  Log.debug("entered serve_workers")

  # count the number of available GPUs 
  ngpu = os.use_gpu ? length(CUDA.devices()) : 0
  if os.use_gpu && length(workers) > ngpu
    sworkers = Stl.name("$(length(workers)) workers")
    sngpu = Stl.quant("$ngpu")
    Log.info("note: $sworkers will share $sngpu GPU device[s]")
  end

  # start all workers by calleng the serve_work function on them
  @sync begin
    for i in 1:length(workers)
      Distributed.@spawnat workers[i] begin
        # set the CUDA GPU device
        os.use_gpu && CUDA.device!((i-1) % ngpu)
        # do the actual work, i.e., either conduct self plays or simulate
        # a competition
        serve_work(ch, os, i, req_contest[i], res_contest, mcs[i])
      end
    end
  end
end

function serve_work(ch, os, i, reqc, resc, mc)
  Log.debug(mc, "entered serve_work")
  reqd = ch["data-request"]

  # variable that will contain information about the prior iteration. reusing
  # it, if the session has not changed, saves some computational effort
  cache = nothing

  # reaction if a channel is closed
  msg = "channel became unavailable"
  on_catch() = (close(mc); close(reqc))
  exts = [InvalidStateException, WorkerStopException]

  # main simulation loop. one iteration is one set of self plays or one contest
  while isopen(mc)
    catch_recoverable(ch, on_catch, msg, exts, warn = s -> Log.warn(mc, s)) do
      # get the current session. if we are not connected to a jtac-train server
      # this will block. the block will be released on reconnect, or if the
      # user decides to exit (which will cause an InvalidStateException here)
      session = fetch(ch["session"])

      # TODO: I don't know how to do the selection of the next request to tackle
      # without polling when we cannot wait / select for the first out of two
      # events...
      req = nothing
      while isnothing(req)
        if     isready(reqc) req = take!(reqc)
        elseif isready(reqd) req = fetch(reqd)
        else   sleep(0.25) end
      end

      # conduct the request
      x, cache = serve_play(ch, os, i, reqc, resc, mc, cache, req)

      # before we upload the data record, we have to check if we are still in
      # the same session
      if fetch(ch["session"]) == session
        put!(ch["upload"], x)
      
      # if the session has changed, we have to discard the data
      else
        sreq = Stl.quant(req)
        Log.warn(mc, "discarding record for $sreq (session changed)")
        cache = nothing
      end

      # make sure that we free as much memory as possible. not doing this has
      # shown to lead to suboptimal GC performance
      GC.gc()
      GC.gc()
    end
  end
end

function serve_play(ch, os, i, reqc, _, mc, cache, req :: TrainDataServe)
  Log.debug(mc, "entered serve_play")
  # how many playings will actually take place
  k = clamp(os.playings, req.min_playings, req.max_playings)
  sk = Stl.quant("$k")
  sreq = Stl.quant(req)
  Log.info(mc, "starting $sk self plays for $sreq")

  # we experienced memory issues when not regularly calling GC.gc(). this
  # callback takes care of this, and also checks if we should stop playing
  # due to user exit
  now = Dates.time()
  cb() = begin
    if isready(ch["exit"])
      throw(WorkerStopException())

    # the absolute value here makes sure that we call the GC even if the
    # system time is changed at some point during the self playings
    elseif abs(Dates.time() - now) > os.gc_interval
      now = Dates.time()
      GC.gc()
      GC.gc()
    end
  end

  # check the cache and reuse the old player if the request id has not changed
  if !isnothing(cache) && req.reqid == cache[1]
    Log.debug(mc, "reusing cached player")
    player = cache[2]

  # if the request id has changed, we have to build the player anew
  else
    Log.debug(mc, "building new player from request $sreq")
    player = build_player(req.spec, gpu = os.use_gpu, async = os.async)
  end
  
  # conduct the actual self plays and measure the time it takes the worker
  # to do them
  time = @elapsed begin
    prepare = Jtac.prepare(steps = req.init_steps)
    branch = Jtac.branch(prob = req.branch, steps = req.branch_steps)
    Log.debug(mc, "generating dataset...")
    ds = Jtac.record_self( player, os.playings
                         , augment = req.augment
                         , prepare = prepare
                         , branch = branch
                         , callback_move = cb
                         , merge = false)
  end

  stime = Stl.quant(round(time, digits = 3))
  Log.info(mc, "finished $sk self plays for $sreq in $stime seconds")

  # return the data package and the new cache
  ServeData(ds, req.reqid, i, time), (req.reqid, player)
end


function serve_play(ch, os, i, reqc, resc, mc, cache, req :: TrainContestServe)
  Log.debug(mc, "entered serve_play")
  @assert false "contests are not yet implemented"
end

