
#
# jtac-play
#

import .Msg: Train, Play
using .Events

function rchannel(:: Type{T}, n) where {T}
  Distributed.RemoteChannel(() -> Channel{T}(n))
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

"""
    play(<keyword arguments>)

Start a jtac-play instance that connects to a jtac-train instance to support it.
The train instance sends all models and player related data to the play
instance, and it tasks the play instance to either generate datasets for
training (via self plays) or to simulate contests.

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
- `workers` : list of worker ids that are used for computationally intensive tasks, must not contain the main process (default: all available workers)
"""
function play(
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

  pname = Stl.name("jtac-play")
  Log.info("initializing $pname")

  # leave early if the specified workers are not valid
  # TODO: should probably also do some other sanity checks
  if !valid_workers(workers) return end

  # initialize channels that will be used for communication between tasks
  # / workers
  ch = Dict{String, Distributed.RemoteChannel}(
      "exit"            => rchannel(Bool, 1)
    , "session"         => rchannel(String, 1)
    , "data-request"    => rchannel(Msg.ToPlay.DataReq, 1)
    , "contest-request" => rchannel(Msg.ToPlay.ContestReq, 100)
    , "upload"          => rchannel(Union{Msg.FromPlay.Data, Msg.FromPlay.Contest}, 100))

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
    with_gentle_exit(on_exit, name = "jtac-play", log = Log.info) do
      @sync begin
        @async play_communication(ch, cos)
        @async play_computation(ch, wos, workers)
      end
    end
  finally
    close(ch["exit"])
    Log.debug("finally reached toplevel. closing exit channel")
  end

  Log.debug("exiting $pname")
end

# communication task
# ---------------------------------------------------------------------------- #

function play_communication(ch, os)
  Log.debug("entered play_communication")

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
  cleanup(s = true) = begin
    isnothing(sock[])    || close(sock[])
    isnothing(c_data)    || close(c_data)
    isnothing(c_contest) || close(c_contest)

    # we do not want to close stop if we are waiting for reconnect
    s && (isnothing(stop) || close(stop))
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

    # play_connect! waits for a login confirmation on the socket
    # it creates. to make this responsive to user exit, we have
    # to pass a reference to the socket that is modified by play_connect!
    # session is nothing if the connection attempt failed
    session = play_connect(ch, sock, os)

    # if something is not right, wait and clean up, then try again
    if isnothing(sock[]) || isnothing(session) || isready(ch["exit"])
      cleanup(false)
      sdelay = Stl.quant("$(os.delay)")
      Log.warn("trying again in $sdelay seconds...")
      wait_or_exit(stop, os.delay)
    
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
          Log.info("will resume previous session $s")
        end
        @sync begin
          # these two functions are blocking through sock, c_data, and
          # c_contest. they shall only throw exceptions if they encounter a hard
          # bug. in case of connection problems (sock closes), they simply
          # return and the loop begins anew
          @async play_download(ch, sock[], c_data, c_contest)
          @async play_upload(ch, sock[], c_data, c_contest, id_data, id_contest)
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

  Log.debug("exiting play_communication")
end

function play_connect(ch, sock, os)
  Log.debug("entered play_connect")

  dest = Stl.name("$(os.ip):$(os.port)")
  Log.debug("trying to log in to $dest")

  # when we hit io based errors in one of the steps of connecting + login, we do
  # not want to rethrow them, but to cancel play_connect! gracefully.
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
      login = Msg.FromPlay.Login(os.user, os.token, os.accept_data, os.accept_contest)
      Msg.send(sock[], login)
    end
  end

  if !isnothing(sock[])
    # wait for the train instance to reply
    reply = catch_io_exn("receiving auth from $dest failed") do
      Msg.receive(sock[], Msg.LoginAuth)
    end

    # the reply could not be understood
    if isnothing(reply)
      Log.warn("receiving auth from $dest failed: could not understand reply")

    # the reply tells us that we were refused
    elseif !reply.accept
      smsg = Stl.string(reply.msg)
      Log.warn("connection to $dest refused: $smsg")

    # the reply tells us that we were accepted
    else
      smsg = Stl.string(reply.msg)
      ssess = Stl.string(reply.session)
      Log.info("connection to $dest established. welcoming message:")
      Log.info("  $smsg")
      Log.debug("exiting play_connect (session $ssess)")
      return reply.session
    end
  end

  Log.debug("exiting play_connect (no session)")
end

function play_download(ch, sock, c_data, c_contest)
  Log.debug("entered play_download")

  # when something goes wrong, we want to return from this function and also
  # cause play_upload to return
  cleanup() = (close(sock); close(c_data); close(c_contest))

  catch_com_exn(f) = begin
    exts = [InvalidStateException, Base.IOError, EOFError]
    msg(exn) = "download routine failed: $exn"
    catch_recoverable(f, ch, _ -> cleanup(), msg, exts)
  end

  # the train instance wants us to disconnect
  handle(msg :: Msg.ToPlay.Disconnect) = begin
    Log.info("received request to disconnect")
    cleanup()
  end

  # we receive a new data generation request
  handle(msg :: Msg.ToPlay.DataReq) = begin
    sreq = Stl.quant(msg)
    Log.info("received new request $sreq")
    isready(ch["data-request"]) && take!(ch["data-request"])
    put!(ch["data-request"], msg)
  end

  # we receive a new contest request
  # note that contest requests can stack, while there is always only one current
  # data request
  handle(msg :: Msg.ToPlay.ContestReq) = begin
    sreq = Stl.quant(msg)
    Log.info("received new request $sreq")
    put!(ch["contest-request"], msg)
  end

  # we receive the confirmation that a data or contest package was
  # uploaded sucessfully. this information is passed to play_upload via
  # c_data and c_contest
  handle(msg :: Msg.ToPlay.DataConfirm)    = put!(c_data, msg.id)
  handle(msg :: Msg.ToPlay.ContestConfirm) = put!(c_contest, msg.id)

  # we received input that we could not parse
  # handle(:: Nothing) =

  # loop until the socket is closed
  while isopen(sock)
    catch_com_exn() do
      handle(Msg.receive(sock, Msg.Message{Train, Play}))
    end
  end
  Log.debug("exiting play_download")
end

function play_upload(ch, sock, c_data, c_contest, id_data, id_contest)
  Log.debug("entered play_upload")

  # when something goes wrong, we want to return from this function and also
  # notify play_download to return
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
      if data isa Msg.FromPlay.Data
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
        Msg.send(sock, data)
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
  Log.debug("exiting play_upload")
end


# computation task
# ---------------------------------------------------------------------------- #

function play_computation(ch, os, workers)
  Log.debug("entered play_computation")
  
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
  req_contest = [rchannel(Msg.ToPlay.ContestReq, 100) for _ in workers]
  res_contest = rchannel(Msg.FromPlay.Contest, 3 * length(workers))

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
      @async play_contests(ch, os, req_contest, res_contest)

      # start all workers 
      @async play_workers(ch, os, workers, req_contest, res_contest, mcs)
    end
  catch err
    rethrow_or_exit(ch, err)
  finally
    cleanup()
  end

  Log.debug("exiting play_computation")
end

function play_contests(ch, os, req_contest, res_contest)
  Log.debug("entered play_contests")

  # each loop corresponds to the processing of one received contest request
  while isopen(res_contest)
    # we wait for the next request. take it and divide it into smaller
    # pieces that are sent to the workers
    req = take!(ch["contest-request"])
    subreqs = play_subdivide_contests(req, playings) #TODO
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
  Log.debug("exiting play_contests")
end

function play_workers(ch, os, workers, req_contest, res_contest, mcs)
  Log.debug("entered play_workers")

  # count the number of available GPUs 
  ngpu = os.use_gpu ? length(CUDA.devices()) : 0
  if os.use_gpu && length(workers) > ngpu
    sworkers = Stl.name("$(length(workers)) workers")
    sngpu = Stl.quant("$ngpu")
    Log.info("note: $sworkers will share $sngpu GPU device[s]")
  end

  # start all workers by calleng the play_work function on them
  @sync begin
    for i in 1:length(workers)
      Distributed.@spawnat workers[i] begin
        # set the CUDA GPU device
        os.use_gpu && CUDA.device!((i-1) % ngpu)
        # do the actual work, i.e., either conduct self plays or simulate
        # a competition
        play_work(ch, os, i, req_contest[i], res_contest, mcs[i])
      end
    end
  end
  Log.debug("exiting play_workers")
end

function play_work(ch, os, i, reqc, resc, mc)
  Log.debug(mc, "entered play_work")
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
      x, cache = play_selfplay(ch, os, i, reqc, resc, mc, cache, req)

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
  Log.debug(mc, "exiting play_work")
end

function play_selfplay(ch, os, i, reqc, _, mc, cache, req :: Msg.ToPlay.DataReq)
  Log.debug(mc, "entered play_selfplay (data)")

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
    player = Player.build_player(req._spec |> decompress, gpu = os.use_gpu, async = os.async)
  end
  
  # conduct the actual self plays and measure the time it takes the worker
  # to do them
  time = @elapsed begin
    prepare = Training.prepare(steps = req.init_steps)
    branch = Training.branch(prob = req.branch, steps = req.branch_steps)
    Log.debug(mc, "generating dataset...")
    ds = Training.record_self( player, os.playings
                             , augment = req.augment
                             , prepare = prepare
                             , branch = branch
                             , callback_move = cb
                             , merge = false)
  end

  stime = Stl.quant(round(time, digits = 3))
  Log.info(mc, "finished $sk self plays for $sreq in $stime seconds")

  Log.debug(mc, "exiting play_selfplay (data)")

  # return the data package and the new cache
  Msg.FromPlay.Data(ds, req.reqid, i, time), (req.reqid, player)
end


function play_selfplay(ch, os, i, reqc, resc, mc, cache, req :: Msg.ToPlay.ContestReq)
  Log.debug(mc, "entered play_selfplay (contest)")
  @assert false "contests are not yet implemented"
  Log.debug(mc, "exiting play_selfplay (contest)")
end

