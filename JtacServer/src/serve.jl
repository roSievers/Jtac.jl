
#
# jtac serve
#

import Base.Experimental: @sync

function keyboard_shutdown(channels)
  while !eof(stdin) end
  log_info("received shutdown signal by the user")
  soft_shutdown!(channels)
end

function serve(
              ; ip    = "127.0.0.1"
              , port  = 7788
              , user  = ENV["USER"]
              , token = ""
              , playings = 50
              , use_gpu  = false
              , async    = false
              , workers  = Distributed.workers()
              , delay    = 10
              , buffer_size    = 100 
              , gc_interval    = 10
              , accept_data    = true
              , accept_contest = true)

  log_info("initializing jtac serve session")
  
  channels = Dict{String, Distributed.RemoteChannel}(
      "shutdown"        => remote_channel( Bool, 1 )
    , "session"         => remote_channel( String, 1 )
    , "data-request"    => remote_channel( TrainDataServe, 1 )
    , "contest-request" => remote_channel( TrainContestServe, 100 )
    , "data"            => remote_channel( ServeData, buffer_size )
    , "contest"         => remote_channel( ServeContest, buffer_size ))

  # we are not connected to any active session
  change!(channels["session"], "")

  com = nothing
  ply = nothing

  try
    @sync begin
      @async keyboard_shutdown(channels)
      com = @async comtask(channels, ipv4(ip), port, token, user, delay, accept_data, accept_contest)
      ply = @async plytask(channels, workers, playings, use_gpu, async, gc_interval)
    end
  catch err
    ok  = err isa TaskFailedException && shutdown_fail(err.task)
    ok |= shutdown_exn(err)
    ok |= err isa LoadError && shutdown_exn(err.error)
    if ok
      log_info("received shutdown signal in toplevel")
    else
      log_error("uncaught error reached toplevel: $err")
      soft_shutdown!(channels)
      rethrow(err)
    end
    soft_shutdown!(channels)
    wait_tasks([com, ply])
  end

  hard_shutdown!(channels)
  log_info("exiting jtac serve session")
end


function connect_server(channels, ip, port, login, delay)
  res = nothing
  while !check_shutdown(channels) && isnothing(res)
    sock = nothing

    try_login() = begin
      log_info("trying to log in to jtac train instance at $ip:$port")
      sock = Sockets.connect(ip, port)
      send(sock, login)
      reply = receive(sock, LoginAuth)
      if isnothing(reply)
        log_warn("connection to $ip:$port failed: receiving login answer failed")
      elseif reply.accept
        log_info("connected to $ip:$port")
        log_info("joining session $(reply.session)")
        log_info("message from $ip:$port received:\n  $(reply.msg)")
        sock, reply.session
      else
        log_info("connection to $ip:$port refused by server: $(reply.msg)")
      end
    end

    on_shutdown() = begin
      !isnothing(sock) && close(sock)
      throw_shutdown()
    end

    try
      res = wrap_shutdown(try_login, on_shutdown, channels)
    catch err
      if err isa Base.IOError
        log_error("connecting to $ip:$port failed: connection refused")
      elseif err isa Shutdown
        soft_shutdown!(channels)
        log_error("connecting to $ip:$port failed: shutdown exception")
      else
        log_error("connecting to $ip:$port failed: $err")
      end
      !isnothing(sock) && close(sock)
    end

    if isnothing(res) && !check_shutdown(channels)
      log_info("trying again in $delay seconds...")
      sleep_shutdown(channels, delay)
    end
  end
  res
end

function comtask(channels, ip, port, token, user, delay, accept_data, accept_contest)
  log_info("communications task initiated")
  login = ServeLogin(user, token, data = accept_data, contest = accept_contest)

  sess = "" # session string
  did = Ref(1)
  cid = Ref(1)

  on_new_session() = begin
    did[] = cid[] = 1
    for key in ["data-request", "data", "contest-request", "contest"]
      while isready(channels[key]) take!(channels[key]) end
    end
  end

  while !check_shutdown(channels)
    datac    = Channel{Int}(1)
    contestc = Channel{Int}(1)

    connect() = connect_server(channels, ip, port, login, delay)
    ret = wrap_shutdown(connect, () -> nothing, channels)
    if isnothing(ret) continue end
    sock, s = ret
    change!(channels["session"], s)

    if sess != s
      on_new_session()
    else
      log_info("received known session identifier $s, will resume previous session")
    end

    dtask() = download_requests(channels, sock, datac, contestc)
    utask() = upload_data(channels, sock, datac, contestc, did, cid)

    on_shutdown() = begin
      close(sock)
      close(datac)
      close(contestc)
    end

    try
      wrap_shutdown([dtask, utask], on_shutdown, channels, _ -> on_shutdown())
    catch err
      on_shutdown()
      log_error("communications task received unexpected error: $err")
      rethrow(err)
    end

    change!(channels["session"], "")
    sess = s
  end

  log_info("shutting down communications task")
end

function upload_data(channels, sock, datac, contestc, did, cid)
  log_info("upload routine initialized")

  upload(channel, confirm, id) = begin
    data = take!(channel)
    r, t = data isa ServeData ? ("d", "D") : ("c", "C")
    worker = data.id
    data.id = id[]
    log_info("uploading record $r$(data.id):$t$(data.reqid) from worker $worker")
    time = @elapsed begin
      send(sock, data)
      i = take!(confirm)
    end
    if i == id[]
      log_info("uploaded $r$i:$t$(data.reqid) in $time seconds")
      id[] += 1
    else
      log_error("inconsistent confirmation: $r$i instead of $r$(data.id)")
      # TODO: should probably raise an exception here and make us reconnect?
    end
  end

  while isopen(sock)
    if isready(channels["data"])
      upload(channels["data"], datac, did)
    elseif isready(channels["contest"])
      upload(channels["contest"], contestc, cid)
    else
      sleep(0.5)
    end
  end
end


function download_requests(channels, sock, datac, contestc)
  log_info("download routine initialized")

  receive_message() = try
    !eof(sock)
    receive(sock, Message{Train, Serve})
  catch err
    close(sock)
    if err isa Union{Base.IOError, EOFError}
      log_error("receiving data failed: connection closed")
    elseif err isa ErrorException
      log_error("receiving data failed: $err")
    elseif shutdown_exn(err)
      rethrow(err)
    else
      log_error("receiving data failed unexpectedly: $err")
      rethrow(err)
    end
  end

  handle(msg :: TrainDisconnectServe) = begin
    log_info("received request to disconnect")
    close(sock)
    throw_shutdown()
  end

  handle(msg :: TrainDataServe) = begin
    log_info("received new request D$(msg.reqid)")
    change!(channels["data-request"], msg)
  end

  handle(msg :: TrainContestServe) = begin
    log_info("received new request C$(msg.reqid)")
    put!(channels["contest-request"], msg)
  end

  handle(msg :: TrainDataConfirmServe) = put!(datac, msg.id)
  handle(msg :: TrainContestConfirmServe) = put!(contestc, msg.id)
  handle(:: Nothing) = (close(sock); log_info("suspending download task"))

  while isopen(sock) handle(receive_message()) end
end


function check_workers(channels, workers)
  if 1 in workers
    log_error("using the main process as worker is not supported")
    soft_shutdown!(channels)
    throw_shutdown()
  elseif !issubset(workers, Distributed.workers())
    log_error("requested workers are not available")
    soft_shutdown!(channels)
    throw_shutdown()
  end
end

function distribute_contests(channels, creq, cres, playings)
  while !check_shutdown(channels)
    if isready(channels["contest-request"])
      req = take!(channels["contest-request"])
      sreqs = subdivide(req, playings)
      k = length(sreqs)
      log_info("dividing C$(req.reqid) with length $(req.length) in $k subrequests")

      for (i, sreq) in enumerate(sreqs)
        idx = (i % m) + 1
        put!(creq[idx], sreq)
      end

      res = map(_ -> take!(cres), 1:k)
      time = sum(x -> x.time, res)
      data = combine(res, time)

      l = sum(cdata.data)
      log_info("collecting record of length $l for C$(cdata.reqid)")
      put!(channels["contest"], data)
    else
      sleep(0.5)
    end
  end
end

function plytask(channels, workers, playings, use_gpu, async, gc_interval)
  log_info("play task initiated")

  check_workers(channels, workers)

  opt = ( use_gpu = use_gpu
        , async = async
        , playings = playings
        , gc_interval = gc_interval )

  creq = [remote_channel(TrainContestServe, 100) for _ in workers]
  cres = remote_channel(ServeContest, 3 * length(workers))
  msgs = [remote_channel(String, 100) for _ in workers]

  handle_msg(i) = while true
    log_info("worker $i: " * take!(msgs[i]))
  end

  ctask() = distribute_contests(channels, creq, cres, playings)
  wtask() = serve_workers(channels, creq, cres, msgs, workers, opt)

  tasks = [ctask, wtask]
  for i in 1:length(workers) push!(tasks, () -> handle_msg(i)) end

  on_shutdown() = begin
    sleep(0.5)  # TODO: we should do something different here to make sure that worker-messages return
    close(cres)
    for msg in msgs close(msg) end
    for req in creq close(req) end
  end

  try
    wrap_shutdown(tasks, on_shutdown, channels)
  catch err
    if err isa Distributed.ProcessExitedException
      log_warn("worker died: $err")
      soft_shutdown!(channels)
      on_shutdown()
    else
      log_error("play task failed: $err")
      soft_shutdown!(channels)
      on_shutdown()
      rethrow(err)
    end
  end

  log_info("shutting down play task")
end

function serve_workers(channels, creq, cres, msgs, workers, options)
  ngpu = options.use_gpu ? length(CUDA.devices()) : 0
  if options.use_gpu && length(workers) > ngpu
    log_info("note: $(length(workers)) workers will share $ngpu GPU device[s]")
  end

  @sync begin
    for i in 1:length(workers)
      Distributed.@spawnat workers[i] begin 
        options.use_gpu && CUDA.device!((i-1) % ngpu)
        play(req, sess, cache) = begin
          serve_play(channels, req, i, cres, msgs[i], sess, cache, options)
        end
        serve_worker(channels, i, creq, msgs[i], play)
      end
    end
  end
end

function serve_worker(channels, wid, creq, msg, play)

  log_info(msg, "worker $wid initialized")
  dreq = channels["data-request"]

  cache = nothing

  # one iteration is one set of selfplays / one (partial) contest
  try
    while true
      req = nothing

      while isnothing(req)
        if isready(creq[wid])
          req = take!(creqs[wid])
        elseif isready(dreq)
          req = fetch(dreq)
        else
          sleep(0.5)
        end
      end

      sess  = fetch(channels["session"])
      cache = play(req, sess, cache)
      GC.gc(); GC.gc()
    end
  catch err
    if shutdown_exn(err) || check_shutdown(channels)
      log_info(msg, "received shutdown signal")
      throw_shutdown()
    else
      log_error(msg, "unexpected error: $err")
      rethrow(err)
    end
  end
end

function serve_play(channels, req :: TrainDataServe, wid, cres, msg, sess, cache, opt)
  k = clamp(opt.playings, req.min_playings, req.max_playings)
  rid = req.reqid
  log_info(msg, "initiating $k playings for D$rid")

  now = Dates.time()
  cb() = begin
    if check_shutdown(channels)
      throw_shutdown()
    elseif Dates.time() - now > opt.gc_interval
      now = Dates.time()
      GC.gc(); GC.gc()
    end
  end

  # reuse old player if the request id has not changed
  if !isnothing(cache) && rid == cache[1]
    log_info(msg, "using cached player")
    player = cache[2]
  else
    log_info(msg, "building player from request")
    player = build_player(req.spec, gpu = opt.use_gpu, async = opt.async)
  end
  
  dt = @elapsed begin
    prepare = Jtac.prepare(steps = req.init_steps)
    branch = Jtac.branch(prob = req.branch, steps = req.branch_steps)
    log_info(msg, "generating dataset...")
    ds = Jtac.record_self( player, opt.playings
                         , augment = req.augment
                         , prepare = prepare
                         , branch = branch
                         , callback_move = cb
                         , merge = false)
  end

  # check if we are still in the same session
  s = fetch(channels["session"])
  while !check_shutdown(channels) && s == ""
    # wait for reconnect if we lost the train session
    sleep(0.5)
    s = fetch(channels["session"])
  end

  if s != sess
    log_warn(msg, "discarding dataset for D$rid since session has changed")
    nothing
  else
    dt3 = round(dt, digits = 3)
    log_info(msg, "finished $k playings for D$rid in $dt3 seconds")
    data = ServeData(ds, rid, wid, dt)
    put!(channels["data"], data)
    (rid, player) # cached information
  end
end

function work(channels, req :: TrainContestServe, wid, _, gpu, async, sess, _, cres, msg)
  rid = req.reqid
  log_info(msg, "starting subcontest of length $(req.length) for request C$rid")

  cb() = (check_shutdown(channels) && throw_shutdown())
  players = build_player.(req.specs, gpu = gpu, async = async)

  time = @elapsed begin
    res = Jtac.compete(players, req.length, req.active, callback = cb)
  end

  s = fetch(channels["session"])
  while s == ""
    sleep(0.5)
    s = fetch(channels["session"])
  end

  if s != sess
    log_warn(msg, "discarding dataset for C$rid as session has changed")
    put!(cres, nothing)
  else
    dt = round(time, digits = 3)
    log_info(msg, "finished subcontest $j / $k for C$rid in $dt seconds")
    data = ServeContest(res, [], NaN, NaN, rid, wid, time)
    put!(cres, data)
  end
  nothing
end

