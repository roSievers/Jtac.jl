
const CLIENT_VERSION = 1

struct WorkerExit <: Exception end 

abstract type InternalRecord end

struct InternalDataRecord <: InternalRecord
  data   :: Vector
  reqid  :: Int
  worker :: Int
  dtime  :: Float64
end

struct InternalContestRecord <: InternalRecord
  data   :: Array{Int, 3} 
  reqid  :: Int
  worker :: Int
  dtime  :: Float64
end

function combine(rs :: Vector{InternalContestRecord}, dtime)
  @assert length(rs) > 1
  reqid = rs[1].reqid
  @assert all(x -> x.reqid == reqid, rs)

  data = sum(x -> x.data, rs)
  InternalContestRecord(data, reqid, 0, dtime)
end

function subdivide(req :: ContestRequest, playings :: Int)
  # When we get a competition request from a train server, we will usually
  # want to subdivide it on the differ workers.
  # Each subcompetition should have at least m playings, otherwise there
  # are pairings that do not play.
  m = Jtac.pairings(length(req.specs), length(req.active))

  # Indeed, we only want to do subcompetitions with a length of multiples of m,
  # such that it is guaranteed that each pairing plays equally often. We thus
  # might play more games (l) than requested (req.length).
  l = ceil(Int, req.length / m) * m

  # When deciding how many playings to do in one subcompetition, we orient
  # ourselves by the 'playings' value of the play server, choosing the next
  # multiple of m that equals or is above playings.
  r = max(1, ceil(Int, playings / m))

  # k subcompetitions with length r*m will be conducted, and one subcompetition
  # with length rem = l - (k * r * m) (which divides m by the definition of m)
  k, rem = divrem(l, r*m)

  requests = [ ContestRequest(req.specs, req.active, r*m, req.id) for _ in 1:k ]
  if rem > 0
    push!(requests, ContestRequest(req.specs, req.active, rem, req.id))
  end

  requests
end


sign_record(r :: InternalDataRecord, id) = DataRecord(r.data, r.reqid, id, r.dtime)
sign_record(r :: InternalContestRecord, id) = ContestRecord(r.data, r.reqid, id, r.dtime)
log(name, msg) = println("<$name> " * msg)

function playclient(
                   ; ip                      = ip"127.0.0.1"
                   , port                    = 7788
                   , name                    = ENV["USER"]
                   , token                   = ""
                   , playings                = 100
                   , gpu                     = false
                   , async                   = false
                   , workers                 = Distributed.workers()
                   , buffersize              = 100 
                   , wait_reconnect          = 10
                   , accept_data_requests    = true
                   , accept_contest_requests = true
                   , kwargs... )

  log(name, "initializing jtac play client...")

  buffersize = max(buffersize, 2length(workers))

  getsocket = name -> login( name, ip, port, token
                           , accept_data_requests
                           , accept_contest_requests
                           , wait_reconnect )

  socket = getsocket(name)

  isnothing(socket) && return
  
  # This slot contains the current socket. The current should ideally be only
  # set once when the program starts, but the train server might send reconnect
  # messages, in which case the slot will be replaced by the download task.

  slot = Channel{TCPSocket}(1)
  put!(slot, socket)

  # Each coroutine has to listen to stop. If it is set to true (it
  # should never be set to false) by any coroutine or worker, all coroutines
  # should try their best to return quickly without exceptions. Since we
  # may not be able to prevent that each worker tries to write to stop
  # once (due to race conditions in a maximalist pessimistic situation),
  # we buffer the channel generously.

  stop = RemoteChannel(() -> Channel{Bool}(buffersize))

  # A DataRequest is the request of the train server to produce as much
  # self-play data as possible, following the instructions outlined in the
  # request. There will always only be one active data request - the most
  # recent one that the client has received.
  #
  # The datareq channel is only modified by the download coroutine that
  # handles messages from the train server. It is fetched by the worker
  # processes.

  datareq = RemoteChannel(() -> Channel{DataRequest}(1))

  # A ContestRequest is the request to conduct a contest between players and
  # transmit the results. The contestreq channel is filled by the download
  # coroutine and taken from by the compute coroutine. To prevent race
  # conditions, taking from contestreq is not done by the workers directly,
  # so we do not need a remote channel for constreq: locally in the compute
  # coroutine, each worker gets it own remote channel.

  contestreq = Channel{ContestRequest}(10)

  # This channel stores the data created by the workers on the occasion of
  # DataRequests or ContestRequests. The workers write to it directly, so new
  # datasets might arrive anytime. The upload coroutine will take from buffer
  # whenever it is ready to upload it to the train server.
  #
  # Under stop, this buffer has to be closed (as workers might to try to
  # put! to it), and will consecutively be read out in order to backup the
  # stored data that has not yet been transmitted. 

  buffer_local = Channel{InternalRecord}(buffersize)
  buffer = RemoteChannel(() -> buffer_local)

  # Internal confirmation passd from download to upload coroutines that signal
  # the server has received the uploaded data

  confirm = Channel{Union{DataConfirmation,ContestConfirmation}}(buffersize)


  # Download task that receives information and requests from the train server
  #
  # put!:  slot, stop, datareq, contestreq, confirm, stop
  # take!: slot, datareq
  # fetch: stop, slot
  # close: slot, datareq, contestreq, confirm, socket

  download_task = @async download( name, slot, getsocket, stop
                                 , datareq, contestreq, confirm
                                 , accept_data_requests
                                 , accept_contest_requests ) 


  # Upload task that sends data to the train server
  #
  # put!:  stop
  # take!: buffer, confirmuploads 
  # fetch: slot, buffer, stop
  # close: slot, buffer, confirm

  upload_task = @async upload(name, slot, stop, buffer, confirm)

  # Compute task that generates self play and contest data on requests
  #
  # put!:
  # take!:
  # fetch:
  # close:

  compute_task = @async compute( name, stop, datareq, contestreq, buffer
                               ; gpu = gpu
                               , async = async
                               , playings = playings
                               , workers = workers
                               , kwargs... )

  try
    
    if !fetch(compute_task)
      
      log(name, "compute coroutine requested shutdown")

    end

    cleanup(name, stop, buffer_local, download_task, upload_task, compute_task, throw = true)

  catch err

    if err isa InterruptException

      @error "<$name> Interrupted"

    elseif err isa Base.IOError

      @error "<$name> Connection to $ip:$port was closed unexpectedly"

    else

      @error "<$name> Unexpected exception: $err"
      throw(err.task.exception)

    end

    cleanup(name, stop, buffer_local, download_task, upload_task, compute_task, throw = false)

  end

  log(name, "exiting jtac play client")

end


function login(name, ip, port, token, data, contest, sleeptime)

  while true

    try

      log(name, "trying to log in to server $ip:$port")

      socket = connect(ip, port)
      client_info = Login(token = token, name = name, data = data, contest = contest)

      send(socket, client_info)
      reply = receive(socket) :: Reply

      if reply.accept

        log(name, "connection established: '$(reply.msg)'")
        return socket

      else

        log(name, "login failed: $(reply.msg)")
        log(name, "exiting jtac play client")
        return nothing

      end

    catch err

      if err isa InterruptException
        log(name, "got interrupted...")
        log(name, "exiting jtac play client")
        return nothing
      end

      if err isa TypeError
        log(name, "answer from server unintelligible, assuming incompability")
        log(name, "exiting jtac play client")
        return nothing
      end

      log(name, "connecting to $ip:$port failed")
      log(name, "will try to connect again in $sleeptime seconds...")

      sleep(sleeptime)

    end

  end
  
end


function cleanup(name, stop, buffer, download_task, upload_task, compute_task; throw = false)

  log(name, "stopping coroutines and closing connection...")

  # This should enable the stopping routines of all threads

  put!(stop, true)

  # Rescue the buffer
  
  backup = []
  for rec in buffer
    push!(backup, rec)
  end

  if throw
  
    wait(download_task)
    wait(upload_task)
    wait(compute_task)

    log(name, "all coroutines finished without exception")

  else

    try wait(compute_task)
    catch err

      if err isa TaskFailedException
        log(name, "compute-coroutine raised an exception: $(err.task.exception)")
      elseif err isa InterruptException
        log(name, "compute-coroutine was interrupted while stopping")
      else
        log(name, "stopping the compute-coroutine raised an exception: $err")
      end

    end

    try wait(download_task)
    catch err 

      if err isa TaskFailedException
        log(name, "download-coroutine raised an exception: $(err.task.exception)")
      elseif err isa InterruptException
        log(name, "download-coroutine was interrupted while stopping")
      else
        log(name, "stopping download-coroutine raised an exception: $err")
      end

    end

    try wait(upload_task)
    catch err

      if err isa TaskFailedException
        log(name, "upload-coroutine raised an exception: $(err.task.exception)")
      elseif err isa InterruptException
        log(name, "upload-coroutine was interrupted while stopping")
      else
        log(name, "stopping upload-coroutine raised an exception: $err")
      end

    end

  end
    
end



function download( name, slot, getsocket, stop, datareq, contestreq
                 , confirm, accept_data_requests, accept_contest_requests)

  name = "$name-1"

  log(name, "download coroutine initialized")

  stopper = @async begin
    fetch(stop)
    close(datareq)
    close(contestreq)
    close(confirm)
    close(slot)
    close(fetch(slot))
  end

  main = @async while !eof(fetch(slot))

    msg = receive(fetch(slot)) :: ServerMsg

    if msg isa Reply

      log(name, "received unexpected login reply from server. Ignoring it")

    elseif msg isa Idle

      log(name, "server asks us to idle: '$(msg.msg)'")

      if isready(datareq)
        req = take!(datareq)
        log(name, "discarded data request D.$(req.id)")
      end

    elseif msg isa Disconnect

      log(name, "server asks us to disconnect: '$(msg.msg)'")
      return

    elseif msg isa Reconnect

      log(name, "server asks us to reconnect after $(msg.wait_time) seconds")
      take!(slot)

      log(name, "waiting...")
      sleep(msg.wait_time)

      log(name, "trying to reconnect")
      socket = getsocket(name)
      
      if isnothing(socket)
        log(name, "reconnecting failed")
        return
      else
        put!(slot, socket)
      end

    elseif msg isa DataRequest && accept_data_requests

      log(name, "received request D.$(msg.id)")
      isready(datareq) && take!(datareq)
      put!(datareq, msg)

    elseif msg isa ContestRequest && accept_contest_requests

      log(name, "received contest request C.$(msg.id)")
      put!(contestreq, msg)

    elseif msg isa DataConfirmation

      #log(name, "received remote confirmation for play data #$(msg.id)")
      put!(confirm, msg)

    elseif msg isa ContestConfirmation

      #log(name, "received remote confirmation for contest data #$(msg.id)")
      put!(confirm, msg)

    else

      log(name, "received unsupported request, this should not happen")

    end

  end

  try fetch(main)

    log(name, "connection to server lost")
    put!(stop, true)

  catch err

    if isready(stop)
      log(name, "received stop signal")
    elseif err isa InterruptException
      log(name, "got interrupted")
      put!(stop, true)
    elseif err isa TaskFailedException && err.task.exception isa InterruptException
      log(name, "got interrupted")
      put!(stop, true)
    else
      log(name, "received an unexpected exception: $err")
      put!(stop, true)
      throw(err.task.exception)
    end

  end

  fetch(stopper)
  log(name, "exiting...")

end

function upload(name, slot, stop, buffer, confirm)

  name = "$name-2"

  log(name, "upload coroutine initialized")

  stopper = @async begin
    fetch(stop)
    close(buffer)
    close(confirm)
    close(slot)
  end

  did = cid = 1

  main = @async while isopen(fetch(slot))

    record = fetch(buffer)

    if record isa InternalDataRecord

      log(name, "uploading record d.$did:D.$(record.reqid) received from w$(record.worker)...")

      dtime = @elapsed begin
        send(fetch(slot), sign_record(record, did))
        c = take!(confirm) :: DataConfirmation
      end
      
      if c.id == did
        log(name, "sent d.$did in $dtime seconds")
      else
        log(name, "received inconsistent confirmation id d.$(c.id) (expected d.$did)")
        break
      end

      did += 1

    elseif record isa InternalContestRecord

      log(name, "uploading contest record c.$cid:C.$(record.reqid)...")

      dtime = @elapsed begin
        send(fetch(slot), sign_record(record, cid))
        c = take!(confirm) :: ContestConfirmation
      end
      
      if c.id == cid
        log(name, "sent c.$cid in $dtime seconds")
      else
        log(name, "received inconsistent confirmation id $(c.id)")
        break
      end

      cid += 1

    else

      log(name, "received record of unknown type, this must not happen")
      break

    end

    take!(buffer)

  end

  try

    wait(main)
    put!(stop, true)

  catch err

    if isready(stop)
      log(name, "received stop signal")
    elseif err isa InterruptException
      log(name, "got interrupted")
      put!(stop, true)
    elseif err isa TaskFailedException && err.task.exception isa InterruptException
      log(name, "got interrupted")
      put!(stop, true)
    else
      log(name, "received an unexpected exception: $err")
      put!(stop, true)
      throw(err.task.exception)
    end

  end

  fetch(stopper)
  log(name, "exiting...")

end


function compute( name, stop, datareq, contestreq, buffer
                ; playings = 100
                , gpu = false
                , async = false
                , workers = Distributed.workers()
                , kwargs... )

  name = "$name-3"

  log(name, "compute coroutine initialized")

  # If we let the computations happen on the same process/thread that runs
  # the coroutines, we are getting blocked all the time. We thus delegate the
  # actual simulations to worker processes (threads do not work with Knet (yet)
  # and seem to be generally unstable)

  if workers == [1]

    @error "<$name-3> using the main process as worker is not supported"
    log(name, "exiting...")
    return false

  elseif !issubset(workers, Distributed.workers())

    @error "<$name-3> requested workers are not available"
    log(name, "exiting...")
    return false

  end

  m = length(workers)
  gpu_devices = Knet.cudaGetDeviceCount()

  if m > gpu_devices && gpu
    log(name, "info: $m workers will share $gpu_devices GPU device[s]")
  end

  # Print messages from the workers
   
  messages_local = Channel{Tuple{Int,String}}(100)
  messages = RemoteChannel(() -> messages_local)

  messager = @async for (worker, msg) in messages_local

    log(name, "w$worker says: $msg")

  end

  # Manage contest requests
  # Each worker gets its own channel to help prevent potential race conditions

  chcreate = () -> Channel{Tuple{Int, Int, ContestRequest}}(100)
  creqchannels = [RemoteChannel(chcreate) for _ in workers]
  packchannel = RemoteChannel(() -> Channel{InternalContestRecord}(3m))

  contests = @async begin
    worker = 0
    for creq in contestreq

      try
        # Subdivid the contest request in smaller contest that can be distributed
        # to the workers
        packs = subdivide(creq, playings)
        k = length(packs)

        log(name, "dividing request C.$(creq.id) with length $(creq.length) in $k subrequests")

        for (i, pack) in enumerate(packs)
          worker = (worker % m) + 1
          put!(creqchannels[worker], (i, k, pack))
        end

        # Collect the next k contest results and merge them before sending the
        # results to the buffer
        packresults = map(1:k) do _
          take!(packchannel)
        end

        dtime = sum(x -> x.dtime, packresults) / m # Average time per worker
        cdata = combine(packresults, dtime)
        l = sum(cdata.data)

        log(name, "collecting record of length $l for C.$(cdata.reqid)")
        put!(buffer, cdata)

      catch err
        log(name, "unexpected error in contest thread: $err")
        put!(stop, true)
        return
      end

    end
  end

  # Handle stop signals in the compute coroutine, closing all local and some
  # global channels

  stopper = @async begin
    fetch(stop)
    close(contestreq)
    close(datareq)
    close(packchannel)
    close(buffer)
    map(close, creqchannels)
  end

  promises = map(1:m) do i

    Distributed.@spawnat workers[i] begin

      callback = () -> isready(stop) && throw(WorkerExit())
      gpu && Knet.gpu((i-1) % gpu_devices)


      # In the workers we have to carefully handle exceptions, since even
      # a user interruption may emerge at any point in the code that follows.
      # If we would not communicate this to the outside (and to all other
      # workers), we might hang -> not good...

      @sync begin
        
        @async while !isready(stop)

          j = 0
          k = 0
          req = nothing

          try

            j, k, req = take!(creqchannels[i])

          catch err

            if err isa InterruptException
              put!(messages, (i, "got interrupted"))
              put!(stop, true)
            else
              put!(messages, (i, "received stop signal"))
              put!(stop, true)
            end

            break

          end

          l  = req.length
          id = req.id

          msg = "doing subcontest $j / $k of length $l for request C.$id..."
          put!(messages, (i, msg))

          players = get_player.(req.specs, gpu = gpu, async = async)

          dtime = @elapsed begin
            data = compete(players, req.length, req.active)
          end

          record = InternalContestRecord(data, req.id, i, dtime)
          put!(packchannel, record)

        end

        @async while !isready(stop)

          req = nothing

          try

            req = fetch(datareq)

          catch err

            if err isa InterruptException
              put!(messages, (i, "got interrupted"))
              put!(stop, true)
            else
              put!(messages, (i, "received stop signal"))
              put!(stop, true)
            end

            break

          end

          # Adapt number of playings to train server preferences

          k = clamp(playings, req.min_playings, req.max_playings)

          put!(messages, (i, "doing $k playings for D.$(req.id)..."))

          # Derive player and preparational / branching steps

          player = get_player(req.spec, gpu = gpu, async = async)

          psteps = req.prepare_steps[1]:req.prepare_steps[2]
          bsteps = req.branch_steps[1]:req.branch_steps[2]

          # Try to play the games, catching stop signals via the callback that
          # is called after each move of player

          try
            
            dtime = @elapsed begin
              datasets = record_self( player, k
                           ; augment = req.augment
                           , prepare = prepare(steps = psteps)
                           , branch = branch(prob = req.branch_prob, steps = bsteps)
                           , merge = false
                           , distributed = false
                           , callback_move = callback
                           , kwargs... )
            end

            l = length(datasets)
            k = sum(length, datasets)

            msg = "generated $k states in $(round(dtime, digits=2))s"
            put!(messages, (i, msg))

            # Sent the record to be uploaded

            record = InternalDataRecord(datasets, req.id, i, dtime)
            put!(buffer, record)

          # An exception in the code above could mean a WorkerExit, i.e., the
          # worker received a stop signal and stopped gracefully. It could
          # also be from an error in record_self (I'm looking at you, knet and cuda)
          # as well as an Interrupt event from the user (who knows how it got
          # here...)

          catch err

            if err isa WorkerExit
              put!(messages, (i, "received stop signal"))
            elseif err isa InterruptException
              put!(messages, (i, "got interrupted"))
              put!(stop, true)
            else
              put!(messages, (i, string(err)))
              put!(stop, true)
            end

            break

          end

        end
        
      end

    end

  end

  # Wait for all workers to finish
  
  try

    for t in promises wait(t) end

  # Hopefully, we have treated all exceptions in the workers,
  # so any exception received here should be an InterruptException. If it is
  # not, then consider it a bug and throw it / let hell break loose

  catch err

    !isa(err, InterruptException) && throw(err)

    log(name, "got interrupted")

    # Signal globally that the game is over

    put!(stop, true)

    # Wait for the remaining tasks

    for t in promises
      !isready(t) && wait(t)
    end

  end

  # When we arrive here, we have reason to assume that all should end

  !isready(stop) && put!(stop, true)

  try wait(contests)
  catch err

    if err isa InterruptException
      log(name, "got interrupted")

    # An exception here is to be expected if global stop is set while
    # the channels in creqchannels are overflowing. This should never happen
    # in realistic scenarios
    else
      log(name, "exception closing down contest channels: $err")
    end

  end

  close(messages)
  wait(messager) 
  wait(stopper)

  log(name, "exiting...")

  true

end

