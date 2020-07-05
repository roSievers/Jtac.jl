
"""
Context in which the training of a player takes place. Contains information
like the learning rate, batchsize, or power.
"""
struct Context

  # Player options for dataset generation
  name :: String
  power :: Int
  temperature :: Float32
  exploration :: Float32
  dilution :: Float32

  # Other options for dataset generation
  prepare_steps :: Tuple{Int, Int}
  branch_prob :: Float64 
  branch_steps :: Tuple{Int, Int}
  augment :: Bool

  min_playings :: Int
  max_Playings :: Int

  # Selecting training data from the DataPool
  # Epoch: subset of the pool used for one training iteration
  epochsize :: Int
  testmod :: Int   # every testmod playing is added to testpool instead of trainpool
  max_age :: Int
  max_use :: Int
  min_quality :: Int
  weight_age :: Float64 # Number between 0 and 1 to weight age vs. usage to
                        # assess the quality of a dataset
                        
  # Maximal number of game states stored in the test and trainpools
  capacity :: Int

  # Era: games used for training until the reference model gets its next update
  erasize :: Int

  # Options for training after an epoch has been selected from the DataPool
  batchsize :: Int
  lr :: Float32

  # Folder where backups of the model are stored
  backup_folder :: String
  backups :: Int

end

function Context(
                ; name = "anonymous"
                , power = 50
                , temperature = 1.
                , exploration = 1.41
                , dilution = 0.
                , prepare_steps = (0, 0)
                , branch_prob = 0.
                , branch_steps = (0, 0)
                , augment = true
                , min_playings = 1
                , max_playings = 1000
                , max_age = 3
                , max_use = 3
                , weight_age = 0.5
                , min_quality = 0.
                , testmod = 10
                , epochsize = 5000
                , erasize = 20000
                , capacity = erasize * max_age
                , batchsize = 512
                , lr = 1e-2
                , backup_folder = "./"
                , backups = 1 )

  Context( name, power, temperature, exploration, dilution,
           prepare_steps, branch_prob, branch_steps,
           augment, min_playings, max_playings, epochsize, testmod,
           max_age, max_use, min_quality, weight_age, capacity, erasize,
           batchsize, lr, backup_folder, backups )
end

# Derive a data-request based on a model and a training context
function DataRequest(id, ctx, model :: Model{<:Game, false})

  player = MCTSPlayer( model, power = ctx.power
                     , temperature = ctx.temperature
                     , exploration = ctx.exploration
                     , dilution = ctx.dilution
                     , name = ctx.name * "-$id" )

  DataRequest( id, player
             ; augment = ctx.augment
             , prepare_steps = ctx.prepare_steps
             , branch_prob = ctx.branch_prob
             , branch_steps = ctx.branch_steps
             , min_playings = ctx.min_playings
             , max_playings = ctx.max_playings )

end


"""
Pool of data that is accumulated by collecting the results of play services.
"""
mutable struct DataPool{G <: Game}
  data :: DataSet{G}
  age  :: Vector{Int}
  use  :: Vector{Int}
  reqid :: Vector{Int}
  capacity :: Int
end

DataPool{G}(c :: Int) where {G <: Game} = DataPool{G}(G[], Int[], Int[], Int[], c)

Base.length(dp :: DataPool) = length(dp.age)

function Base.getindex(dp :: DataPool{G}, I) where {G <: Game}
  DataPool{G}(dp.data[I], dp.age[I], dp.use[I], dp.reqid[I])
end

"""
    add!(datapool, dataset, id, history)

Add a `dataset` generated for a specific request `id` to a `datapool`. The
`history` of request ids is used to calculate the age of the added dataset.
"""
function add!(dp :: DataPool{G}, ds :: DataSet{G}, reqid :: Int, history :: Vector{Int}) where {G <: Game}

  index = findfirst(isequal(reqid), history)

  if isnothing(index)

    false

  else

    k = length(ds)
    age = length(history) - index

    append!(dp.data, ds)
    append!(dp.use, fill(0, k))
    append!(dp.age, fill(age, k))
    append!(dp.reqid, fill(reqid, k))

    true

  end

end

"""
    quality(datapool, context)
    quality(datapool, context, sel)

Calculate the quality vector of `datapool` in a given training `context`.
If the argument `sel` is provided, only the quality in the respective selection
is calculated.
"""
function quality(dp :: DataPool, ctx :: Context)
  age = max.(0, ctx.max_age - dp.age) / ctx.max_age
  use = max.(0, ctx.max_use - dp.use) / ctx.max_use
  ctx.weight_age * age + (1-ctx.weight_age) * use
end

quality(dp :: DataPool, ctx :: Context, sel) = quality(dp[sel], ctx)

"""
    select_epoch(datapool, context, n)

Randomly select `n` entries (on average) from `datapool`, weighted by their
quality under `context`.
"""
function select_epoch(dp :: DataPool, ctx :: Context)
  n = ctx.epochsize
  q = quality(dp, ctx)
  k = length(dp)
  r = n / sum(q) / k
  
  indices = findall(rand(k) < r*q)
  indices, dp.ds[indices]
end

"""
    load(datapool)

Calculate the fraction of states in the `datapool` to its maximal capacity.
"""
load(dp :: DataPool) = ctx.length(dp) / ctx.capacity


"""
    clear!(datapool, context)

Clear entries with quality = 0 from `datapool` in a given `context`. If the
capacity of the pool is exceeded, also entries with quality > 0 will be removed.
"""
function clear!(dp :: DataPool, ctx :: Context)

  # Select all indices with positive quality
  q = quality(dp, ctx)
  indices = findall(x -> x > 0, q)

  # In this loop, we iteratively add entries to 'indices' if the capacity of the
  # dataset is exceeded
  while length(indices) > dp.capacity
    
    # Get the minimal quality value in the pool that is not yet flagged for
    # removal
    mq = findmin(q[indices])[1]

    # Find all indices with better quality
    indices = findall(x -> x > mq, q)

    # If these indices do not exhaust the capacity of the pool, we add a random
    # selection of data with quality = mq
    if length(indices) < dp.capacity
      l = dp.capacity - length(indices)
      idx = findall(isequal(mq), q)
      append!(indices, idx[randperm(1:length(idx))[1:l]])
    end

  end

  # TODO: Maybe this function is not efficient, have to benchmark
  # TODO: Add feedback value if we are removing many useful entries or not
  dp.data  = dp.data[indices]
  dp.age   = dp.age[indices]
  dp.use   = dp.use[indices]
  dp.reqid = dp.requid[indices]

end

"""
    update_age!(datapool, history)

Update the age entries of a `datapool` for a given `history` of request ids.
This function will throw exceptions if the datapool contains entries with ids
that are not in the history.
"""
function update_age!(dp :: DataPool, history :: Vector{Int})

  len = length(history)
  agemap = Dict{Int, Int}((id, len - i) for (i,id) in enumerate(history))

  for i in eachindex(dp.age)
    dp.age[i] = agemap[dp.reqid[i]]
  end

end

"""
    update_use!(datapool, sel)

Increment the use counter for the selection `sel` in `datapool`.
"""
update_use!(dp :: DataPool, indices :: Vector{Int}) = (dp.use[indices] .+= 1)

"""
    update_capacity!(datapool, context)

Adjust the capacity of `datapool` to `context`.
"""
update_capacity!(dp :: DataPool, ctx :: Context) = (dp.capacity = ctx.capacity)


"""
Auxiliary type that contains information about a connected play service
"""
struct PlayServiceInfo
  info :: PlayLogin
  sock :: TCPSocket
  id :: UInt
end


"""
    start(Jtac.Service.Train, model, context; <keyword arguments>)

Start a jtac train service with initial model `model` and context `context`.
"""
function start( :: Type{Train}
              , m :: NeuralModel{G, false}
              , ctx = nothing
              ; ip = ip"127.0.0.1"
              , port = 7788
              , name = ENV["USER"]
              , token = ""
              , max_clients = 100 ) where {G <: Game}

  # The context has to be available on all threads and workers, so place
  # it into a remote channel

  context = RemoteChannel(() -> Channel{Context}(1))

  if isnothing(ctx)
    @info "No context specified. Using fallback context."
    put!(context, Context())
  else
    put!(context, ctx)
  end

  # Changes of context, e.g., from manual intervention 

  context_change = RemoteChannel(() -> Channel{Context}(10))

  # The data request channel, which is updated whenever an era passes or the
  # context changes

  request = RemoteChannel(() -> Channel{DataRequest}(1))

  # The model to train is also stored in a remote channel, since the 
  # train worker will need access to it. This channel shall never
  # be taken from, only fetched!

  refmodel = RemoteChannel(() -> Channel{NeuralModel{G, false}}(1))
  put!(model, m)

  # The stream of ingoing data records by connected play services

  datarecords = RemoteChannel(() -> Channel{DataRecord}(100))

  # Global stop signal that is set to true if any problem occur
  globalstop = RemoteChannel(() -> Channel{Bool}(1))
  put!(globalstop, false)

  # List of currently connected play services

  clients  = PlayServiceInfo[]

  # This thread accepts client connections, sends them data and contest
  # requests, and accepts records
  client_task = @async manage_clients( ip, port, name, token, max_clients
                                     , clients, request, datarecords, globalstop )

  train_task = @async train_model(context, refmodel, request, datarecords, globalstop)

end

function train_model(context, context_change, refmodel, request, datarecords, globalstop)

  log = msg -> println("<train> $msg")

  # Store of all past requests in a vector that the remote worker can also
  # access
  req_history = SharedArrays.SharedVector{DataRequest}(0)

  # Request id that is incremented for each request update
  id = RemoteChannel(() -> Channel{Int}(1))
  put!(id, 1) 

  # Read and print messages from the worker
  msgs = RemoteChannel(() -> Channel{String}(100))
  log_task = @async while !fetch(globalstop)
    log(take!(msgs))
  end

  commit_request = msg -> begin
    i = take!(id)
    req = DataRequest(i, fetch(context), fetch(refmodel))
    push!(req_history, req)
    put!(request, req)
    put!(msgs, "commited request D.$i: $msg")
    put!(id, i + 1)
  end

  commit_request("initial request")

  # React to context changes from other sources, e.g., interactive changes
  context_task = @async while !fetch(globalstop)
    ctx = take!(context_change)
    isready(context) && take!(context)
    put!(context, ctx)
    commit_request("context was altered")
  end

  # Actual dataset handling and training takes place on a worker
  train_task = Distributed.@spawn begin

    put!(msgs, "worker: initialized")

    # Pools that will collect all datasets for training and testing
    trainpool = DataPool{G}()
    testpool  = DataPool{G}()

    # Initial fetch of the reference model
    model = fetch(refmodel)

    # Every playing with testcount % context.testmod == 0 is added to the
    # testpool instead of the trainpool
    testcount = 0
    epoch = 0
    era = 1
    eracount = 0

    # One iteration through the following loop is one training epoch

    while !fetch(globalstop)

      # Fix a context for this epoch
      ctx = fetch(context)

      # Fix the request history by collecting the ids
      # These ids are used to calculate the age of a point in the dataset
      hist = map(x -> x.id, req_history)

      # Collect records that have arrived during the last epoch
      while isready(datarecords)

        rec = take!(datarecords)
        
        datasets = decompress(rec._data)
        put!(msgs, "worker: decompressed d.$(rec.id) with $(length(datasets)) datasets")
        
        if !(rec.reqid in hist)

          put!(msgs, "worker: discarded d.$(rec.id) due to unknown request id D.$(rec.reqid)")

        else

          # Add datasets to the train- and testpools
          for ds in datasets 

            pool = testcount % ctx.testmod == 0 ? testpool : trainpool
            add!(pool, ds, rec.reqid, hist)
            testcount += 1

          end

          put!(msgs, "worker: added data from d.$(rec.id) to the train- and testpools")

        end

      end

      # Update and clean the train and testpools
      for pool in [trainpool, testpool]
        update_age!(pool, hist)
        update_capacity!(pool, ctx)
        clear!(pool, ctx)
      end

      # Only train the model if the trainpool is large enough, with a sufficient
      # minimal quality
      
      tsize = length(trainpool)
      qavg  = mean(quality(trainpool))

      if tsize <= ctx.epochsize || qavg < ctx.min_quality

         if tsize <= ctx.epochsize
           put!(msgs, "worker: training pool too small ($tsize)")
         else
           put!(msgs, "worker: quality of training pool too low ($qavg)")
         end

         put!(msgs, "worker: waiting for new data before training continues")

         # Wait until fresh datarecords are available
         while !isready(datarecords) sleep(1) end

      else

        # Select a training epoche
        idx, trainset = select_epoch(trainpool, ctx)

        # Set the 'istraining' flag
        isready(istraining) && take!(istraining)
        put!(istraining, true)

        # Iterate over the data once
        train!( model, trainset
              ; loss = ctx.loss, epochs = 1, batchsize = ctx.batchsize
              , optimizer = nothing )

        # Increment the usage count of the selected data
        update_usage!(trainpool, ctx)
        epoch += 1

        put!(msgs, "worker: epoch $epoch finished")

        # Check if a new era should be ushered in
        eracount += epochsize

        if eracount >= erasize

          # Increment the era and reset the count
          era += 1
          eracount = 0

          # Place the current model as reference model
          isready(refmodel) && take!(refmodel)
          put!(refmodel, model |> to_cpu)

          # Commit a new data request
          commit_request("era $era started")

          # Save the model to the backup folder
          name = "$(ctx.name)-$era"
          save_model(model |> to_cpu, joinpath(ctx.backup_folder, name))

          # Remove possible backup files that are old
          if era - ctx.backups >= 1
            name = "$(ctx.name)-$(era - ctx.backups)"
            path = joinpath(ctx.backup_folder, name)
            
            if Base.Filesystem.isfile(path)
              Base.Filesystem.rm(path)
            end

          end

        end

      end

    end

  end

end

function manage_clients( ip, port, name, token, max_clients
                       , clients, request, datarecords )

  @assert isempty(clients)

  println("Listening for services to connect")
  server  = listen(ip, port)
  req     = Ref(fetch(request)) # current data request

  # Function to handle play services that have succesfully connected
  handle_play_service = client -> begin

    # Initial request
    if !isnothing(req[]) && client.info.accept_data_requests
      println("Sending initial data request to P.$(client.info.name)")
      send(client.sock, req[])
    end

    # Handle messages from this client
    while true
      msg = receive(client.sock) <: Msg{Play, Train}

      if msg isa PlayLogout
        println("Service P.$(client.info.name) has logged out")
        filter!(c -> c.id != client.id, clients)
        close(client.sock)

      elseif msg isa DataRecord
        println("Received data record from P.$(client.info.name) for D.$(msg.reqid)")
        send(client.sock, DataConfirmation(msg.id))
        put!(datarecords, msg)

      elseif msg isa ContestRecord
        error("Contest records not yet implemented")
      end
    end

  end


  @sync begin

    # Accept and handle connections from other services
    @async while !fetch(globalstop)

      sock = accept(server)
      login = receive(sock) <: PlayLogin  # TODO: other services

      reject = msg -> begin
        println("Login of service P.$(login.name) failed: $msg")
        send(sock, PlayAccept(false, msg))
        close(sock)
      end

      # Login unsuccessful
      if login.version != JTAC_SERVICE_VERSION reject("wrong jtac service version")
      elseif login.token != token              reject("wrong token")
      elseif length(clients) >= max_clients    reject("too many clients")

      # Login successful
      else
        println("Play service $(login.name) logged in sucessfully")
        send(sock, PlayAccept(true, "Welcome $(login.name)! Thanks for supporting $name."))

        client = PlayServiceInfo(login, sock, rand(UInt))
        println("Service P.$(client.info.name) was assigned the service id $(client.id)")
        push!(clients, client)

        @async handle_play_service(client)
      end

    end

    # React when the data request is changed in another thread
    @async while !fetch(globalstop)

      req[] = take!(request)
      println("Sending new request D.(req[].id) to all connected clients")

      map(clients) do c
        login.accept_data_requests && send(c.sock, req[])
      end

    end

  end

end



