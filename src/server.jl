
module Server

using ..Jtac

import Sockets: @ip_str, connect, listen
import Serialization

abstract type Msg end

send(sock, msg :: Msg) = Serialization.serialize(sock, msg)
receive(sock) :: Msg   = Serialization.deserialize(sock)

struct Instruction <: Msg

  # Composed reference model
  ref_model     :: Dict{Symbol}

  # Player options
  power         :: Int
  temperature   :: Float32
  exploration   :: Float32
  dilution      :: Float32
  name          :: String

  # Training options
  prepare_steps :: Tuple{Int, Int}
  branch_prob   :: Float64
  branch_steps  :: Tuple{Int, Int}
  augment       :: Bool

  # Dataset options
  min_playings  :: Int
  max_playings  :: Int

  # Hash of the instruction set
  hash          :: UInt64

  function Instruction( player :: MCTSPlayer{G}
                      ; augment = true
                      , prepare_steps = 0
                      , branch_prob = 0.
                      , branch_steps = 1
                      , min_playings = 1
                      , max_playings = 10000
                      ) where {G <: Game}

    @assert !isabstracttype(G)

    tup(a :: Int) = (a, a)
    tup(a) = (a[1], a[2])

    ref_model = base_model(player) |> to_cpu |> Jtac.decompose

    args = ( ref_model
           , player.power
           , player.temperature
           , player.exploration
           , player.dilution
           , player.name
           , tup(prepare_steps)
           , branch_prob
           , tup(branch_steps)
           , augment
           , min_playings
           , max_playings )

    new(args..., hash(args))

  end

end

function create_player(ins :: Instruction; gpu = false, async = false)
  ref_model = Jtac.compose(ins.ref_model)
  @assert ref_model isa NeuralModel

  model = gpu   ? ref_model |> to_gpu : ref_model
  model = async ? Async(model) : model

  MCTSPlayer( model
            , power = ins.power
            , temperature = ins.temperature
            , exploration = ins.exploration
            , dilution = ins.dilution
            , name = ins.name )
end


struct Record <: Msg
  ins_hash :: UInt64
  source   :: String
  number   :: Int
  datasets :: Vector
end

function Record(ins, datasets; source = "", number = 0)
  Record(ins.hash, source, number, datasets)
end

function client(
               ; ip = ip"127.0.0.1"
               , port = 7788
               , name = "unknown"
               , gpu = false
               , async = false
               , playings = 10
               , buffer_length = 10
               , kwargs...)

  ins_slot   = Channel(1)
  buffer     = Channel(buffer_length)

  while true

    train_socket = nothing

    try

      @info "<play> Trying to connect to train server $ip:$port."
      train_socket = connect(ip, port)
      @info "<play> Connection to train server $ip:$port established."

    catch err

      @info "<play> Connection to train server $ip:$port refused."
      @info "<play> Will try to connect again in 10 seconds..."
      sleep(10)

    end

    @sync begin

      try

        @async register_instructions(train_socket, ins_slot)
        @async upload_datasets(train_socket, buffer)
        td = @Threads.spawn record_datasets( train_socket, ins_slot, buffer
                              , name = name
                              , gpu = gpu
                              , async = async
                              , playings = playings
                              , kwargs... )

      catch err

        Base.throwto(td, InterruptException())

        if err isa Base.IOError

          @error "<play> Connection to $ip:$port was closed unexpectedly."
          @info "<play> Will try to reconnect in 10 seconds..."
          sleep(10)

        else

          throw(err)

        end

      end

    end

  end

  @info "<play> Connection to $ip:$port was closed."

end


function register_instructions(sock, slot)

  @info "<play:1> Download task initialized on thread $(Threads.threadid())"

  # Wait until a new instruction arrives
  while !eof(sock)

    # Read the instruction
    ins = receive(sock) :: Instruction

    @info "<play:1> Received instruction $(ins.hash)"
    
    # Another instruction was already provided but not used
    # In this case, replace it
    if isready(slot)
      ins_old = take!(slot)
      @info "<play:1> Cleared expired instruction $(ins_old.hash) from active slot"
    end

    put!(slot, ins)
    @info "<play:1> Placed instruction $(ins.hash) in active slot"
    @info "<play:1> Waiting for new instructions..."

  end

end

function upload_datasets(sock, buffer)

  @info "<play:2> Upload task initialized on thread $(Threads.threadid())"

  while isopen(sock) && iswritable(sock)

    n, record = take!(buffer)
    
    if isopen(sock) && iswritable(sock)
      send(sock, record)
      @info "<play:2> Sent record $n to train server"
    else
      break
    end

  end

end


function record_datasets( sock, slot, buffer
                        ; playings = 10
                        , name = "unknown"
                        , gpu = false
                        , async = false
                        , kwargs... )

  @info "<play:3> Compute task initialized on thread $(Threads.threadid())"

  ins = nothing
  n   = 0

  while isopen(sock) && isopen(buffer)

    n += 1

    if isready(slot) || isnothing(ins)
      ins = take!(slot)
      @info "<play:3> Accepted instruction set $(ins.hash) from active slot"
    end

    if playings < ins.min_playings || playings > ins.max_playings
      m = clamp(playings, ins.min_playings, ins.max_playings)
      @info "<play:3> Adapt number of playings to instructions: playings = $m"
    else
      m = playings
    end

    player = create_player(ins, gpu = gpu, async = async)

    @info "<play:3> Generating dataset $n..."

    psteps = ins.prepare_steps[1]:ins.prepare_steps[2]
    bsteps = ins.branch_steps[1]:ins.branch_steps[2]

    datasets = record_self( player, m
                 ; augment = ins.augment
                 , prepare = prepare(steps = psteps)
                 , branch = branch(prob = ins.branch_prob, steps = bsteps)
                 , merge = false
                 , kwargs... )

    len = length(datasets)
    states = sum(length, datasets)

    #len = 0
    #states = 0
    #sleep(5)
    #datasets = []

    record = Record(ins, datasets, source = name, number = n)
    @info "<play:3> Generated record $n with $len playings and $states states"

    put!(buffer, (n, record))

  end

end


end # module Server


#=

# -------- Training Task ----------------------------------------------------- #

function collect_games!(buffer, rb_queue, rbmax)

  sets = 0

  while isready(rb_queue)

    games = take!(rb_queue)

    # Get the length of every dataset in buffer and the length of the fresh games
    sizes, s = length.(buffer), length(games)

    # If there is not enough space in the buffer left, remove some entries first
    if sum(sizes) + s > rbmax

      k = findfirst(x -> x > s, cumsum(sizes))
      foreach(j -> deleteat!(buffer, j), 1:k)

    end

    push!(buffer, games)

    sets += 1

  end

  sets

end

function train_thread( player
                     , player_queue
                     , rb_queue
                     , rbmax
                     , rbmin
                     , steps_renewal
                     , selfplay_training_ratio
                     ; kwargs... )

  # The number of steps played since the playing model was last renewed
  steps = 0

  # The total number of training steps
  total_steps = 0

  # The total number of game datasets collected from clients 
  total_sets = 0

  # Replay buffer that collects the datasets recorded by the clients. The
  # datasets in this variable are available for training. As more sets are
  # generated over time, the replay buffer is updated: old games (played with
  # potentially older models) are replaced by more current ones.
  replay_buffer = DataSet[]

  while true

    # Update the replay buffer if clients pushed some new games
    total_sets += collect_games!(replay_buffer, rb_queue, rbmax)

    # Train the model for some steps
    steps += train_player!(player, replay_buffer, rbmin; kwargs...)
    total_steps += steps

    # After a certain number of training steps, we employ a new, hopefully
    # improved, player for self-playing
    if steps > steps_renewal

      # Serialize the player and put it in the player_queue. There, it
      # is managed by the server: If desired, a backup copy is saved,
      # and it is propagated to all clients as current player.
      put!(player_queue, _serialize(player))
      steps = 0

    end

  end

end


# -------- Playing Task ------------------------------------------------------ #

send_player(sock, player) = Serialization.serialize(sock, player)

function listen_to_client(c, sock, rb_queue)
  
  try

    while !eof(sock)

      ds = Serialization.deserialize(sock)
      put!(rb_queue, ds)

      @info "<play> Received set from client $c."

    end


  catch err

    @info "<play> Error reading from client $c: $err."
    close(sock)

  end

  ip, port = getsockname(sock)
  @info "<play> Client $c from $ip:$port has disconnected."

end

function play_task(player, player_queue, rb_queue, ip, port)

  clients = []
  current_player = Ref{Any}(_serialize(player))

  server = Sockets.listen(ip, port)
  @info "<play> Listening to $ip:$port."

  @sync begin

    # Accepting incoming client connections
    @async begin

      c = 0

      while true

        sock = Sockets.accept(server)
        ip, port = getsockname(sock)

        c += 1

        @info "<play> Client $c from $ip:$port connected."
        push!(clients, (c, sock))

        @info "<play> Sending current player to client $c."
        send_player(sock, current_player[])

        @async listen_to_client(rb_queue, sock)
        @info "<play> Begin listening to client $c for datasets."

      end

    end

    # Look for new players and tell the clients about them
    @async while true

      player = take(player_queue)
      current_player[] = player

      @info "<play> Received new player from <train>."

      foreach(c -> send_player(c[2], player), clients)

      @info "<play> Sent new player to clients $(first.(clients))."
      
    end

  end

end

# -------- Server ------------------------------------------------------------ #

function jtac_server( player :: MCTSPlayer{G}
                    ; buffer = 500000  # replay buffer size
                    , ip = ip"::1"
                    , port :: Int = 5112
                    , checkpoint :: String = ""
                    ) where {G <: Game}

  @sync begin

    # Channels to communicate between training and playing
    rb_queue = Channel{DataSet{G}}(10)
    player_queue = Channel(1)

    # Start the training activity in the background
    @info "Launch training thread..."
    train_task = @async train_thread(player, player_queue, rb_queue, args...)

    # Start the server, listen to the tcp socket for clients, and manage them
    @info "Launch playing thread..."
    play_task = @async play_thread(args...)

  end

end

=#
