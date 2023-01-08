
using Jtac
using Test
using Random
using Sockets

import .Server.Events
import .Server.Config
import .Server.Api

@testset "Config" begin
  def = Config.default()
  cfg = Config.load("config.toml")
  diff = setdiff(keys(def), keys(cfg))
  @test isempty(diff)
  for key in keys(def)
    diff = setdiff(keys(def[key]), keys(cfg[key]))
    @test isempty(diff)
  end
  @test Config.set_param!(cfg, "server.port", 1000)
  @test cfg[:server][:port] == 1000
end

function create_events()
  G = Game.TicTacToe
  ranking =
    Player.rank([Player.MCTSPlayer(;power) for power in [10, 100]], 5, instance = G)

  evs = Events.Event[]
  push!(evs, Events.StartSession(session_id = UInt(12000), session_name = "test"))
  push!(evs, Events.StopSession(session_id = UInt(12000), session_name = "test"))
  push!(evs, Events.LoginClient(client_id = rand(UInt), client_name = "tester"))
  push!(evs, Events.QueryStatus(client_id = rand(UInt)))
  push!(evs, Events.QueryHistory(client_id = rand(UInt), starting_at = 0, max_entries = -1))
  push!(evs, Events.QueryGeneration(client_id = rand(UInt), wait = true))
  push!(evs, Events.QueryModel(client_id = rand(UInt), generation = 5))
  push!(evs, Events.UploadData(client_id = rand(UInt), generation = 5, length = 100))
  push!(evs, Events.UploadContest(client_id = rand(UInt); ranking))
  push!(evs, Events.SetParam(client_id = rand(UInt), param = ["server.gpu"], value = [3]))
  push!(evs, Events.StartTraining())
  push!(evs, Events.StopTraining())
  push!(evs, Events.StepTraining( batchsize = 5012
                          , stepsize = 35
                          , targets = ["value", "policy", "l1reg"]
                          , weights = [5.0, 1.0, 0.01]
                          , values = [0.28, 1.33, 0.65]))
  push!(evs, Events.StartRecording())
  push!(evs, Events.StopRecording())
  push!(evs, Events.StepRecording(length = 1284))
  evs
end

@testset "Event" begin
  evs = create_events()

  for ev in evs
    p = Pack.pack(ev)
    ev_ = Pack.unpack(p, Events.Event)
    p_ = Pack.pack(ev_)
    @test all(p .== p_)
  end

  p = Pack.pack(evs)
  evs_ = Pack.unpack(p, Vector{Events.Event})
  p_ = Pack.pack(evs_)
  @test all(p .== p_)
end



@testset "Api" begin

  G = Game.TicTacToe
  model = Model.NeuralModel(G, Model.@chain G Dense(50, "relu"))
  player = Player.MCTSPlayer(model)

  ranking =
    Player.rank([Player.MCTSPlayer(;power) for power in [10, 100]], 5, instance = G)

  ds = Player.record(Player.MCTSPlayer(), instance = G)

  respond(:: Api.Login) = Api.LoginRes(client_id = 25)
  respond(:: Api.Logout) = Api.LogoutRes(success = true)
  respond(:: Api.QueryStatus) =
    Api.QueryStatusRes(session_id = 17, clients = ["jtac", "me"], state = :training, generation = 5)
  respond(:: Api.QueryHistory) = Api.QueryHistoryRes(history = create_events())
  respond(:: Api.QueryGeneration) = Api.QueryGenerationRes(generation = 3)
  respond(:: Api.QueryModel) = Api.QueryModelRes(; model, generation = 7)
  respond(:: Api.QueryPlayer) = Api.QueryPlayerRes(; player, generation = 5)
  respond(:: Api.UploadData) = Api.UploadDataRes(success = true)
  respond(:: Api.UploadContest) = Api.UploadContestRes(success = true)
  respond(:: Api.SetParam) = Api.SetParamRes(success = true)
  respond(:: Api.StopTraining) = Api.StopTrainingRes(success = true)
  respond(:: Api.StartTraining) = Api.StartTrainingRes(success = true)
  respond(:: Api.StopRecording) = Api.StopRecordingRes(success = true)
  respond(:: Api.StartRecording) = Api.StartRecordingRes(success = true)

  function dummyserver(server)
    while isopen(server)
      try
        sock = Sockets.accept(server)
        msg = Pack.unpack(sock, Api.Action)
        Server.Api.respond(sock, respond(msg))
      catch err
        if !(err isa IOError)
          @show err
        end
      end
    end
  end

  function dummyclient(server)
    res = Api.login(7238, client_name = "abc", password = "verysafe")
    @test res isa Api.LoginRes
    @test res.client_id == 25
    res = Api.logout(7238, client_id = 25)
    @test res isa Api.LogoutRes
    @test res.success == true
    res = Api.query_status(7238, client_id = 25)
    @test res isa Api.QueryStatusRes
    @test res.session_id == 17
    res = Api.query_history(7238, client_id = 25)
    @test res isa Api.QueryHistoryRes
    @test length(res.history) == length(create_events())
    res = Api.query_generation(7238, client_id = 25)
    @test res isa Api.QueryGenerationRes
    @test res.generation == 3
    res = Api.query_model(7238, client_id = 25)
    @test res isa Api.QueryModelRes
    @test res.model isa Model.NeuralModel
    res = Api.query_player(7238, client_id = 25)
    @test res isa Api.QueryPlayerRes
    @test res.generation == 5
    res = Api.upload_data(7238, client_id = 25, generation = 7, data = ds)
    @test res isa Api.UploadDataRes
    @test res.success == true
    res = Api.upload_contest(7238, client_id = 25, ranking = ranking)
    @test res isa Api.UploadContestRes
    @test res.success == true
    res = Api.set_param(7238, client_id = 25, param = ["a", "b"], value = [1, false])
    @test res isa Api.SetParamRes
    @test res.success == true
    res = Api.start_training(7238, client_id = 25)
    @test res isa Api.StartTrainingRes
    @test res.success == true
    res = Api.stop_training(7238, client_id = 25)
    @test res isa Api.StopTrainingRes
    @test res.success == true
    res = Api.start_recording(7238, client_id = 25)
    @test res isa Api.StartRecordingRes
    @test res.success == true
    res = Api.stop_recording(7238, client_id = 25)
    @test res isa Api.StopRecordingRes
    @test res.success == true
    close(server)
  end

  server = Sockets.listen(7238)

  @async dummyserver(server)
  dummyclient(server)

end

