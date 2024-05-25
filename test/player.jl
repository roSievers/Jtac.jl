  @testset "Player" begin
    G = ToyGames.TicTacToe
    model = Model.RolloutModel(G)
    for player in [
      Player.RandomPlayer(G),
      Player.IntuitionPlayer(model),
      Player.MCTSPlayer(model, power = 10),
      Player.MCTSPlayerGumbel(model, power = 10),
    ]
      @test Player.name(player) isa String

      p = Player.think(player, G())
      @test p isa Vector{Float32}
      @test length(p) == Game.policylength(G)
      @test Player.decide(player, G()) isa Game.ActionIndex
      @test Player.decideturn(player, G()) isa Vector{Game.ActionIndex}
      @test Player.move!(G(), player) isa G
      @test Player.turn!(G(), player) isa G
      @test Player.ntasks(player) == Model.ntasks(player) == 1
      @test Player.gametype(player) == Model.gametype(player) == G

      bytes = Pack.pack(player)
      player2 = Pack.unpack(bytes, Player.AbstractPlayer)
      @test typeof(player) == typeof(player2)
    end
  end

  @testset "Ranking" begin
    G = ToyGames.TicTacToe
    model = Model.RolloutModel(G)
    players = [Player.MCTSPlayer(model; power) for power in [10, 100, 500]]
    rk = Player.rank(players, 50)
    @test rk isa Player.Ranking
    rk = Player.rankmodels([model, model], 10)
    @test rk isa Player.Ranking
  end
