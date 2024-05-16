
@testset "Status" begin
  vals = [1, 0, -1]
  @test Game.isover(Game.undecided) == false
  @test Game.Status.(vals) .|> Game.isover |> all
  @test Game.Status.(vals) == [Game.win, Game.draw, Game.loss]
end

@testset "TicTacToe" begin
  game = Game.TicTacToe()
  @test size(Game.TicTacToe) == size(game) == size(Game.array(game)) == (3, 3, 1)
  @test size(Game.array([game, game])) == (3, 3, 1, 2)
  @test Game.policylength(Game.TicTacToe) == Game.policylength(game) == 9
  @test Game.mover(game) == 1
  @test_throws BoundsError Game.move!(game, -1)

  Game.move!(game, 5)
  @test Game.legalactions(game) == [ i for i in 1:9 if i != 5]
  @test Game.mover(game) == -1
  @test Game.randomaction(game) in Game.legalactions(game)

  Game.move!(game, 3)
  @test .!Game.isactionlegal.(game, [3, 5]) |> all
  @test Game.isactionlegal.(game, Game.legalactions(game)) |> all
  @test Game.status(game) == Game.undecided
  @test copy(game).board == game.board
  @test_throws AssertionError Game.move!(game, 5)

  game = Game.rollout(game)
  @test Game.legalactions(game) == []
  @test Game.isover(game) == Game.isover(Game.status(game))

  games, policies = Game.augment(game, rand(Float32, 9))
  @test length(games) == length(policies) == 8
end

@testset "MetaTac" begin
  game = Game.MetaTac()
  @test size(Game.MetaTac) == size(game) == size(Game.array(game)) == (9, 9, 3)
  @test size(Game.array([game, game])) == (9, 9, 3, 2)
  @test Game.policylength(Game.MetaTac) == Game.policylength(game) == 81
  @test Game.mover(game) == 1
  @test_throws BoundsError Game.move!(game, -1)

  Game.move!(game, 5)
  @test Game.mover(game) == -1
  @test Game.randomaction(game) in Game.legalactions(game)

  Game.randommove!(game)
  @test .!Game.isactionlegal.(game, [3, 5]) |> all
  @test Game.isactionlegal.(game, Game.legalactions(game)) |> all
  @test Game.status(game) == Game.undecided
  @test copy(game).board == game.board
  @test_throws AssertionError Game.move!(game, 5)

  game = Game.rollout(game)
  @test Game.legalactions(game) == []
  @test Game.isover(game) == Game.isover(Game.status(game))

  games, policies = Game.augment(game, rand(Float32, 81))
  @test length(games) == length(policies) == 8
end
