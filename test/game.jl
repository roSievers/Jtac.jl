
@testset "Status" begin
  vals = [1, 0, -1]
  @test Game.is_over(Game.Status()) == false
  @test Game.Status.(vals) .|> Game.is_over |> all
  @test Game.Status.(vals) == vals
  @test (Game.with_default.(Game.Status(), vals) .== vals) |> all
  @test Game.with_default.(Game.Status.(vals), Game.Status()) == vals
end

@testset "Augment" begin
  hmirror = Util.hmirror
  vmirror = Util.vmirror
  dmirror = Util.dmirror
  a = rand(25)
  ma = reshape(a, (5, 5))
  @test all(ma |> hmirror |> hmirror .== ma)
  @test all(ma |> vmirror |> vmirror .== ma)
  @test all(ma |> dmirror |> dmirror .== ma)
  for i in [1, 7, 15, 23]
    @test a[i] == reshape(hmirror(ma), (25,))[hmirror(i, (5,5))]
    @test a[i] == reshape(vmirror(ma), (25,))[vmirror(i, (5,5))]
    @test a[i] == reshape(dmirror(ma), (25,))[dmirror(i, (5,5))]
  end
end

@testset "TicTacToe" begin

  game = Game.TicTacToe()
  @test size(Game.TicTacToe) == size(game) == size(Game.array(game)) == (3, 3, 1)
  @test size(Game.array([game, game])) == (3, 3, 1, 2)
  @test Game.policy_length(Game.TicTacToe) == Game.policy_length(game) == 9
  @test Game.current_player(game) == 1
  @test_throws BoundsError Game.apply_action!(game, -1)

  Game.apply_action!(game, 5)
  @test Game.legal_actions(game) == [ i for i in 1:9 if i != 5]
  @test Game.current_player(game) == -1
  @test Game.random_action(game) in Game.legal_actions(game)

  Game.apply_action!(game, 3)
  @test .!Game.is_action_legal.(game, [3, 5]) |> all
  @test Game.is_action_legal.(game, Game.legal_actions(game)) |> all
  @test Game.status(game) == Game.Status()
  @test copy(game).board == game.board
  @test_throws AssertionError Game.apply_action!(game, 5)

  game = Game.random_playout(game)
  @test Game.legal_actions(game) == []
  @test Game.is_over(game) == Game.is_over(Game.status(game))

  games, policies = Game.augment(game, rand(Float32, 9))
  @test length(games) == length(policies) == 8

end

