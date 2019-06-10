
@testset "Status" begin
  vals = [1, 0, -1]
  @test Jtac.is_over(Jtac.Status()) == false
  @test Jtac.Status.(vals) .|> Jtac.is_over |> all
  @test Jtac.Status.(vals) == vals
  @test (Jtac.with_default.(Jtac.Status(), vals) .== vals) |> all
  @test Jtac.with_default.(Jtac.Status.(vals), Jtac.Status()) == vals
end

@testset "Augment" begin
  hmirror = Jtac.hmirror
  vmirror = Jtac.vmirror
  dmirror = Jtac.dmirror
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

  game = TicTacToe()
  @test size(TicTacToe) == size(game) == size(representation(game)) == (3, 3, 1)
  @test size(representation([game, game])) == (3, 3, 1, 2)
  @test policy_length(TicTacToe) == policy_length(game) == 9
  @test current_player(game) == 1
  @test_throws BoundsError apply_action!(game, -1)

  apply_action!(game, 5)
  @test legal_actions(game) == [ i for i in 1:9 if i != 5]
  @test current_player(game) == -1
  @test Jtac.random_action(game) in legal_actions(game)

  apply_action!(game, 3)
  @test .!Jtac.is_action_legal.(game, [3, 5]) |> all
  @test Jtac.is_action_legal.(game, legal_actions(game)) |> all
  @test Jtac.status(game) == Jtac.Status()
  @test copy(game).board == game.board
  @test_throws AssertionError apply_action!(game, 5)

  game = random_playout(game)
  @test legal_actions(game) == []
  @test Jtac.is_over(game) == Jtac.is_over(Jtac.status(game))

  games, policies = augment(game, rand(Float32, 10))
  @test length(games) == length(policies) == 8

end



