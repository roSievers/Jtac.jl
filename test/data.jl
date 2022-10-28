
import .Game: TicTacToe
import .Player: MCTSPlayer, record

@testset "Pool" begin
  G = TicTacToe
  meta = (uses = Int, age = Int)
  criterion = x -> (2 - x.uses) * (3 - x.age)
  dp = Data.Pool(G, meta, criterion, capacity = 100)
  ds = record(MCTSPlayer(power = 5), 100, game = G)
  append!(dp, ds, (uses = 0, age = 1))
  @test Data.occupation(dp) > 1
  @test length(dp) == length(ds)
  Data.trim!(dp)
  @test length(dp) == Data.capacity(dp)
  (data, sel) = Data.sample(dp, 20)
  @test data isa Data.DataSet{G}
  @test length(data) == length(sel) == 20
  Data.capacity!(dp, 50)
  @test Data.trim!(dp) == 50
  @test length(dp) == Data.capacity(dp)
  Data.update!(dp) do meta
    (;meta..., age = meta.age + 1)
  end
  @test dp.meta[1].age == 2
  Data.criterion!(x -> 2 - x.age, dp)
  @test Data.trim!(dp) == Data.capacity(dp)
  @test length(dp) == 0
end
