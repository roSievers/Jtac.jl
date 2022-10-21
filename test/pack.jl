
@testset "Pack" begin

  function pack_unpack(G :: Type{<: Game.AbstractGame})
    game = G()
    Game.random_turns!(game, 1:10)
    packed_game = Pack.pack(game)
    unpacked_game = Pack.unpack(packed_game, typeof(game))
    game == unpacked_game
  end

  @test pack_unpack(Game.TicTacToe)
  @test pack_unpack(Game.MetaTac)
  @test pack_unpack(Game.Morris)
  @test pack_unpack(Game.Nim)
  @test pack_unpack(Game.Nim2)

  function pack_unpack(f :: Model.NamedFunction)
    packed = Pack.pack(f)
    unpacked = Pack.unpack(packed, Model.NamedFunction)
    f == unpacked
  end

  @test pack_unpack(Model.NamedFunction("id"))
  @test pack_unpack(Model.NamedFunction("relu"))
  @test pack_unpack(Model.NamedFunction("elu"))
  @test pack_unpack(Model.NamedFunction("selu"))
  @test pack_unpack(Model.NamedFunction("tanh"))
  @test pack_unpack(Model.NamedFunction("softmax"))


  function pack_unpack(w :: Model.Weight)
    packed = Pack.pack(w)
    unpacked = Pack.unpack(packed, Model.Weight)
    typeof(w.data) == typeof(unpacked.data) &&
      all(Knet.value(w.data) .== Knet.value(unpacked.data))
  end

  @test pack_unpack(Model.Weight(rand(Float32, 5, 5)))
  @test pack_unpack(Model.Weight(Knet.param(rand(Float32, 5, 5))))


  function pack_unpack(l :: Model.Layer, data)
    packed = Pack.pack(l)
    unpacked = Pack.unpack(packed, Model.Layer)
    all(l(data) .== unpacked(data))
  end

  layer = Model.@chain (9, 9, 3) Pool() Dropout(0.5) Conv(16, "relu") Dense(8, "relu")
  data = rand(Float32, 9, 9, 3, 1)
  @test pack_unpack(layer, data)

  layer = Model.@chain (9, 9, 3) Pool() Dropout(0.5) Conv(16, "relu") Batchnorm()
  data = rand(Float32, 9, 9, 3, 1)
  @test pack_unpack(layer, data)

  function pack_unpack(l :: Model.AbstractModel{G}) where {G}
    fname = tempname()
    Model.save(fname, l)
    unpacked = Model.load(fname)
    games = [Game.random_turns!(G(), 1:10) for _ in 1:100]
    same = all(l(games) .== unpacked(games))
    same && typeof(l) == typeof(unpacked)
  end

  function pack_unpack(l :: Union{Model.DummyModel, Model.RandomModel, Model.Model.RolloutModel})
    packed = Pack.pack(l)
    unpacked = Pack.unpack(packed, Model.AbstractModel)
    typeof(l) == typeof(unpacked)
  end

  for M in [Model.DummyModel, Model.RandomModel, Model.RolloutModel]
    @test pack_unpack(M())
  end

  for G in [Game.TicTacToe, Game.MetaTac, Game.Nim, Game.Nim2, Game.Morris]
    chain = Model.@chain G Dense(128, "relu") Pointwise("tanh")
    for async in [true, false], cache in [true, false]
      model = Model.NeuralModel(G, chain)
      model = Model.tune(model; async, cache)
      @test pack_unpack(model)
    end
  end

  function pack_unpack(ds :: Data.Dataset{G}) where {G}
    name = tempname()
    Data.save(name, ds)
    up = Data.load(name)
    same = all(up.games .== ds.games)
    same &= all(all(x .== y) for (x, y) in zip(up.label, ds.label))
    same &= all(all(x .== y) for (x, y) in zip(up.flabel, ds.flabel))
    same && all(up.features .== ds.features)
  end

  ds = Data.record(Player.MCTSPlayer(power = 20), 100, game = Game.TicTacToe())
  @test pack_unpack(ds)

  ds = Data.record(Player.MCTSPlayer(power = 5), 100, game = Game.MetaTac())
  @test pack_unpack(ds)

  function pack_unpack(p :: Player.IntuitionPlayer{G}) where {G}
    packed = Pack.pack(p)
    up = Pack.unpack(packed, Player.AbstractPlayer)
    games = [Game.random_turns!(G(), 1:10) for _ in 1:100]
    same = all(p.model(games) .== up.model(games))
    same &= p.temperature == up.temperature
    same && p.name == up.name
  end

  function pack_unpack(p :: Player.MCTSPlayer{G}) where {G}
    packed = Pack.pack(p)
    up = Pack.unpack(packed, Player.AbstractPlayer)
    games = [Game.random_turns!(G(), 1:10) for _ in 1:100]
    same = all(p.model(games) .== up.model(games))
    same &= p.temperature == up.temperature
    same &= p.exploration == up.exploration
    same &= p.power == up.power
    same &= p.dilution == up.dilution
    same && p.name == up.name
  end

  G = Game.MetaTac
  model = Model.NeuralModel(G, Model.@chain G Conv(64, "relu") Batchnorm() Pool() Dense(32, "relu"))

  #player = Player.MCTSPlayer(power = 10, temperature = 0.75)
  #@show pack_unpack(player)

  player = Player.IntuitionPlayer(model, temperature = 0.75)
  @test pack_unpack(player)

  player = Player.MCTSPlayer(model, power = 5, temperature = 0.75, exploration = 2.)
  @test pack_unpack(player)

  Training.train!(player, epochs = 1, playings = 10)
  @test pack_unpack(player)

end

# New feature handling... Think about this!
# Don't like current situation with loss functions
# Remove Training module -> make this part of Player:
#   -> Player.train!
# Maybe same for Rank module? Player.compete(...) ?

