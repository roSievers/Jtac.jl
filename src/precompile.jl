
function precompilecontent(G :: Type{<: Game.AbstractGame}; models = nothing, configure = identity)
  if Game.isaugmentable(G)
    Game.augment(Game.instance(G))
  end
  Game.status(Game.instance(G))
  Game.policylength(Game.instance(G))
  Game.legalactions(Game.instance(G))
  Game.array([Game.instance(G), Game.randominstance(G)])
  Game.rollout(Game.instance(G))
  Game.movecount(Game.instance(G))
  copy(Game.instance(G))
  size(Game.instance(G))
  io = IOBuffer()
  show(io, Game.instance(G))
  show(io, MIME"text/plain"(), Game.instance(G))
  # bytes = Pack.pack(Game.instance(G))
  # Pack.unpack(bytes, G)

  if isnothing(models)
    models = [
      Model.RandomModel(G),
      Model.DummyModel(G),
      Model.RolloutModel(G),
      Model.Zoo.ShallowConv(G) |> configure,
      Model.Zoo.MLP(G, Model.Activation(NNlib.relu), widths = [16]) |> configure,
      Model.Zoo.ZeroRes(G, blocks = 1, filters = 16) |> configure,
      Model.Zoo.ZeroConv(G, blocks = 1, filters = 16) |> configure,
    ]
  end

  players = [
    m -> Player.IntuitionPlayer(m),
    m -> Player.MCTSPlayer(m),
    m -> Player.MCTSPlayerGumbel(m),
  ]

  for model in models
    Model.gametype(model)
    Model.ntasks(model)
    Model.isasync(model)
    Model.basemodel(model)
    Model.childmodel(model)
    Model.trainingmodel(model)
    Model.playingmodel(model)
    Model.apply(model, Game.randominstance(G))
    Model.targets(model)
    Model.targetnames(model)
    io = IOBuffer()
    show(io, model)
    show(io, MIME"text/plain"(), model)
    # Model.save(io, model, Model.DefaultFormat())
    # seekstart(io)
    # Model.load(io, Model.DefaultFormat())

    if model isa NeuralModel
      Model.getbackend(model)
    end

    for p in players
      player = p(model)
      Player.gametype(player)
      Player.basemodel(player)
      Player.childmodel(player)
      Player.trainingmodel(player)
      Player.playingmodel(player)
      Player.name(player)
      Player.think(player, Game.instance(G))
      Player.apply(player, Game.instance(G))
      Player.decide(player, Game.instance(G))
      Player.decideturn(player, Game.instance(G))
      Player.move(Game.instance(G), player)
      Player.move!(Game.instance(G), player)

      io = IOBuffer()
      show(io, model)
      show(io, MIME"text/plain"(), model)
    end
    
  end
end

