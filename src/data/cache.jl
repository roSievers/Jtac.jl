
struct Cache{G <: AbstractGame, GPU}

  data        # game representation data

  labels

end

function Cache{G}(data, labels; gpu) where {G}
  at = Model.atype(gpu)
  data = convert(at, data)
  labels = map(labels) do label
    convert(at, label)
  end
  Cache{G, gpu}(data, labels)
end

function Cache(ds :: DataSet{G}; gpu = false) where {G <: AbstractGame}
  data = Game.array(ds.games)
  labels = map(ds.labels) do label
    hcat(label...)
  end

  Cache{G}(data, labels; gpu)
end

Base.length(c :: Cache) = size(c.data)[end]

