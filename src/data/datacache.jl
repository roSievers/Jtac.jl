
struct DataCache{G <: AbstractGame, GPU}

  data        # game representation data

  vlabel      # value labels
  plabel      # policy labels
  flabel      # feature labels

end

function DataCache{G}( data, vlabel, plabel, flabel; gpu ) where {G}
  at = Model.atype(gpu)
  data = convert(at, data)
  vlabel = convert(at, vlabel)
  plabel = convert(at, plabel)
  flabel = isnothing(flabel) ? nothing : convert(at, hcat(ds.flabel...))
  DataCache{G, gpu}(data, vlabel, plabel, flabel)
end

function DataCache( ds :: DataSet{G}
                  ; gpu = false
                  , use_features = false
                  ) where {G <: AbstractGame}

  # Preparation
  at = Model.atype(gpu)
  vplabel = hcat(ds.label...)

  # Convert to at
  data = convert(at, Game.array(ds.games))
  vlabel = convert(at, vplabel[1, :])
  plabel = convert(at, vplabel[2:end, :])
  flabel = use_features ? convert(at, hcat(ds.flabel...)) : nothing

  DataCache{G, gpu}(data, vlabel, plabel, flabel)

end

Base.length(c :: DataCache) = size(c.data)[end]

