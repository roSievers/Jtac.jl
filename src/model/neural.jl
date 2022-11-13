
# -------- Neural Network Head Creation -------------------------------------- #

function prepare_head(head, s, l, gpu)

  if isnothing(head)

    head = Dense(prod(s), l, gpu = gpu)

  else

    @assert valid_insize(head, s) "Head incompatible with trunk."
    @assert prod(outsize(head, s)) == l "Head incompatible with game."
    head = (on_gpu(head) == gpu) ? head : swap(head)

  end

  head

end

# -------- Neural Model ------------------------------------------------------ #

"""
Trainable model that uses a neural network to generate value and policy targets
for a game state. Optionally, the network can also predict other targets.
"""
struct NeuralModel{G, GPU} <: AbstractModel{G, GPU}

  trunk :: Layer{GPU}                    # map tensorized game state to trunk features

  heads :: Vector{Layer{GPU}}            # target heads
  targets :: Vector{PredictionTarget{G}} # targets

end

Pack.register(NeuralModel)
Pack.freeze(m :: NeuralModel) = to_cpu(m)

"""
    NeuralModel(G, trunk [; vhead, phead, value, policy, opt_targets, opt_heads])

Construct a model for gametype `G` based on the layer `trunk`,
with optional targets `opt_targets` enabled. The heads `vhead`, `phead`, and
`opt_heads` are neural network layers that produce the logits for target
prediction. The activation functions of the targets are used to map the logits
to the actual target estimates.
"""
function NeuralModel( :: Type{G}
                    , trunk :: Layer{GPU}
                    ; vhead = nothing
                    , phead = nothing
                    , value :: ValueTarget{G} = ValueTarget(G)
                    , policy :: PolicyTarget{G} = PolicyTarget(G)
                    , opt_targets = []
                    , opt_heads = nothing
                    ) where {G, GPU}

  @assert valid_insize(trunk, size(G)) "Trunk incompatible with $G"

  targets = PredictionTarget{G}[value, policy, (t for t in opt_targets)...]

  if isnothing(opt_heads)
    heads = [vhead, phead, (nothing for _ in targets[3:end])...]
  end

  @assert length(targets) == length(heads) "Number of heads does not match number of headed targets"

  # Check the provided heads or create linear heads if nothing is specified
  os = outsize(trunk, size(G))
  heads = Layer{GPU}[ prepare_head(h, os, length(t), GPU) for (h, t) in zip(heads, targets) ]

  NeuralModel{G, GPU}(trunk, heads, targets) 

end

function add_target!( m :: NeuralModel{G, GPU}
                    , t :: PredictionTarget
                    , head :: Union{Nothing, Layer{GPU}} = nothing
                    ) where {G, GPU}

  os = outsize(m.trunk, size(G))
  head = prepare_head(head, os, length(t), GPU)
  push!(m.targets, t)
  push!(m.heads, head)

end

function Target.adapt(m :: NeuralModel{G, GPU}, targets) where {G, GPU}
  idx = Target.adapt(m.targets, targets)
  NeuralModel{G, GPU}(m.trunk, m.heads[idx], m.targets[idx])
end

# Low level access to neural model predictions
function (m :: NeuralModel{G, GPU})( data
                                   , opt_targets = false
                                   ) where {G <: AbstractGame, GPU}

  # Get the trunk output and batch size
  tout = m.trunk(data)
  bs = size(tout)[end]

  ts = m.targets

  # prepare iteration over targets
  if opt_targets
    ht = zip(m.heads, ts)
  else
    ht = zip(m.heads[1:2], ts[1:2])
  end

  # predict the desired targets
  out = map(ht) do (head, target)

    tmp = reshape(head(tout), length(target), bs)
    res = Target.activate(target, tmp)
    release_gpu_memory!(tmp)
    res

  end

  release_gpu_memory!(tout)
  out

end

# Higher level access
function (m :: NeuralModel{G, GPU})( games :: Vector{G}
                                   , opt_targets = false
                                   ) where {G <: AbstractGame, GPU} 

  at = atype(GPU)
  data = convert(at, Game.array(games))

  results = m(data, opt_targets) 
  release_gpu_memory!(data)

  map(results) do result
    convert(Array, result) :: Matrix{Float32}
  end
end

function (m :: NeuralModel{G})( game :: G
                              , opt_targets = false
                              ) where {G <: AbstractGame}

  m([game], opt_targets)

end

function apply(m :: NeuralModel{G}, game :: G, opt_targets = false) where {G <: AbstractGame}
  if !opt_targets
    v, p = m(game, false)
    (value = v[1], policy = reshape(p |> to_cpu, :))
  else
    output = m(game, true)
    ts = m.targets
    names = (Target.name(t) for t in ts) 
    vals = map(output) do val
      if length(val) == 1
        val[1]
      else
        to_cpu(reshape(val, :))
      end
    end
    (; zip(names, vals)...)
  end
end

function swap(m :: NeuralModel{G, GPU}) where {G, GPU}

  NeuralModel{G, !GPU}( swap(m.trunk)
                      , swap.(m.heads)
                      , m.targets )

end

function Base.copy(m :: NeuralModel{G, GPU}) where {G, GPU}

  NeuralModel{G, GPU}( copy(m.trunk)
                     , copy.(m.heads)
                     , m.targets )

end

Target.targets(m :: NeuralModel) = m.targets

training_model(m :: NeuralModel) = m


function Base.show(io :: IO, m :: NeuralModel{G, GPU}) where {G, GPU}
  at = GPU ? "GPU" : "CPU"
  print(io, "NeuralModel{$(Game.name(G)), $at}(")
  show(io, m.trunk)
  print(io, ")")
end

function Base.show(io :: IO, :: MIME"text/plain", m :: NeuralModel{G, GPU}) where {G, GPU}
  at = GPU ? "GPU" : "CPU"
  println(io, "NeuralModel{$(Game.name(G)), $at}:")
  print(io, "  trunk: "); show(io, m.trunk)
  for (h, t) in zip(m.heads, m.targets)
    name = Target.name(t)
    println(io); print(io, "  $name: "); show(io, h)
  end
end

function tune( m :: NeuralModel{G, GPU}
             ; gpu = GPU
             , async = false
             , cache = false
             ) where {G, GPU}

  I = Union{Signed, Unsigned} # Grr, in julia, Bool <: Integer....
  gpu != GPU && (m = swap(m))
  async == true && (m = Async(m))
  async isa I && async > 0 && (m = Async(m, max_batchsize = async))
  cache == true && (m = Caching(m))
  cache isa I && cache > 0 && (m = Caching(m, max_cachesize = cache))
  m
end


