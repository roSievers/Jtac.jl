
"""
    with_BLAS_threads(f, n)

Call `f()` under the temporary setting `BLAS.set_num_threads(n)`.
"""
function with_BLAS_threads(f, n)
  t = BLAS.get_num_threads()
  BLAS.set_num_threads(n)
  try f()
  finally
    BLAS.set_num_threads(t)
  end
end

"""
Exception used to stop the recording or evaluation of matches via the `callback`
or `instance` argument functions.
"""
struct StopMatch <: Exception end

"""
Exception that is thrown as stop signal for the `record(player, channel)`
method.
"""
struct StopRecording <: Exception end


"""
    stopmatch()
    stopmatch(n)

Can be used in the `callback_move` argument for the function [`record`](@ref)
in order to stop recording a game. The method `stopmatch(n)` returns a function
that takes an argument `k` and calls `stopmatch()` when `k > n`.

# Usage
The following call of `record` will cancel matches that take more
than 50 moves.

    record(args...; callback_move = Training.stopmatch(50), kwargs...)
"""
stopmatch() = throw(StopMatch())
stopmatch(n :: Int) = k -> (k > n && stopmatch())


"""
    branchfunction((prob, steps))
    branchfunction((; prob, steps))

Create a branching function with the arguments `prob` (branching probability)
and `steps` (number of branching steps).
"""
branchfunction(f :: Function) = f
branchfunction(prob :: Real) = branchfunction((prob, 1))
branchfunction(prob :: Tuple{<: Real}) = branchfunction(prob[1])

function branchfunction(arg :: Tuple{<: Real, Any})
  game -> begin
    if rand() < arg[1]
      Game.randommove(game, arg[2])
    end
  end
end

function branchfunction(arg)
  prob = get(arg, :prob, nothing)
  steps = get(arg, :steps, nothing)
  @assert !isnothing(prob) "Branch function constructor expected key :prob"
  if isnothing(steps)
    steps = 1
  end
  branchfunction((prob, steps))
end




"""
    record(player, n = 1; <keyword arguments>)

Generate a [`DataSet`](@ref) by letting `player` play against itself `n` times.
To enable dynamic switching of the player during the match, the player may also
be wrapped in a `Channel{AbstractPlayer}`.

# Arguments
- `instance`: Initial game state provider. Defaults to `() -> Game.instance(G)` \
where `G` is the game type of `player`.
- `branch`: Function applied to each game state after a first selfplay. It can \
  return `nothing` or a branched game state, which is then used as roots for \
  new matches (without recursive branching). For convenience, one can also pass \
  a value `prob` (branching probability), a tuple `(prob, steps)` (number of \
  random steps per branching), or a named tuple `(; prob, steps)`.
- `merge = true`: Whether to return one merged dataset or `n` separate ones.
- `augment`: Whether to apply data augmentation to the generated dataset.
- `targets`: Named tuple of [`AbstractTarget`](@ref)s for which labels are \
recorded. Derived from the player by default.
- `anneal = n -> 1.0`: The temperature at move `n` with which the policy \
obtained via [`Player.think`](@ref) is annealed before sampling the next move.
- `callback_match`: Function that is called after each completed match.
- `callback_move`: Function that is called after each individual move.
- `draw_after`: Move threshold after which ongoing matches are declared draws.
- `ntasks = Player.ntasks(player)`: Number of async tasks used for playing.
- `threads = false`: Whether to use threads or async tasks if `ntasks > 1`.
- `progress = true`: Whether to print progress information.

# Examples
```julia
# Record 20 self play matches of an classical MCTS player with power 250
G = ToyGames.TicTacToe
player = Player.MCTSPlayer(power = 250)
dataset = Training.record(player, 20, instance = G, branch = 0.25)

# Record 10 self play matches of MCTS player with shallow predictor network and
# power 50. After move 4, always use the action with maximal policy weight.
G = ToyGames.TicTacToe
model = Model.NeuralModel(G, Model.@chain G Dense(50, "relu"))
player = Player.MCTSPlayer(model, power = 50)
dataset = Training.record(player, 10; augment = false, anneal = (4 => 0.0))
```
"""
function record( p :: Union{P, Channel{P}}
               , n :: Int = 1
               ; targets = nothing
               , anneal = n -> 1.0
               , merge :: Bool = true
               , callback_move :: Function = _ -> nothing
               , ntasks = Player.ntasks(fetch(p))
               , draw_after = typemax(Int)
               , kwargs...
               ) where {G, P <: AbstractPlayer{G}}

  # Bring the argument anneal in proper anneal function form
  anneal = Player.annealfunction(anneal)

  # Function that plays a single match, starting at state `game` with `moves`
  # moves. It returns a `trace`, which is a named tuple of game states, model
  # policies, and the game outcome
  play = (game, moves) -> begin

    game = copy(game)
    games = G[]
    policies = Vector{Float32}[]

    try
      while !isover(game) && moves < draw_after

        # Get the improved policy from the player
        policy = think(fetch(p), game)

        # Record the current game state and policy target label
        # (we do not know the value label yet)
        push!(games, copy(game))
        push!(policies, policy)

        # Advance the game by randomly drawing from the policy
        temperature = anneal(moves)
        policy = Player.anneal(policy, temperature)
        action = Player.sample(policy)
        move!(game, action)

        moves += 1
        callback_move(moves)
      end

      # Push the final game state to the dataset. This game state is included
      # for calculating optional targets, but it has to be removed afterwards
      pl = policylength(G)
      push!(games, copy(game))
      push!(policies, zeros(Float32, pl))

      # Now we know how the game ended. Add value labels to the dataset.
      # Optional targets have to be recorded *after* augmentation
      if moves >= draw_after
        outcome = Game.draw
      else
        outcome = status(games[end])
      end

      (; outcome, games, policies)
    
    catch err
      if err isa StopMatch
        (outcome = Game.draw, games = G[], policies = Vector{Float32}[])
      else
        throw(err)
      end
    end
  end

  # If no explicit targets are set, derive them from the base model or player
  if isnothing(targets)
    player = fetch(p)
    model = basemodel(player)
    targets = Target.targets(isnothing(model) ? player : model)
    names = targetnames(isnothing(model) ? player : model)
    targets = (; zip(names, targets)...)
  end

  # Add default value and policy targets, if they are not overwritten explicitly
  targets = (;
    Target.defaulttargets(G)...,
    targets...
  )

  ds = with_BLAS_threads(1) do
    # Record several matches with (optional) branching and augmentation
    recordbranching(G, play, n; targets, ntasks, kwargs...)
  end

  merge ? Base.merge(ds) : ds
end

function recordbranching( G :: Type{<: AbstractGame}
                        , play :: Function # maps game to results
                        , n :: Int
                        ; targets
                        , ntasks :: Int
                        , threads = false
                        , augment = Game.isaugmentable(G)
                        , callback_match = () -> nothing
                        , instance = () -> Game.instance(G)
                        , branch = 0.
                        , progress = true )

  # Construct the branching function
  branch = branchfunction(branch)

  if progress
    step, finish = Util.stepper("# recording...", n)
  end

  # Extend the provided play function by random branching 
  playbranching = () -> begin
    root = instance()
    @assert root isa G "Unexpected game instance type (expected $G)"

    # Play the game once without branching. Then, create branchpoints and play
    # them as well.
    moves = Game.movecount(root)
    trace = play(root, moves)
    traces = [trace]
    for (i, game) in enumerate(trace.games[1:end-1])
      branchpoint = branch(game)
      if !isnothing(branchpoint) && !Game.isover(branchpoint)
        trace = play(branchpoint, moves + i - 1)
        push!(traces, trace)
      end
    end

    # Make sure that we only include non-trivial traces
    filter!(trace -> length(trace.games) > 1, traces)

    # Augment the datasets and record optional targets as datasets
    if augment && !isempty(traces)
      traces = mapreduce(augmenttrace, vcat, traces)
    end
    datasets = DataSet{G}[recordtargets(G, targets, trace) for trace in traces]

    callback_match()
    if progress
      step()
    end

    # If not empty (because of StopMatch exceptions), merge the datasets of
    # different branches
    if isempty(datasets)
      DataSet(G, targets)
    else
      Base.merge(datasets)
    end
  end


  # Call playbranching n times, either serially or in parallel
  if ntasks == 1 || n == 1
    ds = [playbranching() for _ in 1:n]
  else
    ds = Vector{DataSet{G}}(undef, n)
    Util.pforeach(1:n; ntasks, threads) do index
      ds[index] = playbranching()
    end
  end

  if progress
    finish()
  end

  ds
end

"""
    augmenttrace(trace)

Augment all games and policies of a trace and return several new traces, one for
each symmetry operation.
"""
function augmenttrace(trace)
  gps = [Game.augment(g, p) for (g, p) in zip(trace.games, trace.policies)]
  gs, ps = first.(gps), last.(gps)

  map(1:length(gs[1])) do j
    games = map(x -> x[j], gs)
    policies = map(x -> x[j], ps)
    (; trace.outcome, games, policies)
  end
end

"""
    recordtargets(G, targets, trace)

Record the prediction targets `targets` for `trace` and return the resulting
datasets.
"""
function recordtargets(G :: Type{<: AbstractGame}, targets, trace)
  ds = DataSet(G, targets)
  if length(trace.games) <= 1
    @warn "Encountered trivial trace"
    return ds
  end
  for index in 1:(length(trace.games) - 1)
    game = trace.games[index]
    policy = trace.policies[index]
    ctx = LabelContext(
      game,
      policy,
      trace.outcome,
      index,
      trace.games,
      trace.policies
    )
    push!(ds.games, game)
    for (j, target) in enumerate(values(targets))
      label = Target.label(target, ctx)
      push!(ds.target_labels[j], label)
    end
  end
  @assert isconsistent(ds)
  ds
end


"""
    record(player, channel; kwargs...)

Stream the datasets recorded by `player` to `channel`. The function returns when
`channel` is closed.
"""
function record( p :: Union{P, Channel{P}}
               , ch :: AbstractChannel{DataSet{G}}
               ; callback_move = _ -> nothing
               , ntasks = Model.ntasks(p)
               , threads = false
               , kwargs...
               ) where {G <: AbstractGame, P <: AbstractPlayer{G}}

  cb(move) = begin
    isopen(ch) || throw(StopRecording())
    callback_move(move)
  end

  main_loop(player) = Util.pforeach(1:ntasks; ntasks, threads) do _
    while isopen(ch)
      ds = nothing
      try ds = record(
        player,
        1;
        kwargs...,
        merge = true,
        ntasks = 1,
        callback_move = cb,
      )
      catch err
        err isa StopRecording && break
        throw(err)
      end
      try put!(ch, ds)
      catch err
        err isa InvalidStateException && break
        throw(err)
      end
    end
    nothing
  end

  with_BLAS_threads(1) do
    main_loop(p)
  end
end


"""
    record(player, match; kwargs...)
    record(player, games; kwargs...)

Let `player` evaluate `match` / `games` to create a dataset with value / policy
targets.
"""
function record( player :: AbstractPlayer{G}
               , games :: Vector{G}
               ; progress = true
               , ntasks = Model.ntasks(player)
               , threads = false ) where {G <: AbstractGame}

  n = length(games)
  ds = DataSet(G)
  resize!(ds.games, n)
  resize!(ds.target_labels[1].data, n)
  resize!(ds.target_labels[2].data, n)

  with_BLAS_threads(1) do
    Util.pforeach(1:length(games); ntasks, threads) do index
      game = games[index]
      value, policy = Model.apply(player, game)
      ds.games[index] = copy(game)
      ds.target_labels[1].data[index] = Float32[value]
      ds.target_labels[2].data[index] = policy
    end
  end

  @assert isconsistent(ds)
  ds
end

function record(p :: AbstractPlayer{G}, match :: Match{G}; kwargs...) where {G}
  record(p, match.games)
end
