
"""
Asynchronous model wrapper that allows a model to be called on a batch of games
in parallel when the single calls take place in an async context. Note that an
`Threaded` model always returns CPU arrays, even if the worker model acts on the
GPU.
"""
mutable struct Threaded{G <: AbstractGame} <: AbstractModel{G}
  model :: NeuralModel{G}

  inout :: Task
  inout_ch :: Channel

  calc :: Task
  calc_ch :: Channel

  max_batchsize :: Int
  buffersize :: Int     # Must not be smaller than max_batchsize!

  lock
  buf_in
  buf_out

  profile
end

Pack.@only Threaded [:model, :max_batchsize, :buffersize]

Threaded{G}(models, max_batchsize, buffersize) where {G} =
  Threaded(models; max_batchsize, buffersize) :: Threaded{G}

"""
    Threaded(model; max_batchsize, buffersize)

Wraps `model :: NeuralModel` to become a threaded model with maximal batchsize
`max_batchsize` for evaluation in parallel and a buffer of size `buffersize` for
queuing.
"""
function Threaded( model :: NeuralModel{G};
                   max_batchsize = 50,
                   buffersize = 2max_batchsize ) where {G <: AbstractGame}

  # Make sure that the buffer is larger than the maximal allowed batchsize
  @assert buffersize >= max_batchsize "buffersize must be larger than max_batchsize"

  l = policylength(G)

  inout_ch = Channel(buffersize)
  calc_ch = Channel{Tuple{Int, Int}}(2)

  lock = [Threads.Condition() for _ in 1:2]
  buf_in = [Vector{G}(undef, max_batchsize) for _ in 1:2]

  buf_v = [zeros(Float32, max_batchsize) for _ in 1:2]
  buf_p = [zeros(Float32, l, max_batchsize) for _ in 1:2]
  buf_out = [(v = buf_v[idx], p = buf_p[idx]) for idx in 1:2]

  profile = (; batchsize = Float64[])


  inout = @async inout_task(inout_ch, calc_ch, lock, buf_in, buf_out, max_batchsize)
  calc = @async calc_task(model, calc_ch, lock, buf_in, buf_out, profile)

  # Create the instance
  tmodel = Threaded{G}( model
                      , inout, inout_ch
                      , calc, calc_ch
                      , max_batchsize
                      , buffersize
                      , lock
                      , buf_in, buf_out
                      , profile )

  # Register finalizer
  finalizer(tmodel) do model
    close(model.inout_ch)
    close(model.calc_ch)
  end

  # Return the instance
  tmodel
end


function apply(m :: Threaded{G}, game :: G) where {G <: AbstractGame}
  out_ch = Channel(1)
  put!(m.inout_ch, (copy(game), out_ch))
  take!(out_ch)
end


function switchmodel(m :: Threaded{G}, model :: AbstractModel{G}) where {G}
  Threaded(model, max_batchsize = m.max_batchsize, buffersize = m.buffersize)
end

adapt(backend, m :: Threaded) = switchmodel(m, adapt(backend, m.model))

isasync(m :: Threaded) = true
ntasks(m :: Threaded) = m.buffersize
childmodel(m :: Threaded) = m.model
basemodel(m :: Threaded) = basemodel(m.model)
trainingmodel(m :: Threaded) = trainingmodel(m.model)

Base.copy(m :: Threaded) = switchmodel(m, copy(m.model))

function Base.show(io :: IO, m :: Threaded{G}) where {G <: AbstractGame}
  print(io, "Threaded($(m.max_batchsize), $(m.buffersize), ")
  show(io, m.model)
  print(io, ")")
end

function Base.show(io :: IO, mime :: MIME"text/plain", m :: Threaded{G}) where {G}
  print(io, "Threaded($(m.max_batchsize), $(m.buffersize)) ")
  show(io, mime, m.model)
end


# -------- Tasks ------------------------------------------------------------- #

function calc_task(model, calc_ch, lock, buf_in, buf_out, profile)

  @info "Calc task initiated on thread $(Threads.threadid())"
  aligndevice!(model)

  while isopen(calc_ch)
    n, idx = try take!(calc_ch) catch _ break end
    push!(profile.batchsize, n)

    Base.lock(lock[idx]) do
      try
        games = buf_in[idx][1:n]
        vp = model(games)
        v, p = convert.(Array{Float32}, vp)
        buf_out[idx].v[1:n] .= reshape(v, :)
        buf_out[idx].p[:, 1:n] .= p
        releasememory!.(vp)

        notify(lock[idx])

      catch err
        @error "Error in Threaded worker: $err"
        close(calc_ch)
      end
    end
  end

  # close(io_ch)
  close(calc_ch)
  foreach(notify, lock)
  @info "Threaded worker: calc_task returns"
end

function inout_task(args...)
  wake = Threads.Channel{Bool}(1)
  put!(wake, true)
  @sync begin
    @async inout_task_idx(1, wake, args...)
#    @async inout_task_idx(2, wake, args...)
  end
end

function inout_task_idx(idx, wake, io_ch, calc_ch, lock, buf_in, buf_out, max_batchsize)
  while isopen(calc_ch)
    take!(wake)
    inputs = collect_inputs(io_ch, calc_ch, max_batchsize)
    put!(wake, true)

    isopen(calc_ch) || break
    n = length(inputs)

    Base.lock(lock[idx]) do
      for i in 1:n
        buf_in[idx][i] = inputs[i][1]
      end

      handle_error(calc_ch, inputs) do
        put!(calc_ch, (n, idx))
        wait(lock[idx])

        for i in 1:n
          value = buf_out[idx].v[i]
          policy = buf_out[idx].p[:, i]
          res = (; value, policy)
          put!(inputs[i][2], res)
        end
      end
    end
  end
end

function handle_error(f, calc_ch, inputs)
  try f()
  catch err
    close(calc_ch)
    for i in 1:length(inputs)
      try close(inputs[i][2]) catch end
    end
  end
end

function collect_inputs(io_ch, calc_ch, max_batchsize)
  inputs = Vector()
  handle_error(calc_ch, inputs) do
    wait(io_ch) # make sure that at least one entry is available
    while isready(io_ch) && length(inputs) < max_batchsize
      val = take!(io_ch)
      push!(inputs, val)
      yield()
    end
  end
  inputs
end

