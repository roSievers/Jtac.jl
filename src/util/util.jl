

"""
    parallelforeach(f, items; ntasks, threads)

Run `f(item)` for each `item` in `items`. If `threads = true`, threading via
`ntasks` tasks is used. If `threads = false`, `ntasks` async tasks are used.
"""
function parallelforeach(f, items :: AbstractVector; ntasks, threads)
  ch = Channel{eltype(items)}(length(items))
  foreach(item -> put!(ch, item), items)
  close(ch)
  if threads
    Threads.foreach(f, ch; ntasks)
  else
    asyncmap(f, ch; ntasks)
  end
  nothing
end


"""
    showindented(io, args...; indent = 0, indentfirst = true)

Call `show(io, args...)` while making sure that each new line is indented with
`indent` spaces. If `indentfirst = true`, the first line is also indented.
"""
function showindented(io, args...; indent = 0, indentfirst = true)
  ind = repeat(" ", indent)
  buf = IOBuffer()
  show(buf, args...)
  str = String(take!(buf))
  str = replace(str, "\n" => "\n" * ind)
  if indentfirst
    str = ind * str
  end
  print(io, str)
end

# -------- Parallel-Stable Progress Maps ------------------------------------- #

"""
    stepper(description, length; <keyword arguments>)

Return a pair of functions `(step, finish)` that control a progress bar with
title `description` and `length` number of steps. For the keyword arguments, see
`ProgressMeter.Progress`.

The progress bar is running in its own thread and is communicated with through
a [`Base.Channel`](@ref). This allows calling the `step` function safely under any
condition. However, you **must** make sure that `step()` is not used after
`finish()` anymore. This means that all tasks or futures have to be fetched
before calling `finish`.

# Examples

```julia
step, finish = Util.stepper("Progress", 15)

# Do some calculations in a parallel fashion
pmap(1:15) do i 
  sleep(0.5)
  step()
  i
end

# Since pmap fetches all tasks/futures it creates, we can safely finish
finish()
```
"""
function stepper(description, n :: Int, kwargs...)

  glyphs = ProgressMeter.BarGlyphs("[=>⋅]")
  progress = ProgressMeter.Progress( n + 1
                                   ; dt = 0.1
                                   , desc = description
                                   , barglyphs = glyphs
                                   , kwargs... )

  # Channel that can be used to signal a step on any thread
  channel = Channel{Bool}(0)

  # Thread that manages progressing the progress bar
  thread = @async begin

    while take!(channel)
      ProgressMeter.next!(progress)
    end

    ProgressMeter.printover(progress.output, "")

  end

  step = () -> (put!(channel, true); yield())
  finish = () -> (put!(channel, false); fetch(thread))

  step, finish
end

