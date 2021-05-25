
import Base.Experimental: @sync

function unwrap_exn(exn :: Exception)
  if exn isa TaskFailedException
    unwrap_exn(exn.task.exception)
  elseif exn isa CompositeException
    unwrap_exn(exn.exceptions[1])
  else
    exn
  end
end

function with_gentle_exit(f, on_exit = () -> nothing; log = println, name = "jtac")
  ret = nothing
  exit = false
  name = Stl.name(name)
  ctrld = Stl.keyword("ctrl-d")
  ctrlc = Stl.faulty("ctrl-c")

  listen() = begin
    while !eof(stdin) && !exit
      user_input = Stl.string(readline())
      log("received unexpected user input $user_input")
      log("to stop $name, press $ctrld")
    end
    if !exit
      log("received exit signal. please wait while shutting down...")
      exit = true
      on_exit()
    end
  end

  run(ltask) = begin
    try
      ret = f()
      log("$name finished")
    catch err
      errstr = Stl.error(string(err))
      log("$name returned with an exception: $errstr")
      rethrow(err)
    finally
      if !exit
        log("press $ctrld to continue")
        exit = true
        wait(ltask)
      end
    end
  end

  try
    log("$name can be stopped gently by pressing $ctrld")
    log("pressing $ctrlc may lead to ungraceful exits (or deadlocks in the REPL)")
    @sync run(@async listen())
    ret
  catch err
    rethrow(unwrap_exn(err))
  end
end

# if we do not want an exception to end the program, and only print a warning
# / message if user exit was not issued
function catch_recoverable(f, ch, on_catch, msg :: Function, exts; warn = Log.warn)
  try f()
  catch err
    if err isa Union{exts...} 
      isready(ch["exit"]) || warn(msg(err))
      on_catch(err)
    else
      rethrow(err)
    end
  end
end

function catch_recoverable(f, ch, on_catch, msg, args...; kw...)
  catch_recoverable(f, ch, on_catch, _ -> msg, args...; kw...)
end

# stolen from https://github.com/JuliaLang/julia/issues/36217
# unfortunately, it does not seem to be possible to wait for
# ch["exit"] directly (since we have to close the resource c
# in the code below at the end of timeout)
function wait_or_exit(c, timeout :: Real) 
  timer = Timer(timeout) do t
    isready(c) || close(c)
  end
  try wait(c)
  catch nothing
  finally close(timer)
  end
end

# Only rethrow an error if "exit" is not set
# If the debug mode is activated, show that an error
# is not rethrown because of "exit"
function rethrow_or_exit(ch, err)
  if !isready(ch["exit"])
    rethrow(err)
  elseif DEBUG[]
    e = Stl.error(string(err))
    Log.debug("caught exception absorbed by exit signal: $e")
    throw(err)
  end
end

# close all data channels that are not the "exit" channel
function close_data_channels!(ch)
  for (key, c) in ch
    if key != "exit" close(c) end
  end
end

