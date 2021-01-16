
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

