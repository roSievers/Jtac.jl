


while isopen
  try
    send_data()
  catch err
    close(sock)
    break
  end
end

while isopen
  try
    send_contest()
  catch err
    close(sock)
    break
  end
end

try
  while isopen

  end
catch err
  if err isa Base.IOError -> connection closed end
  close(sock)
end

@async while 


Want to have following behavior:
  - If exception arises in subtask
    * only propagate it to the outside if shutdown / interrupt
    * if other exception, check for soft global shutdown
    * if no global soft shutdown took place, log the exception but end the function in ordered way
    * if global soft shutdown took place, don't log the exception. Clean up and leave.
    * may kill client connection if sending / receiving failed <-> (may make many reconnects necessary?)


how to stop wait_shutdown without triggering shutdown?

@sync

@async wait_shutdown(channels)
@async 


function wrap_shutdown_return(fs, error_handling, channels)
  tasks = []
  stop = false
  try
    @sync begin
      stoptask = @async begin
        while !stop
          if check_shutdown(ch)
            throw(Shutdown())
          else
            sleep(1)
          end
        end
      end
      push!(tasks, stoptask)
      for f in fs
        push!(tasks, @async f())
      end
    end
  catch err
    stop = true
    if sound_exn(err) || (err isa TaskFailedException && sound_fail(err.task))
      soft_shutdown!(channels)
      error_handling()
      wait_tasks(tasks, false)
    elseif err isa TaskFailedException
      rethrow(err.task.exception)
    else
      rethrow(err)
    end
  end
  stop = true
end

