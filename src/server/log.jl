
export log, @log, @logwarn, @logerror, @logdebug

"""
    log(instance, level, source, msg)

Logging function related to `instance`. Logs `msg` on
loglevel `level` with additional information about the `source`
of the log message, like the originating function or thread.

Loglevels are `0 => :error`, `1 => :warning`, `2 => :info`, `3 => :debug`.
"""
function log end

macro log(k, str)
  quote
    _self_ = Base.nameof(var"#self#")
    log($k, 2, _self_, $str)
  end |> esc
end

macro logerror(k, str)
  quote
    _self_ = Base.nameof(var"#self#")
    log($k, 0, _self_, $str)
  end |> esc
end

macro logwarn(k, str)
  quote
    _self_ = Base.nameof(var"#self#")
    log($k, 1, _self_, $str)
  end |> esc
end

macro logdebug(k, str)
  quote
    _self_ = Base.nameof(var"#self#")
    log($k, 3, _self_, $str)
  end |> esc
end

