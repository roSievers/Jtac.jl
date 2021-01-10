
log_info(io :: IO, str)  = println(io, "<jtac> $str")
log_warn(io :: IO, str)  = println(io, "<jtac> warning: $str")
log_error(io :: IO, str) = println(io, "<jtac> ERROR: $str")

log_info(args :: String ...)  = log_info(stdout, args...)
log_warn(args :: String ...)  = log_warn(stdout, args...)
log_error(args :: String ...) = log_error(stdout, args...)

import Distributed

function log_info(c :: Distributed.RemoteChannel{Channel{String}}, str)
  put!(c, str)
end

function log_warn(c :: Distributed.RemoteChannel{Channel{String}}, str)
  put!(c, "warning: $str")
end

function log_error(c :: Distributed.RemoteChannel{Channel{String}}, str)
  put!(c, "ERROR: $str")
end
