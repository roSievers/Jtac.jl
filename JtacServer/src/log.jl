
const prefix = Styled.comment("jtac │")
const swarn  = Styled.warn("warning:")
const serror = Styled.error("ERROR:")
const sdebug = Styled.debug("DEBUG:")

info(io :: IO, str)  = println(io, "$prefix $str")
warn(io :: IO, str)  = println(io, "$prefix $swarn $str")
error(io :: IO, str) = println(io, "$prefix $serror $str")

function debug(io :: IO, str)
  if DEBUG[] println(io, "$prefix $sdebug $str") end
end

info(args :: String ...)  = info(stdout, args...)
warn(args :: String ...)  = warn(stdout, args...)
error(args :: String ...) = error(stdout, args...)
debug(args :: String ...) = debug(stdout, args...)

function info(c :: RemoteChannel{Channel{String}}, str)
  put!(c, str)
end

function warn(c :: RemoteChannel{Channel{String}}, str)
  put!(c, "$swarn $str")
end

function error(c :: RemoteChannel{Channel{String}}, str)
  put!(c, "$serror $str")
end

function debug(c :: RemoteChannel{Channel{String}}, str)
  if DEBUG[] put!(c, "$sdebug $str") end
end

