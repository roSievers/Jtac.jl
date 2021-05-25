
module Stl

  import ..TrainDataServe
  import ..TrainContestServe
  import ..ServeData
  import ..ServeContest

  struct Style
    text :: String
    bold :: Bool
    color :: Union{Int, Symbol}
  end

  function Style(txt :: String; bold = false, color = :normal)
    Style(txt, bold, color)
  end

  # Some automatic style conversions
  Style(x :: Real; kw...) = Style(Base.string(x); kw...)

  Style(req :: TrainDataServe; kw...) = Style("D$(req.reqid)"; kw...)
  Style(req :: TrainContestServe; kw...) = Style("C$(req.reqid)"; kw...)

  Style(x :: ServeData; kw...) = Style("d$(x.id)"; kw...)
  Style(x :: ServeContest; kw...) = Style("c$(x.id)"; kw...)


  function Base.show(io :: IO, s :: Style)
    ioc = IOContext(io, :color => true)
    printstyled(ioc, s.text, bold = s.bold, color = s.color)
  end

  name(str)    = Style(str; color = :green)
  keyword(str) = Style(str; color = :cyan)
  quant(str)   = Style(str; color = :green)
  faulty(str)  = Style(str; color = :red)
  warn(str)    = Style(str; color = :red)
  error(str)   = Style(str; color = :red, bold = true)
  string(str)  = Style(str; color = :yellow)
  comment(str) = Style(str; color = 245)
  debug(str)   = Style(str; color = :blue)

end # module Stl


module Log

  import ..DEBUG
  import ..Stl
  import Distributed: RemoteChannel

  prefix = Stl.comment("jtac │")
  swarn  = Stl.warn("warning:")
  serror = Stl.error("ERROR:")
  sdebug = Stl.debug("DEBUG:")

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

end # module Log

