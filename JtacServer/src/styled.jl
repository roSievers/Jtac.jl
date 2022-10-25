
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

Style(req :: Msg.ToPlay.DataReq; kw...) = Style("D$(req.reqid)"; kw...)
Style(req :: Msg.ToPlay.ContestReq; kw...) = Style("C$(req.reqid)"; kw...)

Style(x :: Msg.FromPlay.Data; kw...) = Style("d$(x.id)"; kw...)
Style(x :: Msg.FromPlay.Contest; kw...) = Style("c$(x.id)"; kw...)


function Base.show(io :: IO, s :: Style)
  ioc = IOContext(io, :color => true)
  printstyled(ioc, s.text, bold = s.bold, color = s.color)
end

name(str)    = Style(str; color = :green)
keyword(str) = Style(str; color = :cyan)
quant(str)   = Style(str; color = :yellow)
faulty(str)  = Style(str; color = :red)
warn(str)    = Style(str; color = :red)
error(str)   = Style(str; color = :red, bold = true)
string(str)  = Style(str; color = :yellow)
comment(str) = Style(str; color = 245)
debug(str)   = Style(str; color = :blue)

