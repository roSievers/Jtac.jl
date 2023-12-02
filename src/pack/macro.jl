
#
# process formats
#

function _isnativeformat(sym :: Symbol)
  try
    F = getproperty(Pack, sym)
    F <: Format
  catch
    false
  end
end

function _injectmodule(ex)
  if ex isa Symbol
    _isnativeformat(ex) ? :(Pack.$ex) : ex
  elseif ex isa Expr
    Expr(ex.head, map(_injectmodule, ex.args)...)
  else
    ex
  end
end

function _parsescope(ex)
  len = ex isa Symbol ? 0 : length(ex.args)
  if ex isa Symbol # @pack T args...
    (ex, nothing)
  elseif len == 1 && ex.head == :braces
    ex = ex.args[1]
    len = ex isa Symbol ? 0 : length(ex.args)
    if len == 1 && ex.head == :(<:) # @pack {<: T} args...
      (ex, nothing)
    elseif len == 2 && ex.head == :(<:) && ex.args[1] isa Symbol # @pack {S <: T} args...
      (ex, ex.args[1])
    end
  end
end

_isinformat(:: Symbol) = false

function _isinformat(ex :: Expr)
  length(ex.args) == 3 && ex.head == :call && ex.args[1] == :in
end

function _splitinformat(ex :: Expr)
  if _isinformat(ex)
    ex.args[2], ex.args[3]
  end
end

function _parseinformat(ex)
  res = _splitinformat(ex)
  if !isnothing(res)
    res[1], _injectmodule(res[2])
  else
    error("Expected syntax \"A\", \"A in B\" or \"A => B\" in Pack.@pack")
  end
end

function _parsescopeformat(ex)
  len = ex isa Symbol ? 0 : length(ex.args)
  result = _parsescope(ex)
  if isnothing(result)
    ex, format = _parseinformat(ex)
    (_parsescope(ex)..., format)
  else
    (result..., nothing)
  end
end

#
# process fields
#

function _parsefieldformat(ex)
  ex, format = _parseinformat(ex)
  if ex isa Symbol # a in F
    [(ex, format)]
  elseif ex.head in [:tuple, :vect] && all(isa.(ex.args, Symbol)) # (a, b) in F
    map(name -> (name, format), ex.args)
  else
    error("Pack.@pack expected entry expression of form \"a [in F]\"")
  end
end

function _parsefieldformats(args)
  entries = (mapreduce(_parsefieldformat, vcat, args, init = []))
  (; entries...)
end

function _isselection(ex)
  ex isa Symbol || # a
  (ex isa Expr && ex.head == :tuple) || # (a, b; c) is parsed as tuple
  (ex isa Expr && ex.head == :block) || # (a; b) is not parsed as tuple but as block
  (ex isa Expr && ex.head == :call && ex.args[1] != :in) # C(args...; kwargs...)
end

function _parseselection(ex)
  constructor = nothing
  if ex isa Symbol
    names = ex
    nkwargs = 0
  elseif ex.head == :tuple # (...)
    if ex.args[1] isa Expr && ex.args[1].head == :parameters # (args...; kwargs...)
      names = [ex.args[2:end]; ex.args[1].args]
      nkwargs = length(ex.args[1].args)
    else # (args...)
      names = ex.args
      nkwargs = 0
    end
  elseif ex.head == :block && length(ex.args) == 3 # (a; b) not parsed as tuple
    names = [ex.args[1], ex.args[3]]
    nkwargs = 1
  elseif ex.head == :call && ex.args[1] != :in
    if ex.args[2] isa Expr && ex.args[2].head == :parameters # (args...; kwargs...)
      names = [ex.args[3:end]; ex.args[2].args]
      nkwargs = length(ex.args[2].args)
      constructor = ex.args[1]
    else # (args...)
      names = ex.args[2:end]
      nkwargs = 0
      constructor = ex.args[1]
    end
  else
    error("Pack.@pack expected entry list of form \"{field[s] in F, ...}\"")
  end
  @assert all(isa.(names, Symbol)) """
  Pack.@pack expected symbols that reflect field selection names.
  """
  Tuple(Symbol.(names)), nkwargs, constructor
end

#
# main macro
#

"""
    @pack T [in format] [field format customization] [field selection]
"""
macro pack(args...)
  @assert length(args) >= 1 """
  Pack.@pack expects at least one argument ($(length(args)) were given).
  """
  scopeformat_arg = args[1]
  format_args = filter(_isinformat, args[2:end])
  selection_arg = filter(_isselection, args[2:end])

  @assert length(selection_arg) <= 1 """
  Pack.@pack found more than one field selection expression.
  """
  @assert length(format_args) + length(selection_arg) == length(args) - 1 """
  Pack.@pack was unable to parse all argument expressions.
  """

  # Extract the type scope that the packing rules apply to and the (optional)
  # default format.
  body = []
  scope, tvar, fmt = _parsescopeformat(scopeformat_arg)

  # If a default format has been specified in the macro, add the respective
  # method to the body
  if !isnothing(fmt)
    if isnothing(tvar)
      expr = :(Pack.format(:: Type{$scope}) = $fmt())
    else
      expr = :(Pack.format(:: Type{$tvar}) where {$scope} = $fmt())
    end
    push!(body, expr)
  end

  # If format_args and selection_arg are not empty, additional methods for
  # Pack.destruct, Pack.construct, and Pack.valueformat will be defined.
  # In this case, we introduce a typevariable (if non has been specified by
  # the user).
  if tvar == nothing && scope isa Symbol
    tvar = gensym(:S)
    scope = :($scope <: $tvar <: $scope)
  elseif tvar == nothing
    tvar = gensym(:S) 
    scope = Expr(:(<:), tvar, scope.args[1])
  end

  # Define the methods Pack.destruct and Pack.construct if an explicit
  # field selection has been specified
  if length(selection_arg) == 1
    names, nkwargs, constructor = _parseselection(selection_arg[1])
    destruct = quote
      function Pack.destruct(value :: $tvar, :: Pack.MapFormat) where {$scope}
        Pack._destruct(value, $names)
      end
    end
    construct = quote
      function Pack.construct(:: Type{$tvar}, pairs, :: Pack.MapFormat) where {$scope}
        Pack._construct($tvar, pairs, $nkwargs, $constructor)
      end
    end
    push!(body, destruct, construct)
  end

  # If either field selections or a field format customizations have been
  # specified, the Pack.valueformat method has to be adapted, since the default
  # cannot be assumed to work anymore
  if length(selection_arg) > 0 || length(format_args) > 0
    if isempty(selection_arg)
      name = :(Base.fieldname($tvar, index))
    else
      name = :($names[index])
    end
    fieldformats = _parsefieldformats(format_args)
    valuetype = quote
      function Pack.valuetype(:: Type{$tvar}, index) where {$scope}
        name = $name
        Base.fieldtype($tvar, name)
      end
    end
    valueformat = quote
      function Pack.valueformat(:: Type{$tvar}, index) where {$scope}
        name = $name
        $(_valueformatexpr(:name, fieldformats))
      end
    end
    push!(body, valuetype, valueformat)
  end

  @assert !isempty(body) "Pack.@pack has received no packing instructions"
  Expr(:block, body...) |> esc
end



function _valueformatexpr(name, fieldformats)
  expr = nothing
  level = nothing
  for (key, format) in pairs(fieldformats)
    if isnothing(expr)
      expr = Expr(:if, :($name == $(QuoteNode(key))), :($format()))
      level = expr
    else
      tmp = Expr(:elseif, :($name == $(QuoteNode(key))), :($format()))
      push!(level.args, tmp)
      level = tmp
    end
  end
  if isempty(fieldformats)
    expr = :(Pack.DefaultFormat())
  else
    push!(level.args, :(Pack.DefaultFormat()))
  end
  expr
end

function _destruct(value :: T, names) where {T}
  Iterators.map(1:length(names)) do index
    key = names[index]
    val = getfield(value, key)
    (key, val)
  end
end

function _construct(:: Type{T}, pairs, nkwargs, constructor) where {T}
  len = length(pairs)
  @assert len >= nkwargs "inconsistent number of keyword arguments"
  args = []
  kwargs = []
  for (index, pair) in enumerate(pairs)
    if index <= len - nkwargs
      push!(args, pair[2])
    else
      push!(kwargs, pair)
    end
  end
  if isnothing(constructor)
    T(args...; kwargs...)
  else
    constructor(args...; kwargs...)
  end
end


macro typed(a)
  return :(nothing)
end

macro named(a)
  return :(nothing)
end

macro binarray(a, b)
  return :(nothing)
end

macro untyped(a)
  return :(nothing)
end

macro only(a, b)
  return :(nothing)
end
