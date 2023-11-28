# Implements a generic m, n, k game.
# https://en.wikipedia.org/wiki/M,n,k-game
# m: width
# n: height
# k: stones in a row required to win a game.

mutable struct MNKGame{M, N, K} <: AbstractGame
  board :: Vector{Int} # We may want https://github.com/JuliaArrays/StaticArrays.jl
  active_player :: Int
  status :: Status
  move_count :: Int
end

const TicTacToe = MNKGame{3, 3, 3}

MNKGame{M, N, K}() where {M, N, K} =
  MNKGame{M, N, K}(zeros(Int, M * N), 1, Game.undecided, 0)

function Base.copy(s :: MNKGame{M, N, K}) :: MNKGame{M, N, K} where {M, N, K}
  MNKGame{M, N, K}(
    copy(s.board), 
    s.active_player,
    s.status,
    s.move_count,
  )
end

isaugmentable(:: Type{MNKGame{M, N, K}}) where {M, N, K} = true

function Base.:(==)(a::MNKGame{M, N, K}, b::MNKGame{M, N, K}) where {M, N, K}
  all([ all(a.board .== b.board)
      , a.active_player == b.active_player
      , a.status == b.status
      , a.move_count == b.move_count])
end

activeplayer(game :: MNKGame) :: Int = game.active_player

# Returns a list of the Indices of all legal actions
function legalactions(game :: MNKGame{M, N}) :: Vector{ActionIndex} where {M, N}
  if isover(game)
    ActionIndex[]
  else
    result = Vector{ActionIndex}(undef, M*N - game.move_count)
    i = 1
    for j in 1:(M*N)
      if game.board[j] == 0
        result[i] = ActionIndex(j)
        i += 1
      end
    end
    @assert size(result)[1] + 1 == i "There should be " *
            "$(size(result)[1] + 1) legal actions but we found $(i)."

    result
  end
end

# A action is legal, if the board position is still empty
function isactionlegal( game :: MNKGame{M, N}
                      , index :: ActionIndex ) :: Bool where {M, N}
  game.board[index] == 0 && !isover(game)
end

function move!( game :: MNKGame{M, N, K}
              , index :: ActionIndex
              ) :: MNKGame{M, N, K} where {M, N, K}

  @assert isactionlegal(game, index) "Action $index is not allowed."
  # Update the board state
  game.board[index] = game.active_player
  game.active_player = -game.active_player
  
  # Update the status cache
  game.move_count += 1
  game.status = tic_tac_mnk_status(game, index)
  game

end

# All possible directions for winning rows, in flat index notation.
const search_directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

bound_check(m, n, fx, fy) = (fx > 0 && fx <= m && fy > 0 && fy <= n)

function tic_tac_mnk_status( game :: MNKGame{M, N, K}
                           , changed_index :: ActionIndex
                           ) where {M, N, K}

  ci = CartesianIndices((M, N)) 
  x, y = Tuple(ci[changed_index])
  moving = game.board[changed_index]

  for (dx, dy) in search_directions
    count = 1 # There is always the freshly placed token.
    # forward search along the direction
    # Seek using a focus point at (fx, fy).
    (fx, fy) = (x + dx, y + dy)
    while bound_check(M, N, fx, fy) && game.board[fx + (fy - 1) * M] == moving
      count += 1
      fx += dx
      fy += dy
    end
    # backward search along the direction
    (fx, fy) = (x - dx, y - dy)
    while bound_check(M, N, fx, fy) && game.board[fx + (fy - 1) * M] == moving
      count += 1
      fx -= dx
      fy -= dy
    end

    if count >= K
      return Status(moving)
    end
  end

  # Check if there are empty spaces left.
  if game.move_count == M*N
    Game.draw
  else
    Game.undecided
  end

end

status(game :: MNKGame) :: Status = game.status

function policylength(:: Type{<:MNKGame{M, N}}) :: Int where {M, N}
  M * N
end

# Size of the data representation of the game
function Base.size(:: Type{<:MNKGame{M, N}}) :: Tuple{Int, Int, Int} where {M, N}
  (M, N, 1)
end

# Data representation of the game as layered 2d image
function array(game :: MNKGame{M, N}) :: Array{Float32, 3} where {M, N}
  reshape(game.active_player .* game.board, (M, N, 1))
end

# Augment single games
function augment(game :: MNKGame{M, N, K}) where {M, N, K}

  if M == N
    boards = applygroup(DihedralGroup(), reshape(game.board, (M, N))) 
    map(boards) do b
      MNKGame{M, N, K}( reshape(b, (M * N,))
                      , game.active_player
                      , game.status
                      , game.move_count )
    end

  else
    boards = applygroup(KleinFourGroup(), reshape(game.board, (M, N))) 
    map(boards) do b
      MNKGame{M, N, K}( reshape(b, (M * N,))
                      , game.active_player
                      , game.status
                      , game.move_count )
    end

  end

end

# Augment game/label pairs
function augment( game :: MNKGame{M, N, K}
                , policy :: Vector{Float32}
                ) where {M, N, K}
  if M == N

    matpol = reshape(policy, (M, N))
    matpols = applygroup(DihedralGroup(), matpol)
    policies = [ vcat(reshape(mp, (M * N,))) for mp in matpols ]

  else

    matpol = reshape(policy, (M, N))
    matpols = applygroup(KleinFourGroup(), matpol)
    policies = [ vcat(reshape(mp, (M * N,))) for mp in matpols ]

  end

  augment(game), policies
end

hash(game :: MNKGame) = Base.hash(game.board)


function visualize(io :: IO, game :: MNKGame{M, N}) :: Nothing where {M, N}
  board = reshape(game.board, (M,N))
  symbols = Dict(1 => "X", -1 => "O", 0 => "⋅")

  for i in 1:N
    for j in 1:M
      print(io, " $(symbols[board[j,i]])")
      if j != M
        print(io, " │")
      end
    end
    if i != N
      println(io)
      println(io, "───$(repeat("┼───", M - 1))")
    end
  end
end

visualize(game :: MNKGame) = visualize(stdout, game)

function Base.show(io :: IO, game :: MNKGame{M, N, K}) where {M, N, K}
  moves = count(!isequal(0), game.board)
  m = moves == 1 ? "1 move" : "$moves moves"
  if isover(game)
    print(io, "MNKGame{$M, $N, $K}($m, $(status(game)) won)")
  else
    print(io, "MNKGame{$M, $N, $K}($m, $(activeplayer(game)) moving)")
  end
end

function Base.show(io :: IO, :: MIME"text/plain", game :: MNKGame{M, N, K}) where {M, N, K}
  moves = count(!isequal(0), game.board)
  m = moves == 1 ? "1 move" : "$moves moves"
  s = "MNKGame{$M, $N, $K} with $m and "
  if isover(game)
    println(io, s, "result $(status(game)):")
  else
    println(io, s, "player $(activeplayer(game)) moving:")
  end
  draw(io, game)
end




