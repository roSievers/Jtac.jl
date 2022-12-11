

using Jtac
using .Game
using .Model
using .Player
using .Training



module Api



login(; name :: String) = Dict{Symbol}(
    :client_name => name
)

function login(url, port; name)
end

action: login
  arguments: client_name
  returns: client_id, session_id

#action: logout
#  arguments: client_id, session_id
#  returns: logout confirmation

action: status
  arguments: client_id, session_id


action: get_history
  arguments: client_id, session_id [, event_id]
  returns: history starting from the entry after hash

action: get_model
  arguments: client_id, session_id [, model generation, model ]
  returns: nothing if hashes match, else model of generation

action: upload_data
  arguments: client_id, session_id, model generation, dataset
  returns: success message

action: upload_contest
  arguments: client_id, session_id, contest data
  returns: success message

action: modify
  arguments: password, [variable_name: new_value ...]
  returns success message



end


function create_channels(G)
  [ :datasets => Channel{DataSet{G}}()
  , :generations => Channel{NeuralModel{G}}
end


function train( player :: MCTSPlayer{G}
              ; batchsize :: Int = 512
              , gensize = batchsize = 100
              , generations = 0
              , capacity :: Int
              , snapshot_interval = 5
              , gpu = 0
              , port :: Int
              , ) where {G <: AbstractGame}

  model = training_model(player)
  @assert model isa NeuralModel{G} "Invalid training model: expected neural model"

  models = NeuralModel{G}[]
  push!(models, copy(model |> to_cpu))

  data = Tuple{DataSet{G}, Int}[]

  history = Dict{String, Any}[]

  locks = Dict{Symbol, ReentrantLock}
  locks[:data] = ReentrantLock()
  locks[:model] = ReentrantLock()
  locks[:history] = ReentrantLock()

  @sync begin

    @async begin
      # IO stuff
    end

  end


end



