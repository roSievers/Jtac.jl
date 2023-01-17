
module Play

using Printf

# LanguageServer bug :)
# using Jtac

include("../src/Jtac.jl")
using .Jtac
import .Server: Config, Log, Program
import .Log: @log, @logerror, @logdebug, @logwarn
import .Program: SelfplayProgram

struct Settings
  config :: Dict
  session_id :: UInt
  program :: Union{Nothing, SelfplayProgram}
  ch :: Dict
end

function Settings(config)
  session_id = rand(UInt)
  ch = Dict(:stop => Channel{Bool}(1))
  Settings(config, session_id, nothing, ch)
end

function Log.log(:: Settings, level, origin, msg)
  if level == 0
    @error "$origin: $msg"
  elseif level == 1
    @warn "$origin: $msg"
  elseif level == 2
    @info "$origin: $msg"
  elseif level == 3
    @info "[DEBUG] $origin: $msg"
  end
end

struct ConfigError <: Exception end
config_error() = throw(ConfigError())


function start(config)
  s = Settings(config)
  @log s "Initializing jtac client session $(s.session_id)"

  try
    @async handle_user_shutdown(s)
    isolated = determine_mode(s)
    if isolated
      run_isolated(s)
    else
      run_remote(s)
    end

  catch err
    if err isa ConfigError
      @logerror s "Client was stopped due to a configuration error."
    else
      @logerror s "Unexpected error: $err"
    end
  end
end

function determine_mode(s :: Settings)
  c = s.config
  if !haskey(c, :client)
    @log s "Provided configuration misses :client section. Assuming defaults"
    c[:client] = Config.client()
  end
  if c[:client][:isolated]
    @log s "Isolated mode enabled"
    if !haskey(c, :selfplay)
      @logerror s "Configuration section :selfplay required in isolated mode."
      config_error()
    elseif !haskey(c[:selfplay], :model) || isempty(c[:selfplay][:model])
      @logerror s "Selfplay model required in isolated mode."
      config_error()
    end
    model = c[:selfplay][:model]
    @log s "Will run selfplays with local model $model"
    @log s "Options: $(c[:selfplay])"
    true
  else
    @log s "Remote mode enabled. Will connect to server at "
    false
  end
end

function handle_user_shutdown(s :: Settings)
  @log s "Client can be stopped gracefully by pressing <ctrl>+d"
  read(stdin) # block until ctrl-d is typed
  put!(s.ch[:stop], true)
  if !isnothing(s.program)
    Program.stop(s.program)
  end
end


# -------- Isolated mode ----------------------------------------------------- #

function run_isolated(s :: Settings)
  model = nothing
  try
    path = s.config[:selfplay][:model]
    model = Model.load(path)
    G = Model.gametype(model)
    @log s "Loaded model $path for game $G"
  catch err
    @logerror s "Error while loading model: $err"
    config_error()
  end

  s.program = SelfplayProgram(s.config[:selfplay])
  isready(s.ch[:stop]) && return
  @logdebug s "Selfplay program created and attached"

  put!(s.program.player, (model, (;), s.config[:selfplay]))

  @sync begin
    Threads.@spawn Program.start(s.program)
    @async save_data(s)
    @async handle_program_stop(s)
    @async handle_program_log(s)
  end
end

# Should run forever, except if program is shut down.
# Then, k.data is closed and we break.
# If an unexpected exception happens, the program is interrupted
function save_data(s :: Settings)
  try
    sess = string(s.session_id, base = 16)
    data_size = get(s.config[:client], :data_size, 10_000)
    data_folder = get(s.config[:client], :data_folder, "./data")
    data_folder = joinpath(data_folder, sess)
    mkpath(data_folder)

    @log s "Will use folder '$data_folder' to save data sets"
    @log s "Size of data sets to be saved is $data_size"

    count = 0
    while isready(s.program.data)
      datasets = []
      len = 0

      while len <= data_size
        # We ignore the context. It is only interesting
        # in remote mode.
        ds, _ = try take!(s.program.data) catch _ break end
        len += length(ds)
        push!(datasets, ds)
      end

      if !isempty(datasets)
        name = @sprintf "%6d.jtd" count
        path = joinpath(data_folder, name)
        data = merge(datasets)
        Data.save(path, data)
        @log s "Saved data set $count under '$path'"
        count += 1
      end
    end
    @logdebug s "returning"

  catch err
    @logerror "Unexpected error: $err"
    Program.interrupt(k)
  end
end

function handle_program_log(s :: Settings)
  state = Program.getstate(s.program)
  try
    while true
      log = try take!(state.log) catch _ break end
      source = "selfplay:$(log.source)"
      log(s, log.level, source, log.msg)
    end
    @logdebug s "returning"

  catch err
    @logerror s "Unexpected error: $err"
    Program.interrupt(s.program)
  end
end

function handle_program_stop(s :: Settings)
  try
    fetch(s.ch[:stop])
    Program.stop(s.program)
  catch err
    @logerror s "Unexpected error: $err"
    Program.interrupt(s.program)
  end
end


# -------- Remote mode ------------------------------------------------------- #

function run_remote(s :: Settings)
  # TODO
  s
end

end # module Play
