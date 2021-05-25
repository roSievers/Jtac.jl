
module JtacServer

# base packages
import Random
import Distributed
import Distributed: @ip_str
import Serialization
import Sockets
import Statistics
import Base.Filesystem
import Dates

# deep learning packages
import Knet
import CUDA

# compression and communication packages
import Blosc
import LazyJSON

# Jtac core library
using Jtac

# constants
const JTAC_VERSION = "v0.1"
const DEBUG = Ref{Bool}(false)

# exceptions
struct WorkerStopException <: Exception end

# utilities
include("exit.jl")
include("compress.jl")

# training context and data pools
include("context.jl")
include("pool.jl")

# events and messages to other jtac services
include("events.jl")
include("msg.jl")

# cli logging
include("log.jl")

# jtac services
include("train.jl")
include("play.jl")
#include("ai.jl")

export Game, Model, Player, Training, Knet
export Context, train, serve
export @ip_str

end # module JtacServer

