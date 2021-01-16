
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
import Jtac
import Knet
import CUDA

# compression and communication packages
import Blosc
import LazyJSON


# constants
const VERSION = "v0.1"
const DEBUG = Ref{Bool}(true)

# exceptions
struct WorkerStopException <: Exception end

# utilities
include("exit.jl")
include("compress.jl")
include("msg.jl")
include("pool.jl")
include("log.jl")

# services
include("train.jl")
include("serve-v2.jl")
#include("ai.jl")

export Jtac, Knet
export Context, train, serve
export @ip_str

end # module JtacServer

