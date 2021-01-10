
module JtacServer

const JTAC_SERVER_VERSION = "v0.1"

import Jtac
import Knet
import CUDA

import Random
import Distributed
import Distributed: @ip_str
import Serialization
import Sockets
import Statistics
import Base.Filesystem

import Blosc
import LazyJSON
import Dates

include("log.jl")
include("compress.jl")
include("msg.jl")
include("pool.jl")
include("train.jl")
include("serve.jl")
#include("ai.jl")

export Jtac, Knet
export Context, train, serve

end # module JtacServer

