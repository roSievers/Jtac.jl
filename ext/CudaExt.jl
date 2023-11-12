
module CudaExt
using CUDA
using Jtac
import Jtac.Util: register!
import Jtac.Model: Backend, DefaultBackend

Model.releasememory!(x :: CuArray) = CUDA.unsafe_free!(x) 

function __init__()
  register!(Backend, DefaultBackend{CuArray{Float32}}(), :cuda, :cuda32)
  register!(Backend, DefaultBackend{CuArray{Float64}}(), :cuda64)
end

end
