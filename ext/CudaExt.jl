
module CudaExt

using CUDA
using Jtac
import Jtac.Util: register!
import Jtac.Model: Backend, DefaultBackend

# Model.releasememory!(x :: CuArray) = x.data.freed || CUDA.unsafe_free!(x) 

function __init__()
  register!(Backend, DefaultBackend{CuArray{Float32}}(), :cu, :cu32)
  register!(Backend, DefaultBackend{CuArray{Float64}}(), :cu64)
end

end
