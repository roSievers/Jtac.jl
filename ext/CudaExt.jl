
module CudaExt

using CUDA
using Jtac
import Jtac.Util: register!
import Jtac.Model: Backend, DefaultBackend, NeuralModel

Model.releasememory!(x :: CuArray) = x.data.freed || CUDA.unsafe_free!(x)

function Model.aligndevice!( model :: NeuralModel{G, B}
                           ) where {G, T <: CuArray, B <: Backend{T}}
  params = Model.parameters(model)
  @assert !isempty(params) "Unexpected empty parameter list for model"
  dev = CUDA.device(params[1]) # check which device the array lives on
  CUDA.device!(dev) # mark the device active
end

function __init__()
  register!(Backend, DefaultBackend{CuArray{Float32}}(), :cu, :cu32)
  register!(Backend, DefaultBackend{CuArray{Float16}}(), :cu16)
  register!(Backend, DefaultBackend{CuArray{Float64}}(), :cu64)
end

end
