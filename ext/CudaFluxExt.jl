
module CudaFluxExt

using Jtac
import .Model: Backend, DefaultBackend
import Jtac.Util: register!

import CUDA: CuArray

function __init__()
  FluxExt = Base.get_extension(Jtac, :FluxExt)
  FluxBackend = FluxExt.FluxBackend

  register!(Backend, FluxBackend{CuArray{Float32}}(), :cudaflux, :cuflux, :cudaflux32, :cuflux32)
  register!(Backend, FluxBackend{CuArray{Float16}}(), :cudaflux16, :cuflux16)
  register!(Backend, FluxBackend{CuArray{Float64}}(), :cudaflux64, :cuflux64)
  Model._defaultconfig[:backend] = :cudaflux
end

end
