
module CudaFluxExt

import Jtac
import Jtac.Util: register!
import Jtac.Model: Backend, DefaultBackend

import CUDA: CuArray

function __init__()
  FluxExt = Base.get_extension(Jtac, :FluxExt)
  FluxBackend = FluxExt.FluxBackend

  register!(Backend, FluxBackend{CuArray{Float32}}(), :cuflux, :cuflux32)
  register!(Backend, FluxBackend{CuArray{Float64}}(), :cuflux64)
end

end
