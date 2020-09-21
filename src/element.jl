
# -------- Computational Element --------------------------------------------- #

"""
Abstract type to encapsulate the common interface, mainly CPU-GPU transfer, for
(i) neural network layers and (ii) neural network models. We distinguish between
the two, since a Jtac `Model` contains game-specific information while a `Layer`
is game agnostic.
"""
abstract type Element{GPU} end

"""
  swap(element)

Swap a model or layer `element` from/to gpu memory.
"""
swap(l :: Element) = error("Not implemented")

"""
  on_gpu(element)

Check whether `element` currently resides on the GPU.
"""
on_gpu(:: Element{GPU}) where {GPU} = GPU
on_gpu(:: Nothing) = false

"""
  to_cpu(element)

Move `element` to the CPU. Does nothing if the element is already there.
"""
to_cpu(el :: Element{false}) = el
to_cpu(el :: Element{true}) = swap(el)

"""
  to_gpu(element)

Try to move `element` to the GPU. Prints a warning if GPU is not available. Does
nothing if the element is already on the GPU.
"""
to_gpu(el :: Element{true})  = el

function to_gpu(el :: Element{false}) 
  if Knet.gpu() == -1
    @warn "No GPU was found by Knet. Element stays on CPU"
    el
  else
   swap(el)
  end
end

Base.copy(l :: Element) :: Element = error("Not implemented")

