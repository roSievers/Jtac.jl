
@test packcycle(nothing)
@test packcycle(true)
@test packcycle(false)
for v in [-100000000000, -1000, -100, -10, 0, 10, 100, 1000, 10000000000]
  @test packcycle(v)
end
for v in [rand(Float32), rand(Float64)]
  @test packcycle(v)
end
for v in [randstring(n) for n in (3, 16, 32, 100, 1000)]
  @test packcycle(v)
end
@test packcycle(rand(Float32, 1000))
@test packcycle(rand(Float64, 1000))
@test packcycle((:this, :is, "a tuple", (:with, true, :numbers), 5))
@test packcycle((a = "named", b = "tuple", length = 3))

