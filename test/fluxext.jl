
import Jtac.Model: DefaultBackend

@testset "Dense" begin
  for backend in [:flux16, :flux, :flux64]
    for l in [
      Model.Dense(5, 7, :relu),
      Model.Dense(5, 7, sin, bias = false),
    ]
      T = Model.arraytype(Model.getbackend(backend))
      l = Model.adapt(DefaultBackend{T}(), l)
      lflux = Model.adapt(backend, l)
      l2 = Model.adapt(DefaultBackend{T}(), lflux)

      @test Model.getbackend(l) isa DefaultBackend{T}
      @test Model.getbackend(l2) isa DefaultBackend{T}
      @test Model.getbackend(lflux) isa typeof(Model.getbackend(backend))

      @test l isa Model.Dense{T}
      @test l2 isa Model.Dense{T}

      data = rand(eltype(T), 5, 10)
      r1 = l(data)
      r2 = l2(data)
      r3 = lflux(data)

      @test all(r1 .== r2 .== r3)
    end
  end
end

@testset "Conv" begin
  for backend in [:flux16, :flux, :flux64]
    for l in [
      Model.Conv(5, 7, :relu, window = 3, pad = 1),
      Model.Conv(5, 7, sin, window = 3, pad = 2, bias = false),
    ]
      T = Model.arraytype(Model.getbackend(backend))
      l = Model.adapt(DefaultBackend{T}(), l)
      lflux = Model.adapt(backend, l)
      l2 = Model.adapt(DefaultBackend{T}(), lflux)

      @test Model.getbackend(l) isa DefaultBackend{T}
      @test Model.getbackend(l2) isa DefaultBackend{T}
      @test Model.getbackend(lflux) isa typeof(Model.getbackend(backend))

      @test l isa Model.Conv{T}
      @test l2 isa Model.Conv{T}

      data = rand(eltype(T), 3, 6, 5, 10)
      r1 = l(data)
      r2 = l2(data)
      r3 = lflux(data)

      @test all(r1 .== r2 .== r3)
    end
  end
end

@testset "Batchnorm" begin
  for backend in [:flux16, :flux, :flux64]
    for l in [
      Model.Batchnorm(3, :relu),
      Model.Batchnorm(3, sin),
    ]
      T = Model.arraytype(Model.getbackend(backend))
      l = Model.adapt(DefaultBackend{T}(), l)
      lflux = Model.adapt(backend, l)
      l2 = Model.adapt(DefaultBackend{T}(), lflux)

      @test Model.getbackend(l) isa DefaultBackend{T}
      @test Model.getbackend(l2) isa DefaultBackend{T}
      @test Model.getbackend(lflux) isa typeof(Model.getbackend(backend))

      @test l isa Model.Batchnorm{T}
      @test l2 isa Model.Batchnorm{T}

      data = rand(eltype(T), 2, 2, 3, 1)
      r1 = l(data)
      r2 = l2(data)
      r3 = lflux(data)

      @test all(isapprox(r1, r2) && isapprox(r2, r3))
    end
  end
end

@testset "Chain" begin
  for backend in [:flux16, :flux, :flux64]
    T = Model.arraytype(Model.getbackend(backend))
    chain = Model.@chain (5, 5, 3) Conv(10) Batchnorm(:relu) Dense(42)
    for l in [ chain ]
      l = Model.adapt(DefaultBackend{T}(), l)
      lflux = Model.adapt(backend, l)
      l2 = Model.adapt(DefaultBackend{T}(), lflux)

      @test Model.getbackend(l) isa DefaultBackend{T}
      @test Model.getbackend(l2) isa DefaultBackend{T}
      @test Model.getbackend(lflux) isa typeof(Model.getbackend(backend))

      @test l isa Model.Chain{T}
      @test l2 isa Model.Chain{T}

      data = rand(eltype(T), 5, 5, 3, 1)
      r1 = l(data)
      r2 = l2(data)
      r3 = lflux(data)

      @test all(isapprox(r1, r2) && isapprox(r2, r3))
    end
  end
end

@testset "Residual" begin
  for backend in [:flux16, :flux, :flux64]
    T = Model.arraytype(Model.getbackend(backend))
    res = Model.@residual (5, 5, 3) Conv(3, pad = 1) Batchnorm(:relu) Conv(3, pad = 1)
    for l in [ res ]
      l = Model.adapt(DefaultBackend{T}(), l)
      lflux = Model.adapt(backend, l)
      l2 = Model.adapt(DefaultBackend{T}(), lflux)

      @test Model.getbackend(l) isa DefaultBackend{T}
      @test Model.getbackend(l2) isa DefaultBackend{T}
      @test Model.getbackend(lflux) isa typeof(Model.getbackend(backend))

      @test l isa Model.Residual{T}
      @test l2 isa Model.Residual{T}

      data = rand(eltype(T), 5, 5, 3, 1)
      r1 = l(data)
      r2 = l2(data)
      r3 = lflux(data)

      @test all(isapprox(r1, r2) && isapprox(r2, r3))
    end
  end
end

@testset "Model" begin
  for backend in [:flux16, :flux, :flux64]
    T = Model.arraytype(Model.getbackend(backend))
    G = Game.TicTacToe
    model = Model.Zoo.ZeroRes(G, filters = 8, blocks = 3)
    for l in [model.trunk]
      l = Model.adapt(DefaultBackend{T}(), l)
      lflux = Model.adapt(backend, l)
      l2 = Model.adapt(DefaultBackend{T}(), lflux)

      @test Model.getbackend(l) isa DefaultBackend{T}
      @test Model.getbackend(l2) isa DefaultBackend{T}

      games = [Game.randominstance(G) for _ in 1:10]
      data = convert(T, Game.array(games))

      r1 = l(data)
      r2 = l2(data)
      r3 = lflux(data)

      @test all(isapprox(r1, r2) && isapprox(r2, r3))
    end
  end
end
