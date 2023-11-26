
using Revise
using Random
using Jtac
using MsgPack
using BenchmarkTools

function packtest1(x, T = typeof(x); format = Pack.format(T))
  bin = Pack.pack(x, format)
  y = Pack.unpack(bin, T, format)
  all(bin .== Pack.pack(y, format))
end

function packtest2(x, T = typeof(x))
  bin1 = Pack.pack(x)
  bin2 = MsgPack.pack(x)
  v1 = MsgPack.unpack(bin1, T)
  v2 = Pack.unpack(bin2, T)
  bin11 = Pack.pack(v1)
  bin22 = Pack.pack(v2)
  all(bin11 .== bin11) && all(bin2 .== bin22)
end

function speedtest(x, T = typeof(x))
  println("speedtest for $x")
  bin = Pack.pack(x)
  println("Pack")
  @btime Pack.pack($x)
  @btime Pack.unpack($bin, $T)
  println("MsgPack")
  @btime MsgPack.pack($x)
  @btime MsgPack.unpack($bin, $T)
end

using Test

@testset "Nothing" begin
  @test packtest1(nothing)
  @test packtest2(nothing)
end
# speedtest(nothing)

@testset "Bool" begin
  @test packtest1(true)
  @test packtest2(true)
  @test packtest1(false)
  @test packtest2(false)
end
# speedtest(true)

@testset "Integer" begin
  for i in -100:100
    @test packtest1(i)
    @test packtest2(i)
  end
  for T in [Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64]
    i = rand(T)
    @test packtest1(i)
    i = typemin(T)
    # MsgPack also uses unsigned format for signed types, therefore loading
    # fails.
    @test packtest2(i)
  end
end
# speedtest(-13)
# speedtest(12)

@testset "String" begin
  @test packtest1("haha")
  @test packtest2("haha")
  @test packtest1(:haha)
  @test packtest2(:haha)
  for len in [2, 2^8-1, 2^8 + 1, 2^16 + 1]
    str = String(rand(UInt8, len))
    @test packtest1(str)
    @test packtest2(str)
  end
end
# speedtest("hahaha")

@testset "Float" begin
  f32 = rand(Float32)
  f64 = rand(Float64)
  @test packtest1(f32)
  @test packtest2(f32)
  @test packtest1(f64)
  @test packtest2(f64)
end
# speedtest(rand(Float32))
# speedtest(rand(Float32))

@testset "Vector" begin
  for n in [1, 10, 100, 1000]
    a = rand(n)
    @test packtest1(a)
    @test packtest2(a)
  end
  b = (-5, "test", 3.0, true, nothing, rand(5)) 
  @test packtest1(b)
  @test packtest2(b)
  c = (-5, "test", 3.0, true, nothing, rand(5), ("a", :b, 3)) 
  @test packtest1(c)
  # @test packtest2(c) # MsgPack.jl fails here
end

@testset "Array" begin
  for x in [rand(5, 5, 7), ["a" :b; 5 7.0], rand(Int32, 1000, 1000)]
    bytes = Pack.pack(x)
    val = Pack.unpack(bytes)
    @test haskey(val, "datatype") && haskey(val, "size") && haskey(val, "data")
    @test packtest1(x)
  end
  @test packtest1(rand(5), format = Pack.ArrayFormat())
end

@testset "BinVector" begin
  for n in [1, 10, 100, 1000]
    a = rand(n)
    @test packtest1(a, format = Pack.BinVectorFormat())
    # @test packtest2(a, format = Pack.BinVectorFormat())
  end
end

@testset "BinArray" begin
  for n in [1, 10, 100, 1000]
    a = rand(n)
    @test packtest1(a, format = Pack.BinArrayFormat())
    # @test packtest2(a, format = Pack.BinVectorFormat())
  end
  for x in [rand(5, 5, 7), rand(Int32, 1000, 1000)]
    bytes = Pack.pack(x, Pack.BinArrayFormat())
    val = Pack.unpack(bytes)
    @test haskey(val, "datatype") && haskey(val, "size") && haskey(val, "data")
    @test packtest1(x)
  end
end

# speedtest(rand(500))

@testset "Map" begin
  x = (a = 3, b = 7.0, c = "test")
  @test packtest1(x)
  x = Dict(pairs(x))
  @test packtest1(x)

  struct TestStruct
    a :: Int
    b :: Float64
    c :: String
  end
  Pack.format(:: Type{TestStruct}) = Pack.MapFormat()
  
  x = TestStruct(3, 7.0, "test")
  @test packtest1(x)
end

# speedtest(v1)
# speedtest(tuple(v1...))

