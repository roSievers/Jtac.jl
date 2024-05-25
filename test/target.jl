
G = ToyGames.TicTacToe

vt = Target.DefaultValueTarget(G)
pt = Target.DefaultPolicyTarget(G)

policy = rand(Float32, 9)
policy ./= sum(policy)
ctx = Target.LabelContext(G(), policy, Game.loss, 0, G[], Vector{Float32}[])

@test length(vt) == 1
@test Target.label(vt, ctx) == [-1f0]
@test Target.defaultlossfunction(vt) == :sumabs2

@test length(pt) == Game.policylength(G)
@test Target.label(pt, ctx) == policy
@test Target.defaultlossfunction(pt) == :crossentropy

@test packcycle(vt)
@test packcycle(pt)

