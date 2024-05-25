
# Player
```@docs
Player
Player.AbstractPlayer
```

## Interface
```@docs
Player.think
Player.apply
Player.decide
Player.decideturn
Player.move!
Player.turn!
Player.name
Player.ntasks
Player.gametype
```

## MCTS players
```@docs
Player.MCTSPlayer
Player.MCTSPlayerGumbel
```
### Policy extraction
```@docs
Player.MCTSPolicy
Player.ModelPolicy
Player.VisitCount
Player.ImprovedPolicy
Player.Anneal
Player.Lognormal
Player.Gumbel
```

### Action selection
```@docs
Player.ActionSelector
Player.SampleSelector
Player.MaxSelector
Player.PUCT
Player.VisitPropTo
Player.SequentialHalving
```

## Other players
```@docs
Player.IntuitionPlayer
Player.HumanPlayer
Player.RandomPlayer
```
