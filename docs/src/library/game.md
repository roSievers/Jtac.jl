
# Game
```@docs
Game
Game.AbstractGame
Game.Status
```

## Basic interface
```@docs
Game.status
Game.mover
Game.legalactions
Game.isactionlegal
Game.isover
Game.move!
Game.move
Game.instance
Game.visualize
Game.moves
Game.hash
```

## Randomization
```@docs
Game.randomaction
Game.randommove!
Game.randommove
Game.randomturn!
Game.randomturn
Game.rollout!
Game.rollout
Game.randominstance
```

## Augmentation
```@docs
Game.isaugmentable
Game.augment
```

## Implementations
```@docs
ToyGames.MNKGame
ToyGames.TicTacToe
ToyGames.MetaTac
```

## Matches
```@docs
Game.Match
Game.randommatch!
Game.randommatch
```
