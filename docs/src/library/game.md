
# Game
```@docs
Game
Game.AbstractGame
Game.Status
```

## Basic interface
```@docs
Game.status
Game.activeplayer
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
Game.randommatch!
Game.randommatch
Game.randominstance
Game.branch
```

## Tensor representation
```@docs
Game.array
Game.array!
Game.arraybuffer
```

## Augmentation
```@docs
Game.isaugmentable
Game.augment
```

## Implementations
```@docs
Game.MNKGame
Game.TicTacToe
Game.MetaTac
```
