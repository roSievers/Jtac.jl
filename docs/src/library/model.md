# Model

```@docs
Model
Model.AbstractModel
```

## Interface
```@docs
Model.apply
Model.assist
Model.gametype
Model.ntasks
Model.isasync
Model.basemodel
Model.childmodel
Model.trainingmodel
Model.playingmodel
Model.targets
Model.targetnames
Model.configure
Model.save
Model.load
```
## Simple models
```@docs
Model.RolloutModel
Model.DummyModel
Model.RandomModel
```

## Wrapper models
```@docs
Model.AsyncModel
Model.CachingModel
Model.AssistedModel
```

## Neural network models
```@docs
Model.NeuralModel
```

### Layers
```@docs
Model.Activation
Model.Layer
Model.Dense
Model.Conv
Model.Batchnorm
Model.Chain
Model.Residual
```

### Backends
```@docs
Model.Backend
Model.DefaultBackend
```

### Auxiliary methods
```@docs
Model.addtarget!
Model.aligndevice!
Model.layers
Model.isvalidinputsize
Model.isvalidinput
Model.outputsize
Model.parameters
Model.parametercount
Model.getbackend
Model.istrainable
Model.arraytype
Model.adapt
```

### Architectures
```@docs
Model.Zoo.ZeroConv
Model.Zoo.ZeroRes
Model.Zoo.ShallowConv
Model.Zoo.MLP
Model.Zoo.zero_head
```

## Formats
```@docs
Model.Format
Model.DefaultFormat
Model.extension
Model.isformatextension
```
