
using Jtac

G = MetaTac

chain = @chain G begin
          @stack stack_input = true Conv(512, relu, stride = 3) Conv(1024, relu)
	  Dense(1024, relu)
          Dropout(0.3)
	  Dense(512, relu)
          Dropout(0.3)
          Dense(256, relu)
          Dropout(0.3)
          Dense(256, relu)
        end

#chain = @chain G Conv(2048) Dense(1024) Dense(512) Dense(256) Dense(128)
model = NeuralModel(G, chain) |> to_gpu |> Async
player = MCTSPlayer(model, power = 250)
#player = MCTSPlayer(power = 50)


