
using Jtac

i = 0
ds = record_self(MCTSPlayer(power=10), 10, game = MetaTac(), branching = 0.1, callback = () -> (global i+=1; println(i)))
