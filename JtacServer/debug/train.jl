import Distributed
import Distributed: @everywhere
using Sockets

if length(Distributed.workers()) > 1
  worker = Distributed.workers()[2]
else
  worker = Distributed.addprocs(1)[1]
end

using Jtac
@everywhere using JtacServer

model = NeuralModel(TicTacToe, @chain TicTacToe Conv(64, relu) Dense(32, relu))
#model = ""

options = (
    batch_size = 50
  , learning_rate = 0.01
  , epoch_size = 6000
  , era_size = 60000
  , test_frac = 0.05
  , backups = 0
  , max_use = 10
  , max_age = 1
  , power = 50
)

name = "test-2"
port = 7788
token = "12345"
ip = ip"0.0.0.0"

context = JtacServer.Context(0; options...)
JtacServer.train(model, name, context, ip = ip, port = port, token = token, worker = worker)

