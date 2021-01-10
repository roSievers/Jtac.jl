

#include("bin/jtac.jl")
#include("bin/train.jl")

using Jtac
using JtacServer

import Sockets
import .JtacServer: ServeLogin, LoginAuth,
                    Message, Train, Serve,
                    TrainDataServe, ServeData

import .JtacServer: receive, send, build_player

function login() 
  login = JtacServer.ServeLogin("tom-serve", "12345")
  sock = Sockets.connect("127.0.0.1", 7788)
  send(sock, login)
  auth = JtacServer.receive(sock, JtacServer.LoginAuth)

  if auth.accept
    println("login accepted")
    sock
  else
    println("login rejected")
  end
end

function receive_request(sock)
  req = receive(sock, Message{Train, Serve})
  if req isa TrainDataServe
    println("received data request D$(req.reqid)")
    req
  else
    println("do not understand request of type $(typeof(req))")
  end
end

function send_data(sock, req, n = 100)
  player = build_player(req.spec)
  println("player built")
  augment = req.augment
  prep = prepare(steps = req.init_steps)
  bran = branch(prob = req.branch, steps = req.branch_steps)
  println("producing selfplays")
  data = nothing
  time = @elapsed begin
    data = record_self(player, n, merge = false, prepare = prep, branch = bran, augment = augment)
  end
  send(sock, ServeData(data, 1, req.reqid, time))
  println("selfplays sent!")
end

