[![Build Status](https://travis-ci.org/roSievers/Jtac.jl.svg?branch=master)](https://travis-ci.org/roSievers/Jtac.jl)

# Jtac

A julia package to train AIs that can play two-player boardgames. It is based on
the Alpha-Zero paradigm and is inspired by open implementations like KataGo.

# Games

The following games are currently part of Jtac:
- [M,n,k-game](https://en.wikipedia.org/wiki/M,n,k-game) (including tic-tac-toe)
- [Meta tic-tac-toe](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)
- [Three men's morris](https://en.wikipedia.org/wiki/Three_men%27s_morris)

Extending Jtac by adding support for new games is easy and independent from the
rest of Jtac's functionality.

# Usage
Some tutorial will soon follow here

We are also writing a frontend in Elm to help with inspecting models. It lives at https://github.com/kreibaum/Jtac-frontend

# Links

- [Blogpost by Surag Nair](https://web.stanford.edu/~surag/posts/alphazero.html)
- [Alpha
  Zero](https://kstatic.googleusercontent.com/files/2f51b2a749a284c2e2dfa13911da965f4855092a179469aedd15fbe4efe8f8cbf9c515ef83ac03a6515fa990e6f85fd827dcd477845e806f23a17845072dc7bd)
- [KataGo](https://arxiv.org/pdf/1902.10565.pdf)
- [ELF OpenGo](https://arxiv.org/pdf/1902.04522.pdf)
- [SAI](https://arxiv.org/pdf/1905.10863.pdf)
