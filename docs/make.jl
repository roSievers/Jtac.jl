using Documenter, Jtac

makedocs(
  sitename="Jtac.jl Documentation",
  # modules = [Jtac],
  pages = [
    "Tutorial" => "tutorial.md",
    "Library" => [
      "Game" => "library/game.md",
      "Target" => "library/target.md",
      "Model" => "library/model.md",
      "Player" => "library/player.md",
      "Training" => "library/training.md",
      "Pack" => "library/pack.md",
    ]
  ]
)