using CairoMakie, MakiePublication

with_theme(theme_web()) do 
  fig = Figure()
  ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

  l1 = lines!(ax, vec(t), y_pred; linewidth=3)
  l2 = lines!(ax, vec(t), y; linewidth=3)

  fig
end
