using CairoMakie, MakiePublication

with_theme(theme_web()) do 
  fig = Figure()
  ax = CairoMakie.Axis(fig[1, 1]; xlabel="t", ylabel="u")

  l1 = lines!(ax, t, y_pred; linewidth=3)
  l2 = lines!(ax, t, y_true; linewidth=3)

  axislegend(ax, [l1, l2], ["PINN", "True Dynamics"])

  fig
end
