using CairoMakie, MakiePublication

# with_theme(theme_web()) do
#   fig = Figure()
#   ax = CairoMakie.Axis(fig[1, 1]; xlabel="t", ylabel="u")

#   l1 = lines!(ax, xs, ys; linewidth=3)
#   l2 = lines!(ax, xs, y_learned; linestyle=:dot, linewidth=3)

#   axislegend(ax, [l1, l2], ["True Dynamics", "Learned Dynamics"])

#   fig
# end

begin
  fig = Figure()
  ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

  _θs = collect(0:1.0f-2:Float32(2π))

  _rs = r.(_θs)
  _xs = _rs .* cos.(_θs)
  _ys = _rs .* sin.(_θs)

  _rs_pred = vec(first(model(reshape(_θs, 1, :), sol.u, st)))
  _xs_pred = _rs_pred .* cos.(_θs)
  _ys_pred = _rs_pred .* sin.(_θs)

  l1 = lines!(ax, _xs, _ys; linewidth=3)
  l2 = lines!(ax, _xs_pred, _ys_pred; linestyle=:dot, linewidth=3)

  axislegend(ax, [l1, l2], ["True", "Learned"]; position=:ct)

  fig
end

save(joinpath(@__DIR__, "../figures/lux/cmaes_plot.pdf"), fig, pt_per_unit=0.75)