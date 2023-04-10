using CairoMakie, MakiePublication

begin
  fig = Figure()

  for (idx, method) in enumerate(["LN_COBYLA", "CMAES"])
    ax = CairoMakie.Axis(fig[1, idx]; xlabel="x", ylabel=idx == 1 ? "y" : "", title=method)

    _θs = collect(0:1.0f-2:Float32(2π))

    _rs = r.(_θs)
    _xs = _rs .* cos.(_θs)
    _ys = _rs .* sin.(_θs)

    _rs_pred = vec(first(model(reshape(_θs, 1, :), ComponentArray(sols[method], ps_ax), st)))
    _xs_pred = _rs_pred .* cos.(_θs)
    _ys_pred = _rs_pred .* sin.(_θs)

    l1 = lines!(ax, _xs, _ys; linewidth=3)
    l2 = lines!(ax, _xs_pred, _ys_pred; linestyle=:dot, linewidth=3)

    idx == 1 && axislegend(ax, [l1, l2], ["True", "Learned"]; position=:ct)
  end

  fig
end

save(joinpath(@__DIR__, "../figures/lux/gfopt_plot.pdf"), fig, pt_per_unit=0.75)