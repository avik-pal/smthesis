using CairoMakie, MakiePublication, Zygote

predict(θ, ts) = Array(solve(prob, Tsit5(); p=θ, saveat=ts))

ts__ = 0.0f0:1f-2:1.0f0
learned = predict(res.u, ts__)

fig = begin
  fig = Figure()

  ax = CairoMakie.Axis(fig[1, 1]; xlabel="t")

  l1 = lines!(ax, ts__, learned[1, :]; linewidth=3)
  l2 = lines!(ax, ts__, learned[2, :]; linestyle=:dot, linewidth=3)
  l3 = lines!(ax, ts__, learned[3, :]; linestyle=:dash, linewidth=3)
  l4 = lines!(ax, ts__, learned[4, :]; linestyle=:dashdotdot, linewidth=3)

  s1 = scatter!(ax, ts, y_true[1, :]; markersize=10)
  s2 = scatter!(ax, ts, y_true[2, :]; markersize=10)

  axislegend(ax, [l1, l2, l3, l4, s1, s2], [L"x", L"y", L"u_x", L"u_y", L"x_{true}", L"y_{true}"]; position=:lc)

  fig
end

save(joinpath(@__DIR__, "../figures/lux/diffeq_plot.pdf"), fig, pt_per_unit=0.75)

Zygote.@adjoint function RecursiveArrayTools.ArrayPartition(xs...)
  a = RecursiveArrayTools.ArrayPartition(xs...)
  function ArrayPartition_pullback(Δ)
    szs = [0, cumsum(length.(a.x))...]
    return ntuple(length(a.x)) do i
      reshape(Δ[(szs[i] + 1):szs[i + 1]], size(a.x[i]))
    end
  end
  return a, ArrayPartition_pullback
end

Base.vec(nt::NamedTuple{(:x,)}) = vcat(vec.(nt.x)...)

Zygote.gradient(loss_function, ps)