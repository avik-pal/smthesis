using CairoMakie, MakiePublication

xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(ϕ([x, t], res.u)) for x in xs] for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(ϕ([x, t], res.u))) for x in xs] for t in ts]

fig = begin
    fig = Figure(; resolution=(1200, 400))

    ax1 = CairoMakie.Axis(fig[1, 1]; title="Predictions")
    ls = []
    labels = AbstractString[]

    for (i, u) in enumerate(u_predict)
        l = lines!(ax1, xs, u; linewidth=1)
        push!(ls, l)
        push!(labels, L"$u_{%$i}$")
    end

    axislegend(ax1, ls, labels; position=:rt)

    ax2 = CairoMakie.Axis(fig[1, 2]; title="Analytic")

    for u in u_real
        lines!(ax2, xs, u; linewidth=1)
    end

    ax3 = CairoMakie.Axis(fig[1, 3]; title="Absolute Error")

    for u in diff_u
        lines!(ax3, xs, u; linewidth=1)
    end

    fig
end

save(joinpath(@__DIR__, "../figures/lux/pinn_plot.pdf"), fig, pt_per_unit=0.75)

# p1 = plot(xs, u_predict; title="predict")
# p2 = plot(xs, u_real; title="analytic")
# p3 = plot(xs, diff_u; title="error")
# plot(p1, p2, p3)
