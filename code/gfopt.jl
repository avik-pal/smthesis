using ComponentArrays, FillArrays, Lux, Optimization, OptimizationCMAEvolutionStrategy, OptimizationNLopt, StableRNGs, Statistics, Zygote

r(θ) = exp(sin(θ)) - 2cos(4θ) + sin((2θ - oftype(θ, π)) / 12)^5

θs = collect(0:1.0e-2:2π)
rs = r.(θs)

model = Chain(Dense(1 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1))
ps, st = Lux.setup(StableRNG(1234), model)
ps = ComponentArray(ps)
ps_flat, ps_ax = Float64.(getdata(ps)), getaxes(ps)

function loss_function(ps, _)
    ps_ = ComponentArray(ps, ps_ax)
    r_pred, st_ = model(reshape(θs, 1, :), ps_, st)
    return mean(abs2, vec(r_pred) .- rs)
end

opt_func = OptimizationFunction{false}(loss_function, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps_flat)

sol_LN_NEWUOA = solve(opt_prob, OptimizationNLopt.NLopt.LN_NEWUOA(); maxiters=100_000)
sol_CMAES = solve(opt_prob, CMAEvolutionStrategyOpt(); maxiters=100_000)
