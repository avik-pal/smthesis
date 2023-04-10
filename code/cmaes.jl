using Optimization, OptimizationCMAEvolutionStrategy
using ComponentArrays, Lux, NNlib, StableRNGs, Statistics, Zygote

r(θ) = exp(sin(θ)) - 2cos(4θ) + sin((2θ - oftype(θ, π))/12)^5

θs = collect(0:1.0f-2:Float32(2π))
rs = r.(θs)

model = Chain(Dense(1 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1))
ps, st = Lux.setup(StableRNG(1234), model)
ps = ComponentArray(ps)

function loss_function(ps, _)
    r_pred, st_ = model(reshape(θs, 1, :), ps, st)
    return mean(abs2, vec(r_pred) .- rs)
end

opt_func = OptimizationFunction{false}(loss_function)
opt_prob = OptimizationProblem(opt_func, ps)

sol = solve(opt_prob, CMAEvolutionStrategyOpt(); maxiters=100_000)
