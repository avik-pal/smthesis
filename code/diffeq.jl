using ComponentArrays, Lux, Optimization, OptimizationOptimisers, OrdinaryDiffEq, RecursiveArrayTools, SciMLSensitivity, StableRNGs

u0, du0, tspan = [0.0f0; 2.0f0], [0.0f0; 0.0f0], (0.0f0, 1.0f0)
ts = range(tspan[1], tspan[2], length=21)

model = Chain(Dense(2 => 50, tanh), Dense(50 => 2))
ps, st = Lux.setup(StableRNG(1234), model)
ps = ComponentArray(ps)

ff(du, u, p, t) = first(model(u, p, st))
prob = SecondOrderODEProblem{false}(ff, u0, du0, tspan, ps)

predict(θ) = Array(solve(prob, Tsit5(); p=θ, saveat=ts))

y_true = vcat(collect(0:0.05f0:1)', collect(2:-0.05f0:1)')

loss_function(θ) = sum(abs2, predict(θ)[1:2, :] .- y_true)

optprob = OptimizationProblem(OptimizationFunction{false}((x, p) -> loss_function(x), Optimization.AutoZygote()), ps)

res = Optimization.solve(optprob, Adam(0.01f0); maxiters=1000)
