using NeuralPDE, Lux, NNlib, ModelingToolkit, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum

@parameters x, t
@variables u(..)
∂t = Differential(t)
∂x = Differential(x)
∂x² = Differential(x)^2
∂x³ = Differential(x)^3
∂x⁴ = Differential(x)^4

α = 1
β = 4
γ = 1
eq = ∂t(u(x, t)) + u(x, t) * ∂x(u(x, t)) + α * ∂x²(u(x, t)) + β * ∂x³(u(x, t)) + γ * ∂x⁴(u(x, t)) ~ 0

uₑ(x, t; θ=-x / 2 + t) = 11 + 15 * tanh(θ) - 15 * tanh(θ)^2 - 15 * tanh(θ)^3
du(x, t; θ=-x / 2 + t) = 15 / 2 * (tanh(θ) + 1) * (3 * tanh(θ) - 1) * sech(z)^2

bcs = [
    u(x, 0) ~ uₑ(x, 0),
    u(-10, t) ~ uₑ(-10, t),
    u(10, t) ~ uₑ(10, t),
    ∂x(u(-10, t)) ~ du(-10, t),
    ∂x(u(10, t)) ~ du(10, t),
]

# Space and time domains
domains = [x ∈ Interval(-10.0, 10.0), t ∈ Interval(0.0, 1.0)]

# Discretization
dx = 0.4;
dt = 0.2;

# Neural network
chain = Chain(Dense(2 => 12, σ), Dense(12 => 12, σ), Dense(12 => 1))

discretization = PhysicsInformedNN(chain, GridTraining([dx, dt]))
@named pde_system = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])
prob = discretize(pde_system, discretization)

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

opt = OptimizationOptimJL.BFGS()
res = Optimization.solve(prob, opt; callback, maxiters=2000)
ϕ = discretization.phi