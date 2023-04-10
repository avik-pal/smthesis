using ComponentArrays, Lux, NNlib, StableRNGs, Statistics, TaylorDiff,
    Optimisers, Zygote

rng = StableRNG(1234)

model = Chain(Dense(1 => 32, relu; allow_fast_activation=false), Dense(32 => 1))
ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps)

function taylor_derivative(f, x, ::Val{N}) where {N}
    x_ = map(__x -> TaylorDiff.make_taylor(__x, 1.0f0, Val(N + 1)), x)
    out = f(x_)
    if out isa Tuple
        y, others... = out
        return (TaylorDiff.extract_derivative(y, N + 1), others...)
    else
        return TaylorDiff.extract_derivative(out, N + 1)
    end
end

function gradient(model, x, ps, st)
    ps_ = map(__x -> TaylorDiff.make_taylor(__x, 1.0f0, Val(2)), ps)
    y, st_ = model(x, ps_, st)
    return TaylorDiff.extract_derivative(sum(y), 2)
end

gradient(model, randn(Float32, 1, 1), ps, st)

function g(model, t, ps, st)
    y, st_ = model(t, ps, st)
    return t .* y .+ 1.0f0, st_
end

function loss_function(model, ps, st)
    t = reshape(collect(-1.0f0:1.0f-1:1.0f0), 1, :)
    y = cos.(Float32(2π) .* t)

    ∂³g∂t³, st_ = taylor_derivative(__t -> g(model, __t, ps, st), t, Val(3))
    return mean(abs2, ∂³g∂t³ .- y), st_
end

loss_function(model, ps, st)



Zygote.gradient(p -> first(loss_function(model, p, st)), ps)

opt = Adam(0.01f0)
st_opt = Optimisers.setup(opt, ps)

for i in 1:10000
    loss, pb = pullback(p -> first(loss_function(model, p, st)), ps)
    gs = only(pb(one(loss)))
    i % 100 == 1 && @info iteration = i loss = loss
    st_opt, ps = Optimisers.update(st_opt, ps, gs)
end

t = collect(-1.0f0:1.0f-2:1.0f0)
y_true = vec(1 .- sin.(Float32(2π) .* t) ./ Float32(8π))
y_pred = vec(first(g(model, reshape(t, 1, :), ps, st)))
