using ComponentArrays, Lux, StableRNGs, Statistics, TaylorDiff, Optimisers, Zygote

rng = StableRNG(0)

model = Chain(Dense(1 => 32, tanh), Dense(32 => 1))
ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps)

function g(model, t, ps, st)
  y, st_ = model(t, ps, st)
  return t .* y .+ 1.0f0, st_
end

function loss_function(model, ps, st)
  t = reshape(collect(-1.0f0:1.0f-2:1.0f0), 1, :)
  y = cos.(Float32(2π) .* t)

  t_ = map(__t -> TaylorDiff.make_taylor(__t, 1.0f0, Val(3)), t)
  ∂³g∂t³, st_ = g(model, t, ps, st)

  return mean(abs2, ∂³g∂t³ .- y), st_
end

opt = Adam(0.01f0)
st_opt = Optimisers.setup(opt, ps)

for i in 1:10000
  loss, pb = pullback(p -> first(loss_function(model, p, st)), ps)
  gs = only(pb(one(loss)))
  i % 100 == 1 && @info iteration=i loss=loss
  st_opt, ps = Optimisers.update(st_opt, ps, gs)
end

t = reshape(collect(-1.0f0:1.0f-2:1.0f0), 1, :)
y = vec(cos.(Float32(2π) .* t))
y_pred = vec(first(g(model, t, ps, st)))
