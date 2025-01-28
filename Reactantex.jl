using Lux, Reactant, Enzyme, Random, Zygote
using Functors, Optimisers, Printf

model = Chain(
    Dense(2 => 32, gelu),
    Dense(32 => 32, gelu),
    Dense(32 => 2)
)
ps, st = Lux.setup(Random.default_rng(), model)

x = randn(Float32, 2, 32)
y = x .^ 2

const xdev = reactant_device()

x_ra = x |> xdev
y_ra = y |> xdev
ps_ra = ps |> xdev
st_ra = st |> xdev
nothing

pred_lux, _ = model(x, ps, Lux.testmode(st))
model_compiled = @compile model(x_ra, ps_ra, Lux.testmode(st_ra))
pred_compiled, _ = model_compiled(x_ra, ps_ra, Lux.testmode(st_ra))

pred_lux .- Array(pred_compiled)

function loss_function(model, ps, st, x, y)
    pred, _ = model(x, ps, st)
    return MSELoss()(pred, y)
end

loss_function(model, ps, st, x, y)

#∂ps_zyg = only(Zygote.gradient(ps -> loss_function(model, ps, st, x, y), ps))

function enzyme_gradient(model, ps, st, x, y)
    return Enzyme.gradient(Enzyme.Reverse, Const(loss_function), Const(model),
        ps, Const(st), Const(x), Const(y))[2]
end

enzyme_gradient_compiled = @compile enzyme_gradient(model, ps_ra, st_ra, x_ra, y_ra)

∂ps_enzyme = enzyme_gradient_compiled(model, ps_ra, st_ra, x_ra, y_ra)
