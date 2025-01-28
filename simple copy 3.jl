using Lux, DiffEqFlux, OrdinaryDiffEq, Plots, Printf, Statistics
using ComponentArrays
using Optimization, OptimizationOptimisers
#using Optimisers
using Enzyme
using Dates
using Random
using StaticArrays
using SciMLSensitivity
using SciMLStructures

using Dates
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")


function evolve!(dc, c, p, t)
    p1 = p[1]
    p2 = p[2]
    dc .= c .* p2 * p1
end

function simulate(i1, i2, a, b, t_span)
    p2 = exp(-i2 * a)
    p1 = i1 * b
    p = (p1, p2)
    #println("p from function", p)
    #ismutablescimlstructure(p) = true
    #println("ismutablescimlstructure(p): ", ismutablescimlstructure(p))
    c0 = [1.0 0.0; 1.0 0.0]
    prob = ODEProblem(evolve!, c0, t_span, p)
    println(prob.p)
    #ismutablescimlstructure(prob.p) = true
    #println("ismutablescimlstructure(prob.p): ", ismutablescimlstructure(prob.p))
    sol = solve(prob, Tsit5())
    return Array(sol[end])
end


rng = Xoshiro(0)
b = [0.0 1.0; 1.0 0.0]
a = 0.6
n = length(b[1, :])
println("n:", n)
println("b:", b)
i1 = 0.18
i2 = 2.5 
timespan = (0.0, 5.0)
p2 = exp(-i2 * a)
p1 = i1 * b
p = (p1, p2)
isscimlstructure(p) = true
ismutablescimlstructure(p) = true
c0 = [1.0 0.0; 1.0 0.0]
prob = ODEProblem(evolve!, c0, timespan, p)
sol = solve(prob, Euler(), dt = 0.5)
ans = Array(sol[end])

"""
u0 = prob.u0
p = prob.p
tmp2 = Enzyme.make_zero(p)
t = prob.tspan[1]
du = zero(u0)

if DiffEqBase.isinplace(prob)
    _f = prob.f
    println("Inplace")
else
    _f = (du, u, p, t) -> (du .= prob.f(u, p, t); nothing)
    prinln("not inplace")
end

_tmp6 = Enzyme.make_zero(_f)
tmp3 = zero(u0)
tmp4 = zero(u0)
ytmp = u0
tmp1 = zero(u0)

Enzyme.autodiff(Enzyme.Reverse, Enzyme.Duplicated(_f, _tmp6),
    Enzyme.Const, Enzyme.Duplicated(tmp3, tmp4),
    Enzyme.Duplicated(ytmp, tmp1),
    Enzyme.Duplicated(p, tmp2),
    Enzyme.Const(t))

"""


display(ans)

inputs = [i1, i2]
input_size = length(inputs)
output_size = length(a) + length(b)
nn = Chain(
    Dense(input_size, input_size*3*n, tanh),
    Dense(input_size*3*n, output_size*2, tanh),
    Dense(output_size*2, output_size, sigmoid)
)

u, st = Lux.setup(rng, nn)

function convert_to_float64(x)
    if isa(x, AbstractArray)
        return Float64.(x)
    elseif isa(x, NamedTuple)
        return NamedTuple{keys(x)}(convert_to_float64.(values(x)))
    elseif isa(x, Dict)
        return Dict(k => convert_to_float64(v) for (k, v) in x)
    else
        return x
    end
end

# Convert weights to Float64
#u = convert_to_float64(u)


function predict_neuralode(u)
    # Get parameters from the neural network
    output, outst = nn(inputs, u, st)

    # Segregate the output
    p_a = output[1]
    pp_b = output[length(a)+1:end]
    p_b = zeros(n, n)
    index = 1
    for i in 1:n
        for j in 1:n
            p_b[i, j] = pp_b[index]
            index += 1
        end
    end

    nn_output = [p_a, p_b]
    #println("nn_output: ", nn_output)
    pred = simulate(i1, i2, p_a, p_b, timespan)
    return Array(pred)
end

function loss_neuralode(ans, u)
    pred = predict_neuralode(u)
    loss = sum(abs2, ans .- pred)
    return [loss], [pred]
end

function loss!(loss, ans, pinit)
    loss[1] = loss_neuralode(ans, pinit)[1][1]
    return nothing
end

#pinit = ComponentArray(u)

pred = predict_neuralode(u)
println("Training data: ", size(ans))
println("Prediction:", size(pred))

loss, pred = loss_neuralode(ans, u)
dloss = zero(loss)
dp = make_zero(u)
dloss[1] = 1.0

Enzyme.autodiff(Reverse, loss!, Duplicated(loss, dloss), Const(ans), Duplicated(u, dp))


"""
_loss = Enzyme.make_zero(loss!)
dp = Enzyme.make_zero(pinit)
dloss = zero(loss)


Enzyme.autodiff(Enzyme.Reverse, Enzyme.Duplicated(loss!, _loss),
    Enzyme.Const, Enzyme.Duplicated(loss, dloss),
    Enzyme.Const(ans),
    Enzyme.Duplicated(pinit, dp))



println("Loss: ", loss)
println("Training data: ", Array(ans))
println("Prediction: ", Array(pred))
"""

"""
loss_values = Float64[]
callback = function (p, l, pred; doplot = false)
    println(l)
    push!(loss_values, l)
    # plot current prediction against data
    if doplot
        plt = scatter(tsteps, ans[1,1], label = "Phase 1 Data", color = :blue)
        scatter!(plt, tsteps, ans[1,2], label = "Phase 2 Data", color = :red)
        scatter!(plt, tsteps, ans[2,1], label = "Phase 3 Data", color = :green)
        scatter!(plt, tsteps, ans[2,2], label = "Phase 4 Data", color = :yellow)
        scatter!(plt, tsteps, pred[1,1], label = "Phase 1 Prediction", color = :blue, shape = :cross)
        scatter!(plt, tsteps, pred[1,2], label = "Phase 2 Prediction", color = :red, shape = :cross)
        scatter!(plt, tsteps, pred[2,1], label = "Phase 3 Prediction", color = :green, shape = :cross)
        scatter!(plt, tsteps, pred[2,2], label = "Phase 4 Prediction", color = :yellow)
        display(plot(plt))
        savefig(plt, "training_$timestamp.svg")
    end
    return false
end

pinit = ComponentArray(u)
callback(pinit, loss_neuralode(ans, pinit)...)

adtype = Optimization.AutoEnzyme()

optf = Optimization.OptimizationFunction((u,_) -> loss_neuralode(ans, u), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.02); callback = callback, maxiters = 50)
"""

