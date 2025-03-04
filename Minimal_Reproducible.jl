using Lux, DiffEqFlux, OrdinaryDiffEq, Plots, Printf, Statistics
using ComponentArrays
using Optimization, OptimizationOptimisers
using Enzyme
using Random
using StaticArrays
using SciMLSensitivity
using SciMLStructures

Enzyme.API.looseTypeAnalysis!(true)

# Define the evolve! function
function evolve!(dc, c, p, t)
    dc .= c * p.p2 * p.p1
end

# Define the simulate function
function simulate(i1, i2, a, b, t_span)
    p2 = exp(-i2 * a)
    p1 = i1 * b
    p = (p1, p2)
    p_named = NamedTuple{(:p1, :p2)}(p)
    p = ComponentArray(p_named)
    c0 = [1.0 2.0; 1.0 0.0]
    prob = ODEProblem(evolve!, c0, t_span, p)
    sol = solve(prob, Tsit5())
    return Array(sol[end])
end

# Define the neural network and optimization process
rng = Xoshiro(0)
b = [0.0 1.0; 1.0 0.0]
a = 0.6
n = length(b[1, :])
i1 = 0.18
i2 = 2.5 
timespan = (0.0, 5.0)
p2 = exp(-i2 * a)
p1 = i1 * b
p = (p1, p2)
p_named = NamedTuple{(:p1, :p2)}(p)
p = ComponentArray(p_named)
c0 = [1.0 2.0; 1.0 0.0]
prob = ODEProblem(evolve!, c0, timespan, p)
sol = solve(prob, Euler(), dt = 0.5)
ans = Array(sol[end])

inputs = [i1, i2]
input_size = length(inputs)
output_size = length(a) + length(b)
nn = Chain(
    Dense(input_size, input_size*3*n, tanh),
    Dense(input_size*3*n, output_size*2, tanh),
    Dense(output_size*2, output_size, sigmoid)
)

u, st = Lux.setup(rng, nn)

function predict_neuralode(u)
    output, outst = nn(inputs, u, st)
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
    pred = simulate(i1, i2, p_a, p_b, timespan)
    return Array(pred)
end

function loss_neuralode(u)
    pred = predict_neuralode(u)
    loss = sum(abs2, ans .- pred)
    return loss, pred
end

callback = function (state::Optimization.OptimizationState, loss_value::Float64; doplot = false)
    p = state.u
    l, pred = loss_neuralode(p)
    println(l)
    return false
end

pinit = ComponentArray(u)
adtype = Optimization.AutoEnzyme(; mode=set_runtime_activity(Reverse))
optf = Optimization.OptimizationFunction((x,_) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.02); callback = callback, maxiters = 5)