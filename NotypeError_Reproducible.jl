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
    dc .= c * p.p2 * p.p1
end

function simulate(i1, i2, a, b, t_span)
    p2 = exp(-i2 * a)
    p1 = i1 * b
    p = (p1, p2)
    p_named = NamedTuple{(:p1, :p2)}(p)
    p = ComponentArray(p_named)
    c0 = [1.0 0.0; 1.0 0.0]
    prob = ODEProblem(evolve!, c0, t_span, p)
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
p_named = NamedTuple{(:p1, :p2)}(p)
p = ComponentArray(p_named)
c0 = [1.0 0.0; 1.0 0.0]
prob = ODEProblem(evolve!, c0, timespan, p)
sol = solve(prob, Euler(), dt = 0.5)
ans = Array(sol[end])


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
u = convert_to_float64(u)


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
    pred = simulate(i1, i2, p_a, p_b, timespan)
    return Array(pred)
end

function loss_neuralode(ans, u)
    pred = predict_neuralode(u)
    loss = sum(abs2, ans .- pred)
    return [loss], [pred]
end

function loss!(loss, ans, pinit)
    loss .= loss_neuralode(ans, pinit)[1]
    return nothing
end

loss, pred = loss_neuralode(ans, u)
dloss = zero(loss)
dp = make_zero(u)
dloss[1] = 1.0

#Enzyme.autodiff(Reverse, loss!, Duplicated(loss, dloss), Const(ans), Duplicated(u, dp))
Enzyme.autodiff(set_runtime_activity(Reverse), Const(loss!), Duplicated(loss, dloss), Const(ans), Duplicated(u, dp))
