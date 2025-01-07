using Lux, DiffEqFlux, OrdinaryDiffEq, Plots, Printf, Statistics
using ComponentArrays
using Optimization, OptimizationOptimisers
using SciMLSensitivity
#using Optimisers
using Enzyme
using Dates
using Random
using StaticArrays


using Dates
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

function evolve!(dc, c, p, t)
    p1 = p[1]
    p2 = p[2]
    dc .= c .* p2 * p1
end

function simulate(i1, i2, a, b, t_span)
    p2 = exp(-i2[1] * a[1])
    p1 = i1[1] * b
    p = (p1, p2)
    c0 = [1.0 0.0; 1.0 0.0]
    prob = ODEProblem(evolve!, c0, t_span, p)
    sol = solve(prob, Euler(), save_everystep=false, dt = 0.5)
    return Array(sol[end])
end

function simulate!(sol, i1, i2, a, b, t_span)
    sol .= simulate(i1, i2, a, b, t_span)
    return nothing
end

rng = Xoshiro(0)
b = [0.0 1.0; 1.0 0.0]
a = [0.6]
n = length(b[1, :])
println("n:", n)
println("b:", b)
i1 = [0.18]
i2 = [2.5]
timespan = [0.0, 5.0]
sol = simulate(i1, i2, a, b, timespan)
di1 = zero(i1)
di2 = zero(i2)
da = zero(a)    
db = zero(b)
dsol = zero(sol)
dsol[1,1] = 1.0 
Enzyme.autodiff(Reverse, simulate!, Duplicated(sol, dsol), Duplicated(i1,di1), Duplicated(i2,di2), Duplicated(a,da), Duplicated(b,db), Const(timespan))
display(ans)
"""
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
    println("nn_output: ", nn_output)
    pred = simulate(i1, i2, p_a, p_b, timespan)
    return Array(pred)
end

function loss_neuralode(ans, u)
    pred = predict_neuralode(u)
    loss = sum(abs2, ans .- pred)
    return loss, pred
end

pred = predict_neuralode(u)
println("Training data: ", size(ans))
println("Prediction:", size(pred))

"""