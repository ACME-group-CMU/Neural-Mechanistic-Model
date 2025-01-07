using Lux, DiffEqFlux, OrdinaryDiffEq, Plots, Printf, Statistics
using ComponentArrays
using Optimization, OptimizationOptimisers
using Enzyme
using Dates
using Random
using StaticArrays

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


rng = Xoshiro(0)
b = [0.0 1.0; 1.0 0.0]
a = [0.6]
n = length(b[1, :])
i1 = [0.18]
i2 = [2.5]
timespan = (0.0, 5.0)
ans = simulate(i1, i2, a, b, timespan)

display(ans)

inputs = [0.18, 2.5]
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
    p_a = [output[1]]
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

loss, pred = loss_neuralode(ans, u)

loss_values = Float64[]
callback = function (p, l, pred; doplot = false)
    println(l)
    push!(loss_values, l)
end

pinit = ComponentArray(u)
callback(pinit, loss_neuralode(ans, pinit)...)

adtype = Optimization.AutoEnzyme()

optf = Optimization.OptimizationFunction((u,_) -> loss_neuralode(ans, u), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.02); callback = callback, maxiters = 50)