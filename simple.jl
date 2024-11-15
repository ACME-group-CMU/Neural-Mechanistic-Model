using Lux, DiffEqFlux, OrdinaryDiffEq, Plots, Printf, Statistics
using ComponentArrays
using Optimization, OptimizationOptimisers
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
    p2 = exp(-i2 * a)
    p1 = i1 * b
    p = [p1, p2]
    c0 = [1.0 0.0; 1.0 0.0]
    prob = ODEProblem(evolve!, c0, t_span, p)
    sol = solve(prob, Euler(), save_everystep=false, dt = 0.5)
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
ans = simulate(i1, i2, a, b, timespan)

display(ans)

inputs = [i1, i2]
input_size = length(inputs)  # Replace with the actual size of `inputs` if it's not a 1D vector
output_size = length(a) + length(b)
nn = Chain(
    Dense(input_size, input_size*3*n, tanh),
    Dense(input_size*3*n, output_size*2, tanh),
    Dense(output_size*2, output_size, sigmoid)
)

u, st = Lux.setup(rng, nn)

function predict_neuralode(u)
    # Get parameters from the neural network
    inputs = [T, flow_rate]
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
    # Amorphous phase goes to zero
    nn_output = (p_a, p_b)
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

loss, pred = loss_neuralode(ans, p)

println("Loss: ", loss)
println("Training data: ", Array(ans))
println("Prediction: ", Array(pred))

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