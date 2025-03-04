using Lux, DiffEqFlux, OrdinaryDiffEq, Plots, Printf, Statistics
using ComponentArrays
using Optimization, OptimizationOptimisers,OptimizationOptimJL
#using Optimisers
using Enzyme
using Dates
using Random
using StaticArrays
using SciMLSensitivity
using SciMLStructures

Enzyme.API.looseTypeAnalysis!(true)

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
    c0 = [1.0 2.0; 1.0 0.0]
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
c0 = [1.0 2.0; 1.0 0.0]
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

function loss_neuralode(u)
    pred = predict_neuralode(u)
    loss = sum(abs2, ans .- pred)
    return loss, pred
end

"""
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
#Enzyme.autodiff(set_runtime_activity(Reverse), Const(loss_neuralode), Duplicated(loss, dloss), Const(ans), Duplicated(u, dp))
"""
loss_values = []
predictions = []

callback = function (state::Optimization.OptimizationState, loss_value::Float64; doplot = false)
    p = state.u
    l, pred = loss_neuralode(p)
    println(l)
    #push!(loss_values[1], l)
    # plot current prediction against data
    push!(loss_values, l)
    push!(predictions, pred)
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
#callback(pinit, loss_neuralode(pinit)[1])

adtype = Optimization.AutoEnzyme(; mode=set_runtime_activity(Reverse))

optf = Optimization.OptimizationFunction((x,_) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.02); callback = callback, maxiters = 50)

# Plot the loss values over iterations
plot(1:length(loss_values), loss_values, xlabel = "Iteration", ylabel = "Loss", title = "Loss over Iterations")
#savefig("loss_over_iterations_$timestamp.svg")

#organize the value of every [1,1] in predictions[i] into a vector

predictions11 = [predictions[i][1,1] for i in 1:length(predictions)]
predictions12 = [predictions[i][1,2] for i in 1:length(predictions)]
predictions21 = [predictions[i][2,1] for i in 1:length(predictions)]
predictions22 = [predictions[i][2,2] for i in 1:length(predictions)]


# Plot the predictions over iterations

plt = plot(1:length(loss_values), predictions11, label = "Phase 1 Prediction", color = :blue, shape = :cross)
plot!(plt, 1:length(loss_values), predictions12, label = "Phase 2 Prediction", color = :red, shape = :cross)
plot!(plt, 1:length(loss_values), predictions21, label = "Phase 3 Prediction", color = :green, shape = :cross)
plot!(plt, 1:length(loss_values), predictions22, label = "Phase 4 Prediction", color = :yellow)
hline!(plt, [ans[1,1]], label = "Phase 1 Data", color = :blue, linestyle = :dash)
hline!(plt, [ans[1,2]], label = "Phase 2 Data", color = :red, linestyle = :dash)
hline!(plt, [ans[2,1]], label = "Phase 3 Data", color = :green, linestyle = :dash)
hline!(plt, [ans[2,2]], label = "Phase 4 Data", color = :yellow, linestyle = :dash)
display(plot(plt))
