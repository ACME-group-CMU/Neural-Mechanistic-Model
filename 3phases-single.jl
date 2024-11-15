using Lux, DiffEqFlux, OrdinaryDiffEq, Plots, Printf, Statistics
using ComponentArrays
using Optimization, OptimizationOptimisers
using Enzyme
using ArrheniusModel
using Dates
using Random

G_values = [-5.10, -5.97, -5.85]
Ea_constants = [0.00 1.0 0.36; 1.0 0.00 0.38; 0.36 0.38 0.00]
gr()
rng = Xoshiro(0)
pe = PhaseEnergies(G_values, Ea_constants)
n = n_phases(pe)
display(pe.barriers)
T = 300.0
flow_rate = 1.5
threshold = 0.3 # Threshold for most preferable state
t = 600 # seconds
dt = 0.5 # seconds
num_steps = round(Int, t/dt)
num_layers = floor(Int, t/0.5)+1
para_sim = num_steps, num_layers, dt
phase_names = ["x", "β", "κ"]
compositions_all = simulate_deposition(flow_rate, T, pe.barriers, para_sim, 0.0022)
compositions_all = Array(compositions_all)
display(compositions_all)

inputs = [T, flow_rate]
input_size = length(inputs)  # Replace with the actual size of `inputs` if it's not a 1D vector
barrier_size = (n ^ 2)
fcoeff_size = 1 #sigmoid 0~1 #
precoeff_size = 0
output_size = barrier_size + fcoeff_size + precoeff_size
nn = Chain(
    Dense(input_size, input_size*3*n, tanh),
    Dense(input_size*3*n, output_size*2, tanh),
    Dense(output_size*2, output_size, sigmoid)
)

p, st = Lux.setup(rng, nn)

function predict_neuralode(p)
    # Get parameters from the neural network
    inputs = [T, flow_rate]
    output, outst = nn(inputs, p, st)

    # Segregate the output
    pp_barrier = output[1:barrier_size]
    p_barrier = reshape(pp_barrier, (n, n))
    p_fcoeff = output[barrier_size+1:barrier_size+fcoeff_size]
    # Amorphous phase goes to zero
    nn_output = (p_barrier, p_fcoeff)
    predicted_composition = simulate_deposition(flow_rate, T, p_barrier, para_sim, p_fcoeff[1])
    return Array(predicted_composition)
end

function loss_neuralode(ans, p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ans .- pred)
    return loss, pred
end

function loss_neuralode2(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, compositions_all .- pred)
    return loss
end

pred = predict_neuralode(p)
println("Training data: ", size(compositions_all))
println("Prediction:", size(pred))

loss, pred = loss_neuralode(compositions_all, p)

println("Loss: ", loss)
println("Training data: ", Array(compositions_all))
println("Prediction: ", Array(pred))

ode_data_avg = mean(compositions_all, dims=1)


loss_values = Float64[]
callback = function (p, l, pred; doplot = false)
    println(l)
    push!(loss_values, l)
    # plot current prediction against data
    if doplot
        pred_avg = mean(pred, dims=1)
        #pred_avg = reshape(pred_avg, (3, 21))
        #plot the three phases from ode_data_avg and pred_avg
        plt = scatter(tsteps, ode_data_avg[1], label = "Phase 1 Data", color = :blue)
        scatter!(plt, tsteps, ode_data_avg[2], label = "Phase 2 Data", color = :red)
        scatter!(plt, tsteps, ode_data_avg[3], label = "Phase 3 Data", color = :green)
        scatter!(plt, tsteps, pred_avg[1], label = "Phase 1 Prediction", color = :blue, shape = :cross)
        scatter!(plt, tsteps, pred_avg[2], label = "Phase 2 Prediction", color = :red, shape = :cross)
        scatter!(plt, tsteps, pred_avg[3], label = "Phase 3 Prediction", color = :green, shape = :cross)
        display(plot(plt))
        savefig(plt, "training_$timestamp.svg")
    end
    return false
end

pinit = ComponentArray(p)
callback(pinit, loss_neuralode(compositions_all, pinit)...)

adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((p,_) -> loss_neuralode(compositions_all, p), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.02); callback = callback, maxiters = 50)