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


struct PhaseEnergies
    G::AbstractVector
    Ea::AbstractMatrix
    barriers::AbstractMatrix
    function PhaseEnergies(G::AbstractVector, forward_Ea::AbstractMatrix)
        n = length(G)
        @assert size(forward_Ea) == (n, n)
        deltaG = ΔG(G)
        barriers = [i == j ? 0 : (deltaG[i, j] > 0 ? deltaG[i, j] + forward_Ea[i, j] : forward_Ea[i, j])
                for j in 1:n, i in 1:n]
        new(G, forward_Ea, Matrix{eltype(G)}(barriers))
    end
end

n_phases(pe::PhaseEnergies) = length(pe.G)
ΔG(gmat::AbstractVector) = [gmat[j] - gmat[i] for j in eachindex(gmat), i in eachindex(gmat)]
ΔG(pe::PhaseEnergies) = ΔG(pe.G)


kb = 8.617e-5 #eV/K

function flow_coefficient(effecting_nums, decay_coefficient)
    effecting_nums = round(Int, effecting_nums)
    fcoeff = ones(effecting_nums)
    for j in 1:effecting_nums
        fcoeff[j] = decay_coefficient
    end
    return vcat(zeros(effecting_nums-1), fcoeff) #Add 0s for reverse
end

arrhenius_rate(pe::PhaseEnergies, T::Real=300) = arrhenius_rate(Array(pe.barriers), T)

# move calculation to helper fcn to make AD easier

function arrhenius_rate(barriers, T=300)
"""
Inputs:
    barriers: array of barriers
    T: temperature
Output:
    K: array of rate constants
Checkstat: Checked
"""
    kb = 8.617e-5 #eV/K
    A = 1.0 # Arrhenius prefactor
    #Assign dummy values to each K[i][i]
    n = size(barriers, 1)
    K = zeros(n,n)
    for i in 1:length(n)
        K[i, i] = 1.0
    end
    return K
end

function deposition_rates!(dc, c, p, t)
    dc .= c .* p.fcoeff[num_layers: 2*num_layers-1] * p.K
end

function simulate_deposition(flow_rate, T, barriers::Matrix, decay_constant = 0.00001)
    # Initialize existing_layers as a 2D array
    decay_coefficients = decay_constant * flow_rate
    fcoeff = flow_coefficient(num_layers, decay_coefficients)
    c0 = zeros(num_layers, n)
    c0[1, 1] = 1.0
    K = arrhenius_rate(barriers, T)
    """
    K = ones(n, n)
    K[1,2] = 0.5
    K[2,1] = 1.5
    """
    p = (fcoeff, K)
    p = NamedTuple{(:fcoeff, :K)}(p)
    p = ComponentArray(p)
    prob = ODEProblem(deposition_rates!, c0, timespan, p)
    sol = solve(prob, Euler(), save_everystep = false, dt=dt)
    #sol = solve(prob, Euler(), dt = 0.5)
    return Array(sol.u[end])
end




G_values = [-5.10, -5.97, -5.85]
Ea_constants = [0.00 1.0 0.36; 1.0 0.00 0.38; 0.36 0.38 0.00]
rng = Xoshiro(0)
pe = PhaseEnergies(G_values, Ea_constants)
n = n_phases(pe)
display(pe.barriers)
T = 300.0
flow_rate = 1.5
t = 20 # seconds
dt = 0.5 # seconds
num_layers = Int(t/dt)
timespan = (0.0, t)
compositions_all = simulate_deposition(flow_rate, T, pe.barriers, 0.0022)
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

u, st = Lux.setup(rng, nn)

function predict_neuralode(u)
    # Get parameters from the neural network
    inputs = [T, flow_rate]
    output, outst = nn(inputs, u, st)

    # Segregate the output
    pp_barrier = output[1:barrier_size]
    p_barrier = reshape(pp_barrier, (n, n))
    p_fcoeff = output[barrier_size+1:barrier_size+fcoeff_size]
    # Amorphous phase goes to zero
    nn_output = (p_barrier, p_fcoeff)
    predicted_composition = simulate_deposition(flow_rate, T, p_barrier, p_fcoeff[1])
    return Array(predicted_composition)
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, compositions_all .- pred)
    return loss, pred
end

callback = function (state::Optimization.OptimizationState, loss_value::Float64; doplot = false)
    p = state.u
    l, pred = loss_neuralode(p)
    println(l)
    if doplot
        println("No plots at this point")
    end
    return false
end

pinit = ComponentArray(u)
#callback(pinit, loss_neuralode(compositions_all, pinit)...)
l, pred = loss_neuralode(pinit)
println(l)
println(pred)

adtype = Optimization.AutoEnzyme(; mode=set_runtime_activity(Reverse))

optf = Optimization.OptimizationFunction((x,_) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.02); callback = callback, maxiters = 5)
