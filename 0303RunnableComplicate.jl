using Lux, DiffEqFlux, OrdinaryDiffEq, Plots, Printf, Statistics
using ComponentArrays
using Optimization, OptimizationOptimisers,OptimizationOptimJL
#using Optimisers
using Enzyme
using Random
using StaticArrays
using SciMLSensitivity
using SciMLStructures

Enzyme.API.looseTypeAnalysis!(true)
rng = Xoshiro(0)


function estimatefcoefficient(type, effecting_nums, decay_coefficient)
    effecting_nums = round(Int, effecting_nums)
    fcoeff = ones(effecting_nums)
    for j in 1:effecting_nums
        if type == "exponential"
            fcoeff[j] = exp(-decay_coefficient * j)
        elseif type == "linear"
            fcoeff[j] = (1 - decay_coefficient * j)
        elseif type == "dummy"
            fcoeff[j] = decay_coefficient
        end
    end
    return vcat(zeros(effecting_nums-1), fcoeff) #Add 0s for reverse
end

function estimateK(B, T=300)
    k = 8.
    A = 1.0 
    K = A * exp.(-B ./ (k * T))
    # Adjust the diagonal elements
    for i in axes(K,1)
        K[i, i] =  -1 * sum(K[i, [1:i-1; i+1:end]])
    end
    return K
end

function evolve!(dc, c, p, t)
    j = floor(Int, t / 0.5) + 1
    #j0 = p.j0
    f = p.fcoeff[j: num_layers+j-1]
    #println(t)
    #println(f)
    dc .= c .* f * p.K
    """
    if j != j0
        c[j, 1] = 1.0
        j0 = j
    end
    """
end

function simulate(F, T, B::Matrix, timespan, decay_constant = 0.00001)
    decay_coefficients = decay_constant * F
    fcoeff = estimatefcoefficient("exponential", num_layers, decay_coefficients)
    c0 = zeros(num_layers, n)
    for i in 1:num_layers
        c0[i, 1] = 1.0
    end
    K = estimateK(B, T)
    j = 0
    j0 = 0
    p = (fcoeff, K, num_steps, num_layers, j, j0)
    p = NamedTuple{(:fcoeff, :K, :num_steps, :num_layers, :j, :j0)}(p)
    p = ComponentArray(p)
    prob = ODEProblem(evolve!, c0, timespan, p)
    sol = solve(prob, Euler(), dt=0.1)
    return Array(sol.u[end])
end


B = [0.0 1.0 0.36; 1.87 0.0 0.5000000000000001; 1.1099999999999999 0.38 0.0]
n = size(B,1)
T = 3000.0
F = 1.5
t = 20.1 # seconds
dt = 0.5 # seconds
num_steps = round(Int, t/dt)
num_layers = floor(Int, t/0.5)+1
timespan = (0.0, t)
ans = simulate(F, T, B, timespan, 0.0022)

display(ans)

inputs = [T, F]
input_size = length(inputs)  # Replace with the actual size of `inputs` if it's not a 1D vector
B_size = (n ^ 2)
fcoeff_size = 1 #sigmoid 0~1 #
precoeff_size = 0
output_size = B_size + fcoeff_size + precoeff_size
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
    pp_B = output[1:B_size]
    p_B = reshape(pp_B, (n, n))
    p_fcoeff = output[B_size+1:B_size+fcoeff_size]

    pred = simulate(F, T, p_B, timespan, p_fcoeff[1])
    return pred
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ans .- pred)
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
