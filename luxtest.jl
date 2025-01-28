using Lux, Printf, Statistics
using ComponentArrays
using Optimization, OptimizationOptimisers
#using Optimisers
using Enzyme
using Dates
using Random
using StaticArrays

function generate_data(rng::AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, Float32, (1, 128)) .* 0.1f0
    return (x, y)
end
rng = MersenneTwister()
Random.seed!(rng, 12345)

(x, y) = generate_data(rng)

# Define a simple neural network
nn = Chain(Dense(1 => 16, relu), Dense(16 => 1))

# Initialize the parameters
ps, st = Lux.setup(rng, nn)

# Define the prediction function
function predict(p, x)
    y_pred, _ = nn(x, p, st)
    return y_pred
end

# Define the loss function
function loss_neuralode(p)
    y_pred = predict(p, x)
    return [sum(abs2, y .- y_pred)]
end

function loss!(loss, p)
    loss[1] = loss_neuralode(p)[1]
    return nothing
end

ypred = predict(ps, x)


loss = loss_neuralode(ps)
dloss = zero(loss)
dloss[1] = 1.0
dp = make_zero(ps)

Enzyme.autodiff(Reverse, loss!, Duplicated(loss, dloss), Duplicated(ps, dp))

"""
# Define the optimization problem
optf = Optimization.OptimizationFunction((p, _) -> loss(p), Optimization.AutoEnzyme())
optprob = Optimization.OptimizationProblem(optf, ps)
# Solve the optimization problem
result = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); maxiters = 100)
"""
# Print the result
#println("Optimized parameters: ", result.minimizer)
#println("Final loss: ", result.minimum)