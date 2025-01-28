using Lux
using DifferentialEquations
using Optimisers
using Enzyme
using Random

# Step 1: Define the ODE system
function ode_system!(du, u, p, t)
    du[1] = -p[1] * u[1] + p[2] * u[2]
    du[2] = p[1] * u[1] - p[2] * u[2]
end

# Step 2: Solve the ODE with known parameters (synthetic data)
true_params = [0.5, 0.3]  # True parameters for synthetic data
u0 = [1.0, 0.0]          # Initial conditions
tspan = (0.0, 5.0)       # Time span for simulation

# Generate synthetic data
prob = ODEProblem(ode_system!, u0, tspan, true_params)
solution = solve(prob, Tsit5(), saveat=0.1)
observed_data = hcat(solution.u...) + 0.05 .* randn(size(hcat(solution.u...)))  # Add noise

# Step 3: Define the parameterized model (initial guesses)
model = Chain(
    Dense(2 => 32, gelu),
    Dense(32 => 32, gelu),
    Dense(32 => 2)
)

ps, state = Lux.setup(Random.default_rng(), model)  # Initialize parameters

# Step 4: Define the loss function
function solve_ode(p, u0, tspan)
    prob = ODEProblem(ode_system!, u0, tspan, p)
    solution = solve(prob, Tsit5(), saveat=0.1)
    return hcat(solution.u...)
end

function loss_fn(ps, u0, tspan, observed_data)
    p = model(ps)  # Model predicts parameters
    simulated_data = solve_ode(p, u0, tspan)
    return sum((simulated_data .- observed_data).^2)  # Mean squared error
end

# Step 5: Train the model
optimiser = Optimisers.Adam(0.01)  # Optimizer
epochs = 500  # Training epochs

for epoch in 1:epochs
    # Compute gradients using Enzyme
    grads = Enzyme.gradient(Reverse, Const(loss_fn), Const(ps), Const(u0), Const(tspan), Const(observed_data))
    
    # Update parameters
    ps = Optimisers.update!(ps, grads, optimiser)
    
    # Print loss every 50 epochs
    if epoch % 50 == 0
        current_loss = loss_fn(params, u0, tspan, observed_data)
        println("Epoch $epoch, Loss: $current_loss")
    end
end

# Step 6: Display the learned parameters
learned_params = model(params)
println("True Parameters: $true_params")
println("Learned Parameters: $learned_params")
