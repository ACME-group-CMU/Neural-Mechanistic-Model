using ArrheniusModel
using Statistics
using OrdinaryDiffEq
using Plots
using Printf
G_values = [-5.10, -5.97, -5.85]
Ea_constants = [0.00 1.0 0.36; 1.0 0.00 0.38; 0.36 0.38 0.00]
gr()
pe = PhaseEnergies(G_values, Ea_constants)
display(pe.barriers)
T_range = 600.0:100.0:1200.0
flow_rates = 0.5:0.5:2.5
T_grid, f_grid = meshgrid(T_range, flow_rates)
threshold = 0.3 # Threshold for most preferable state
t = 3600 # seconds
dt = 0.5 # seconds
num_steps = round(Int, t/dt)
num_layers = floor(Int, t/0.5)+1
para_sim = num_steps, num_layers, dt
phase_names = ["x", "β", "κ"]
compositions_all = simulate_deposition.(f_grid, T_grid, Ref(pe.barriers), Ref(para_sim), 0.0022)
display(compositions_all)
max_compositions = most_preferable_state.(compositions_all, Ref(threshold), Ref(phase_names))
display(max_compositions)
display(compositions_all[3,2])

# Create a list of colors
#colorlist = ["red", "blue", "green", "yellow", "orange", "purple", "cyan", "magenta", "black", "gray"]
colorlist = ["gray", "blue", "pink", "red", "black"]

# Assign Colors
unique_values = unique(max_compositions)
color_dict = Dict(unique_values[i] => colorlist[i % length(colorlist) + 1] for i in 1:length(unique_values))
#increase the size of the data points 
scatter_size = 8

# Create a scatter canvas
p = scatter([], [], xlabel="Substrate Temperature (°C)", ylabel="Effective TEGa flow rate (sccm)", title="Artificial datapoints",
 label=false, size=(660, 600), ylims=(0.45,3.1), framestyle=:box, legendfontsize=10, ylabelfontsize=14, xlabelfontsize=14, titlefontsize=18)  # Create an empty scatter plot with no label
for (i, flow_rate) in enumerate(flow_rates)
    for (j, T) in enumerate(T_range)
        T -= 273.15  # Convert to Celsius
        color = color_dict[max_compositions[i, j]]  # Get the color based on the max_composition
        scatter!(p, [T], [flow_rate], color=color, label=false, markersize=scatter_size)
    end
end

# Add Legend
for (i, value) in enumerate(unique_values)
    scatter!(p, [], [], color=color_dict[value], label="Composition: $value")  # Add an empty scatter plot with the correct color and label
end

display(p)