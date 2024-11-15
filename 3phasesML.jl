using Lux, DiffEqFlux, OrdinaryDiffEq, Plots, Printf, Statistics
using Enzyme
using Dates

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