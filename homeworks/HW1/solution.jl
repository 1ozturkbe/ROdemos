# Activating Julia environment
using Pkg
Pkg.activate(".")

# Packages
using JuMP, Gurobi, Random, Distributions, MathOptInterface, Plots
MOI = MathOptInterface

# Generating data
n = 20;
b = 4;
a = rand(20);
c = rand(20);

rhos = collect(0:0.1:2)
costs = []
for ρ in rhos
    m = JuMP.Model(Gurobi.Optimizer)
    @variable(m, x[1:n], Bin)
    # Ellipsoidal uncertainty on the cost of the form...
    # @uncertain(m, u[1:n])
    # @constraint(m, norm(u, 2) <= ρ)
    if ρ != 0
        @constraint(m, [(b - sum(a .* x))/ρ, x...] in MOI.SecondOrderCone(n + 1))
    else
        @constraint(m, sum(a .* x) ≤ b)
    end
    @objective(m, Max, sum(c .* x))    
    optimize!(m)
    push!(costs, JuMP.objective_value(m))
end

plot(rhos, costs, title = "Robustness/optimality tradeoff", xlabel="Safety factor", ylabel="Optimal cost", legend=false)
