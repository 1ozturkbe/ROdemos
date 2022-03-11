# Activating Julia environment
using Pkg
Pkg.activate(".")

# Packages
using JuMP, Gurobi, Random, Distributions, LinearAlgebra, Plots

T = 25
S = 5
I = 6
Random.seed!(MersenneTwister(314)) # Do not change the seed. 
function generate_data(I::Int64, S::Int64, T::Int64)
    mvdist = Multinomial(I, S) # Multinomial w.r.t. server allocation
    raws = Array{Any}(zeros(S)) # Generating tridiagonal multivariate normals for the servers
    for s = 1:S 
        randos = 1/5*randperm(S)
        mat = diagm(0 => 1*ones(S), 1 => randos, -1 => randos)
        symmat = mat * mat'
        raws[s] = MvNormal(symmat)
    end
    D = zeros(I, S, T); # Generating demand data
    for t = 1:T
        D[:,:, t] += 2*rand(mvdist,I)'
        for i = 1:I
            for s = 1:S
                D[i, s, t] += abs(rand(raws[s])[1])
            end
        end
    end
    return D
end
D = generate_data(I,S,T);

heatmap(1:I, 1:S, var(D, dims=3)[:,:,1], xlabel = "Resources", ylabel = "Servers", title = "Time-variance of demand")

heatmap(1:I, 1:S, mean(D, dims=3)[:,:,1], xlabel = "Resources", ylabel = "Servers", title = "Time-mean of demand")

# Plotting different kinds of resource costs, i ∈ [1, ... , I]
V = 2*[1.5, 1.3, 0.8, 1.3, 1.2, 0.5]; # expansion costs
F = [1.5, 1.3, 0.8, 1.3, 1.2, 0.9]; # fixed costs
C = [0.4, 0.5, 0.6, 0.7, 0.8, 0.7]; # reallocation costs 
# bardata = hcat(V, F, C)
# sx = repeat(["expansion", "fixed", "reallocation"], inner = 6)
# nam = repeat("Resource " .* string.(1:I), outer = 3)
# groupedbar(nam, bardata, group = sx, ylabel = "Costs", 
#         title = "Costs for each resource")

# OPTIMIZATION MODEL HERE.
m = JuMP.Model(Gurobi.Optimizer)
@variable(m, r[i=1:I, s=1:S] >= 0)
@variable(m, e[i=1:I, s=1:S, t=1:T] >= 0)
@variable(m, h[s=1:S, t =1:T] >= 0)
@variable(m, u[i=1:I, s1=1:S, s2=1:S, t=1:T] >= 0)

# Demand uncertainty is described by: 
# @uncertain(m, d[i=1:I, s=1:S, t =1:T])
# for s=1:S
#     @constraint(m, norm(d[:, s, :], 1) <= Γ)
# end
# Which is a budget uncertainty with ρ = 1 for the ∞ norm, and γ bounding the 1-norm.
# ρ = 0
# Γ = 0
ρ = 1
Γ = sqrt(2log(1/0.05))*sqrt(I*T)

@objective(m, Min, T*sum(sum(F .* r[:, s]) for s=1:S) + 
                sum(sum(sum(V .* e[:, s, t]) for s=1:S) for t=1:T) + 
                sum(sum(sum(sum(C .* u[:,s1,s2,t]) for s1=1:S) for s2=1:S) for t=1:T));

# Using the robust counterpart here
for i = 1:I
    for s = 1:S
        for t = 1:T 
            y = @variable(m, [j=1:I, k=1:T])
            abs_y = @variable(m, [j=1:I, k=1:T])
            max_diff = @variable(m)
            @constraint(m, abs_y .≥ y)
            @constraint(m, abs_y .≥ -y)
            for j = 1:I # The right hand side uncertainty can be tricky!
                for k = 1:T
                    if j == i && k == t
                        @constraint(m, max_diff ≥ (-1 + y[j,k]))
                        @constraint(m, max_diff ≥ -(-1 + y[j,k]))
                    else
                        @constraint(m, max_diff ≥ y[j,k])
                        @constraint(m, max_diff ≥ -y[j,k])
                    end
                end
            end
            @constraint(m, D[i,s,t] + ρ*sum(abs_y) + Γ*max_diff + sum(u[i, s2, s, t] for s2 = 1:S) <= r[i,s] + e[i,s,t] + sum(u[i, s, s2, t] for s2 = 1:S))
        end
    end
end

for s = 1:S
    @constraint(m, h[s, 1] == sum(e[:,s,1] ./3))
    @constraint(m, h[s, 2] == sum(e[:,s,2] ./3) + sum(e[:,s,1] ./3))
    @constraint(m, h[s, 1] <= 1)
    @constraint(m, h[s, 2] <= 1)
    for t = 3:T
        @constraint(m, h[s ,t] == sum(e[:,s,t-2] ./3) + sum(e[:,s,t-1] ./3) + sum(e[:,s,t] ./3))
        @constraint(m, h[s, t] <= 1)
    end
end
optimize!(m)

# Fixed capacities plot
plt1 = heatmap(value.(r)', xlabel = "Resources", ylabel = "Servers", title = "Fixed capacities")

# Time-mean of expansions plot
plt2 = heatmap(mean(value.(e), dims=3)[:,:,1]', xlabel = "Resources", 
        ylabel = "Servers", title = "Time-mean of expansions")

# Time-variance of expansions
plt3 = heatmap(var(value.(e), dims=3)[:,:,1]', xlabel = "Resources", 
ylabel = "Servers", title = "Time-variance of expansions")

# Time-mean of job transfers out
transfers_out = zeros(I,S,T); # Computing the transfers out of each server
[transfers_out[i,s,t] = sum(value.(u)[i, s, :, t]) for i=1:I, s = 1:S, t = 1:T];
plt4 = heatmap(mean(transfers_out, dims=3)[:,:,1]', xlabel = "Resources", 
        ylabel = "Servers", title = "Time-mean of job transfers out")

# Time-variance of job transfers out
plt5 = heatmap(var(transfers_out, dims=3)[:,:,1]', xlabel = "Resources", 
ylabel = "Servers", title = "Time-variance of job transfers out")

# Plots of temperature
temps = value.(h)
plt6 = plot(1:T, temps[1,:], label=1)
for s=2:S
    plot!(1:T, temps[s,:], label=s, title = "Server temperatures", xlabel = "Time period (t)", ylabel = "Temperature", 
        legend = :bottomright)
end

# Displaying
display(plt6)