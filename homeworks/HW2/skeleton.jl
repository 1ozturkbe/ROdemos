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
...
optimize!(m)

# Fixed capacities plot
plt1 = heatmap(getvalue.(r)', xlabel = "Resources", ylabel = "Servers", title = "Fixed capacities")

# Time-mean of expansions plot
plt2 = heatmap(mean(getvalue.(e), dims=3)[:,:,1]', xlabel = "Resources", 
        ylabel = "Servers", title = "Time-mean of expansions")

# Time-variance of expansions
plt3 = heatmap(var(getvalue.(e), dims=3)[:,:,1]', xlabel = "Resources", 
ylabel = "Servers", title = "Time-variance of expansions")

# Time-mean of job transfers out
transfers_out = zeros(I,S,T); # Computing the transfers out of each server
[transfers_out[i,s,t] = sum(getvalue.(u)[i, s, :, t]) for i=1:I, s = 1:S, t = 1:T];
plt4 = heatmap(mean(transfers_out, dims=3)[:,:,1]', xlabel = "Resources", 
        ylabel = "Servers", title = "Time-mean of job transfers out")

# Time-variance of job transfers out
plt5 = heatmap(var(transfers_out, dims=3)[:,:,1]', xlabel = "Resources", 
ylabel = "Servers", title = "Time-variance of job transfers out")

# Plots of temperature
temps = getvalue.(h)
plt6 = plot(1:T, temps[1,:], label=1)
for s=2:S
    plot!(1:T, temps[s,:], label=s, title = "Server temperatures", xlabel = "Time period (t)", ylabel = "Temperature", 
        legend = :bottomright)
end

# Displaying
display(plt6)