using JuMP, JuMPeR, Gurobi, Random, Distributions, LinearAlgebra, DataFrames, Plots

n = 10 # Number of facilities
m = 50 # Number of customers

# Generating random data (please don't change the seeds.)
facilities = 0.6.*rand(MersenneTwister(5), n,2) .+ 0.2;
customers = rand(MersenneTwister(2), m, 2); 
c = [LinearAlgebra.norm(customers[i, :] .- facilities[j, :])[1] for j=1:n, i=1:m];
f = rand(MersenneTwister(3), n)*1 .+ 5;
s = rand(MersenneTwister(4), n)*2 .+ 15;
d = rand(MersenneTwister(5), m)*0.5 .+ 0.75

# P matrix
R_D = 0.25
P = [0.2*exp(-1/R_D .*LinearAlgebra.norm(customers[i, :] .- customers[j, :])[1]) for j=1:m, i=1:m];
P = (P .>= 0.2*exp(-1/R_D .* R_D)) .* P 

""" Cutting plane facility location model. 
    Note this is just the nominal problem, but with linear policies y(z) = u + Vz when z = 0. """
function CP_facility_model(c::Matrix, f::Vector)
    n, m = size(c) 
    @assert length(f) == n
    model = Model(solver = GurobiSolver(OutputFlag = 0))

    # VARIABLES
    @variable(model, x[1:n], Bin)     # Facility locations
    @variable(model, V[1:n, 1:m, 1:m])  # Linear decision rule w.r.t. z
    @variable(model, u[1:n, 1:m])  # Fixed policy  

    # CONSTRAINTS    
    for j=1:m # Demand constraints
        dt = zeros(n, m);
        [dt[i,j] = 1 for i=1:n];
        @constraint(model, sum(dt .* u) >= d[j]) 
    end
    for i=1:n # Capacity constraints
        dt = zeros(n, m);
        [dt[i,j] = 1 for j=1:m];
        @constraint(model, sum(dt .* u) <= s[i] * x[i])
    end
    # OBJECTIVE
    @variable(model, F)
    @constraint(model, F >= sum(f[j] * x[j] for j = 1:n) + sum(c[i, j] * u[i, j] for i=1:n, j=1:m)) 
    @objective(model, Min, F)
    return model, x, u, V
end

function apply_heuristic(model, x, u, V)
    @constraint(model, u .>= 0)
    @constraint(model, V .>= 0)
    return
end

""" Finds and adds worst case cuts for the facility location problem. """
function find_wc_cuts(model, x, u, V, xvals, uvals, Vvals, rho, Gamm)
    wc_model = Model(solver = GurobiSolver(OutputFlag = 0)) # suppressing printouts.
    @variable(wc_model, -rho <= z[1:m] <= rho)
    @variable(wc_model, normdummy[1:m] >= 0)
    @constraint(wc_model, [i=1:m], normdummy[i] >= z[i])
    @constraint(wc_model, [i=1:m], normdummy[i] >= -z[i])
    @constraint(wc_model, sum(normdummy) <= Gamm)
    
    count = length(model.linconstr)
    for i = 1:n # nonnegativity constraints
        for j=1:m
            @objective(wc_model, Min, uvals[i,j] + sum(Vvals[i,j,:] .* z))
            solve(wc_model)
            if getobjectivevalue(wc_model) < -1e-5 # if constraint is violated
                new_z = getvalue(z)
                @constraint(model, u[i,j] + sum(V[i,j,:] .* new_z) >= 0)
            end
        end
    end
    @info("$(length(model.linconstr) - count) nonnegativity cuts added. ")
    count = length(model.linconstr)
    for j = 1:m # demand constraints
        dt = zeros(n, m);
        [dt[i,j] = 1 for i=1:n];
        @objective(wc_model, Min, sum(dt .* uvals) + sum(Vvals[:,j,:] * z) - d[j] - (P*z)[j])
        solve(wc_model)
        if getobjectivevalue(wc_model) < -1e-5 # if constraint is violated
            new_z = getvalue(z)
            @constraint(model, sum(dt .* u) + sum(V[:,j,:] * new_z) >= d[j] + (P*new_z)[j])
        end
    end
    @info("$(length(model.linconstr) - count) demand constraint cuts added.")
    count = length(model.linconstr)
    for i = 1:n # capacity constraints
        dt = zeros(n, m);
        [dt[i,j] = 1 for j=1:m];
        @objective(wc_model, Min, - sum(dt .* uvals) - sum(Vvals[i,:,:] * z) + s[i] * xvals[i])
        solve(wc_model)
        if getobjectivevalue(wc_model) < -1e-5 # if constraint is violated
            new_z = getvalue(z)
            @constraint(model, sum(dt .* u) + sum(V[i,:,:] * new_z) <= s[i] * x[i])
        end
    end
    @info("$(length(model.linconstr) - count) capacity constraint cuts added.")
    # objective
    @objective(wc_model, Max, sum(f[j] * xvals[j] for j = 1:n) + sum(c[i, j] * (uvals[i, j] + sum(Vvals[i,j,:] .* z)) for i=1:n, j=1:m))
    solve(wc_model)
    if getobjectivevalue(wc_model) > getvalue(model.obj) + 1e-5 # if constraint is violated
        new_z = getvalue(z)
        @constraint(model, model.obj >= sum(f[j] * x[j] for j = 1:n) + sum(c[i, j] * (u[i, j] + sum(V[i,j,:] .* new_z)) for i=1:n, j=1:m))
        @info("Objective cut added.")
    end
end

""" Plots the solution of the facility location model. 
Blue circles are active facilities with different capacities.
Orange plus signs are other potential facility locations.
Rays describe connections between facilities and demand nodes. 
"""
function plot_solution(model, x, y, cost = nothing)
    plt = scatter(facilities[:, 1], facilities[:, 2], markersize = 0.4 .* s .* getvalue(x))
    scatter!(facilities[:, 1], facilities[:, 2], markersize = 0.4 .* s, markershape = :+)
    for i=1:n
        for j=1:m
            if getvalue(y[i,j]) >= 1e-10
                plot!([customers[j, 1], facilities[i,1]], [customers[j,2], facilities[i,2]], linewidth = getvalue(y[i,j]), legend=false)
        
            end
        end
    end
    if cost == nothing
        scatter!(customers[:, 1], customers[:, 2], markersize = 3*d, 
                title = "Total cost: $(round(getobjectivevalue(model), sigdigits=5))")
    else
        scatter!(customers[:, 1], customers[:, 2], markersize = 3*d, 
        title = "Total cost: $(round(cost,sigdigits=5))")
    end
    println("Facility cost: $(getvalue(sum(f[j] * x[j] for j = 1:n)))")
    println("Transportation cost: $(getvalue(sum(c[i, j] * y[i, j] for i=1:n, j=1:m)))")
    return plt
end

rho = 1
Gamm = 5
model, x, u, V = CP_facility_model(c, f)
apply_heuristic(model, x, u, V)

# Do 15 iterations of cuts
for i=1:15
    @info("Iteration $(i).")
    modelsize = length(model.linconstr)
    solve(model)
    xvals, uvals, Vvals = getvalue(x), getvalue(u), getvalue(V)
    find_wc_cuts(model, x, u, V, xvals, uvals, Vvals , rho, Gamm)
    if length(model.linconstr) == modelsize
        @info("Optimum reached.")
        @info("Optimal cost:$(getobjectivevalue(model)).")
        break
    end
end

# Problem should converge in ~100 adversarial iterations. 

plot_solution(model, x, u)