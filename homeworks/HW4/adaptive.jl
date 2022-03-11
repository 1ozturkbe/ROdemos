# Activating Julia environment
using Pkg
Pkg.activate(".")

# Packages
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

""" Adaptive facility location model, with JuMP using robust counterpart,
    under budget uncertainty. 
    Note: We restrict u and V to be elementwise positive, for tractability! """
function adaptive_facility_model(c::Matrix, f::Vector, rho::Real, Gamm::Real)
    n, m = size(c) 
    @assert length(f) == n
    model = Model(solver = GurobiSolver())

    # VARIABLES 
    @variable(model, x[1:n], Bin)           # Facility locations
    @variable(model, V[1:n, 1:m, 1:m] >= 0) # Affine policy w.r.t each z
    @variable(model, u[1:n, 1:m] >= 0)     

    # CONSTRAINTS
    for i = 1:n # Positivity constraints
        for j = 1:m
            infdummy = @variable(model)
            normdummy = @variable(model, [1:m])
            y = @variable(model, [1:m])
            @constraint(model, [k=1:m], normdummy[k] >= y[k])
            @constraint(model, [k=1:m], normdummy[k] >= -y[k])
            @constraint(model, [k=1:m], infdummy >= -y[k] + V[i,j,k])
            @constraint(model, [k=1:m], infdummy >= y[k] - V[i,j,k])  
            @constraint(model, u[i,j] >= rho*sum(normdummy) + Gamm * infdummy)
        end
    end
    for j=1:m # Demand constraints
        infdummy = @variable(model)
        normdummy = @variable(model, [1:m])
        y = @variable(model, [1:m])
        @constraint(model, [i=1:m], normdummy[i] >= y[i])
        @constraint(model, [i=1:m], normdummy[i] >= -y[i])
        dt = zeros(n, m);
        [dt[i,j] = 1 for i=1:n];
        @constraint(model, [i=1:m], infdummy >= -y[i] - P[i,j] + sum(V[:,:,i].*dt))
        @constraint(model, [i=1:m], infdummy >= y[i] + P[i,j] - sum(V[:,:,i].*dt))
        @constraint(model, sum(dt .* u) >= d[j] + rho*sum(normdummy) + Gamm * infdummy) 
    end
    for i=1:n # Capacity constraints
        infdummy = @variable(model)
        normdummy = @variable(model, [1:m])
        y = @variable(model, [1:m])
        @constraint(model, [j=1:m], normdummy[j] >= y[j])
        @constraint(model, [j=1:m], normdummy[j] >= -y[j])  
        dt = zeros(n, m);
        [dt[i,j] = 1 for j=1:m];
        @constraint(model, [j=1:m], infdummy >= -y[j] + sum(V[:,:,j].*dt))
        @constraint(model, [j=1:m], infdummy >= y[j] - sum(V[:,:,j].*dt))
        @constraint(model, sum(dt .* u) + rho*sum(normdummy) + Gamm * infdummy <=
                    s[i] * x[i]) # capacity constraint
    end
    # OBJECTIVE
    @variable(model, F)
    infdummy = @variable(model)
    normdummy = @variable(model, [1:m])
    y = @variable(model, [1:m])
    @constraint(model, [j=1:m], normdummy[j] >= y[j])
    @constraint(model, [j=1:m], normdummy[j] >= -y[j])
    @constraint(model, [j=1:m], infdummy >= - y[j] + sum(c.* V[:,:,j]))
    @constraint(model, [j=1:m], infdummy >= y[j] - sum(c.* V[:,:,j]))
    @constraint(model, F >= sum(f[j] * x[j] for j = 1:n) + sum(c[i, j] * u[i, j] for i=1:n, j=1:m) + 
                                rho*sum(normdummy) + Gamm * infdummy)
    @objective(model, Min, F)
    return model, x, u, V
end


""" Plots the solution of the facility location model. 
Blue circles are active facilities with different capacities.
Orange plus signs are other potential facility locations.
Rays describe connections between facilities and demand nodes. 
"""
function plot_solution(model, x, y, cost = nothing)
    plt = scatter(facilities[:, 1], facilities[:, 2], markersize = 0.4 .* s .* value.(x))
    scatter!(facilities[:, 1], facilities[:, 2], markersize = 0.4 .* s, markershape = :+)
    for i=1:n
        for j=1:m
            if value(y[i,j]) >= 1e-10
                plot!([customers[j, 1], facilities[i,1]], [customers[j,2], facilities[i,2]], linewidth = value(y[i,j]), legend=false)
        
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
    println("Facility cost: $(value(sum(f[j] * x[j] for j = 1:n)))")
    println("Transportation cost: $(value(sum(c[i, j] * y[i, j] for i=1:n, j=1:m)))")
    return plt
end

# Adaptive model through robust counterpart
rho = 1
Gamm = 5
model, x, u, V = adaptive_facility_model(c, f, rho, Gamm)
@time solve(model)
plt = plot_solution(model, x, u)
@show plt