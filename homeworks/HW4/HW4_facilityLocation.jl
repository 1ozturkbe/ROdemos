using JuMP, JuMPeR, Gurobi, Random, Distributions, LinearAlgebra, DataFrames, Plots

n = 10 # Possible facility locations
m = 200 # Number of customers

# Generating random data (please don't change the seeds.)
facilities = rand(MersenneTwister(314), n,2);
customers = rand(MersenneTwister(2), m, 2); 
c = [LinearAlgebra.norm(customers[i, :] .- facilities[j, :])[1] for j=1:n, i=1:m];
f = rand(MersenneTwister(3), n)*10 .+ 10;
s = rand(MersenneTwister(4), n)*10 .+ 20;
d = rand(MersenneTwister(5), m)

""" Nominal facility location model. """
function facility_model(c::Matrix, f::Vector)
    n, m = size(c) 
    @assert length(f) == n
    model = Model(solver = GurobiSolver())

    # VARIABLES
    @variable(model, x[1:n], Bin)      # Facility locations
    @variable(model, y[1:n, 1:m] >= 0) # Flow variables
    # CONSTRAINTS
    @constraint(model, [j=1:m], sum(y[i, j] for i = 1:n) >= d[j])      # demand constraint
    @constraint(model, [i=1:n], sum(y[i, j] for j = 1:m) <= s[i]*x[i]) # capacity constraint
    # OBJECTIVE
    @objective(model, Min, 
            sum(f[j] * x[j] for j = 1:n) + sum(c[i, j] * y[i, j] for i=1:n, j=1:m))
    return model, x, y
end

""" Plots the solution of the facility location model. 
Blue circles are active facilities with different capacities.
Orange plus signs are other potential facility locations.
Rays describe connections between facilities and demand nodes. 
"""
function plot_solution(model, x, y)
    plt = scatter(facilities[:, 1], facilities[:, 2], markersize = 0.4 .* s .* getvalue(x))
    scatter!(facilities[:, 1], facilities[:, 2], markersize = 0.4 .* s, markershape = :+)
    for i=1:n
        for j=1:m
            if getvalue(y[i,j]) >= 1e-10
                plot!([customers[j, 1], facilities[i,1]], [customers[j,2], facilities[i,2]], linewidth = getvalue(y[i,j]), legend=false)
        
            end
        end
    end
    scatter!(customers[:, 1], customers[:, 2], markersize = 3*d, 
            title = "Total cost: $(round(getobjectivevalue(model), sigdigits=5))")
    println("Facility cost: $(getvalue(sum(f[j] * x[j] for j = 1:n)))")
    println("Transportation cost: $(getvalue(sum(c[i, j] * y[i, j] for i=1:n, j=1:m)))")
    return plt
end

model, x, y = facility_model(c, f)

solve(model)

plt = plot_solution(model, x, y)

@show plt
