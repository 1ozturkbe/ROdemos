""" Plots the solution of the facility location model. 
Blue circles are active facilities with different capacities.
Orange plus signs are other potential facility locations.
Rays describe connections between facilities and demand nodes. 
"""
function plot_solution(model, x, y, cost = nothing)
    scatter(facilities[:, 1], facilities[:, 2], s = 0.4 .* s .* value.(x))
    scatter(facilities[:, 1], facilities[:, 2], s = 0.4 .* s, marker = :+)
    for i=1:n
        for j=1:m
            if value(y[i,j]) >= 1e-10
                plot([customers[j, 1], facilities[i,1]], [customers[j,2], facilities[i,2]], 
                    linewidth = value(y[i,j]))
            end
        end
    end
    if isnothing(cost)
        scatter(customers[:, 1], customers[:, 2], s = 3*d)
        title("Total cost: $(round(objective_value(model), digits=2))")
    else
        scatter(customers[:, 1], customers[:, 2], s = 3*d)
        title("Total cost: $(round(cost, digits=2))")
    end
    fac_cost = value(sum(f[j] * x[j] for j = 1:n))
    trans_cost = value(sum(c[i, j] * y[i, j] for i=1:n, j=1:m))
    println("Facility cost: $(round(fac_cost, digits=2))")
    println("Transportation cost: $(round(trans_cost, digits=2))")
end