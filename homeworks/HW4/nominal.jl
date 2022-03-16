include("utils.jl")

""" Nominal facility location model. """
function facility_model(c::Matrix, f::Vector)
    n, m = size(c) 
    @assert length(f) == n
    model = Model(Gurobi.Optimizer)

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

model, x, y = facility_model(c, f)

optimize!(model)

plt = plot_solution(model, x, y)
