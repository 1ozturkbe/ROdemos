# The now-defunct JuMPeR formulations of the problems are included here!

""" Robust facility location model, with JuMPeR. """
function robust_facility_model(c::Matrix, f::Vector, rho::Real, Gamm::Real)
    n, m = size(c) 
    @assert length(f) == n
    model = RobustModel(solver = GurobiSolver())
    # model = Model(solver = GurobiSolver())

    # VARIABLES
    @variable(model, x[1:n], Bin)                 # Facility locations
    @variable(model, y[1:n, 1:m] >= 0)            # Flow variables
    @uncertain(model, -rho <= z[1:m] <= rho)      #  Demand uncertainties
    @constraint(model, norm(z, 1) <= Gamm)

    # CONSTRAINTS
    @constraint(model, [j=1:m], sum(y[i, j] for i = 1:n) >= d[j] + (P*z)[j]) # demand constraint
    @constraint(model, [i=1:n], sum(y[i, j] for j = 1:m) <= s[i] * x[i]) # capacity constraint
    # OBJECTIVE
    @objective(model, Min, 
            sum(f[j] * x[j] for j = 1:n) + sum(c[i, j] * y[i, j] for i=1:n, j=1:m))
    return model, x, y
end


""" Adaptive facility location model, with JuMPeR and relaxation of binary variables. """
function adaptive_facility_model(c::Matrix, f::Vector, rho::Real, Gamm::Real)
    n, m = size(c) 
    @assert length(f) == n
    model = RobustModel(solver = GurobiSolver())

    # VARIABLES
    @variable(model, 0 <= x[1:n] <= 1)                 # Facility locations
    @uncertain(model, -rho <= z[1:m] <= rho)      #  Demand uncertainties
    @constraint(model, norm(z, 1) <= Gamm)
    @adaptive(model, y[i=1:n, j=1:m] >= 0, policy = Affine, depends_on=z) # Flow variables

    # CONSTRAINTS
    @constraint(model, [j=1:m], sum(y[i, j] for i = 1:n) >= d[j] + (P*z)[j]) # demand constraint
    @constraint(model, [i=1:n], sum(y[i, j] for j = 1:m) <= s[i] * x[i]) # capacity constraint
    # OBJECTIVE
    @variable(model, F)
    @constraint(model, F >= sum(f[j] * x[j] for j = 1:n) + sum(c[i, j] * y[i, j] for i=1:n, j=1:m))
    @objective(model, Min, F)
    return model, x, y
end

# Adaptive solution with greedy binary heuristic (3.4 and 3.6)
function iterate_adaptive_model(c, f, rho, Gamm)
    ints = []
    xvals = zeros(size(c, 1))
    while !all([val in [0,1] for val in xvals]) || isempty(ints)
        model, x, y = adaptive_facility_model(c, f, rho, Gamm)
        for int in ints
            @constraint(model, x[int] == 1)
        end
        optimize!(model)
        xvals = value.(x)
        nonint = [!(var in [0,1]) for var in xvals] 
        if sum(nonint) == 0
            return model, x, y, ints
        end
        max_val, max_idx = findmax(nonint.*xvals)
        push!(ints, max_idx)
    end
    return
end

# model, x, y, ints = iterate_adaptive_model(c, f, rho, Gamm)

# # Plot generation for adaptive solution (3.4 and 3.6)
# cost = objective_value(model)
# nonints = [val for val in 1:n if !(val in ints)]
# plot_model, x, y = facility_model(c, f)
# @constraint(plot_model, x[ints] .== 1)
# @constraint(plot_model, x[nonints] .== 0)
# optimize!(plot_model)
# plt = plot_solution(model, x, y, cost)
# @show plt