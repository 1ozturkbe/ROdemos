include("utils.jl")

""" Cutting plane facility location model. 
    Note this is just the nominal problem, but with linear policies y(z) = u + Vz. """
function CP_facility_model(c::Matrix, f::Vector)
    n, m = size(c) 
    @assert length(f) == n
    model = Model(GLPK.Optimizer)

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
function find_wc_cuts(model, x, u, V, xvals, uvals, Vvals, rho, Gamm, atol = 1e-2)
    obj_value = objective_value(model)
    wc_model = Model(GLPK.Optimizer)
    @variable(wc_model, -rho <= z[1:m] <= rho)
    @variable(wc_model, normdummy[1:m] >= 0)
    @constraint(wc_model, [i=1:m], normdummy[i] >= z[i])
    @constraint(wc_model, [i=1:m], normdummy[i] >= -z[i])
    @constraint(wc_model, sum(normdummy) <= Gamm)
    
    count = 0
    for i = 1:n # nonnegativity constraints
        for j=1:m
            @objective(wc_model, Min, uvals[i,j] + sum(Vvals[i,j,:] .* z))
            optimize!(wc_model)
            if objective_value(wc_model) < -atol # if constraint is violated
                new_z = value.(z)
                @constraint(model, u[i,j] + sum(V[i,j,:] .* new_z) >= 0)
                count += 1
            end
        end
    end
    @info("$(count) nonnegativity cuts added. ")
    ct = count
    for j = 1:m # demand constraints
        dt = zeros(n, m);
        [dt[i,j] = 1 for i=1:n];
        @objective(wc_model, Min, sum(dt .* uvals) + sum(Vvals[:,j,:] * z) - d[j] - (P*z)[j])
        optimize!(wc_model)
        if objective_value(wc_model) < -atol # if constraint is violated
            new_z = value.(z)
            @constraint(model, sum(dt .* u) + sum(V[:,j,:] * new_z) >= d[j] + (P*new_z)[j])
            count += 1
        end
    end
    @info("$(count - ct) demand constraint cuts added.")
    ct = count
    for i = 1:n # capacity constraints
        dt = zeros(n, m);
        [dt[i,j] = 1 for j=1:m];
        @objective(wc_model, Min, - sum(dt .* uvals) - sum(Vvals[i,:,:] * z) + s[i] * xvals[i])
        optimize!(wc_model)
        if objective_value(wc_model) < -atol # if constraint is violated
            new_z = value.(z)
            @constraint(model, sum(dt .* u) + sum(V[i,:,:] * new_z) <= s[i] * x[i])
            count += 1
        end
    end
    @info("$(count - ct) capacity constraint cuts added.")
    ct = count
    # objective
    @objective(wc_model, Max, sum(f[j] * xvals[j] for j = 1:n) + sum(c[i, j] * (uvals[i, j] + sum(Vvals[i,j,:] .* z)) for i=1:n, j=1:m))
    optimize!(wc_model)
    if objective_value(wc_model) > obj_value + atol # if constraint is violated
        new_z = value.(z)
        @constraint(model, objective_function(model) >= sum(f[j] * x[j] for j = 1:n) + sum(c[i, j] * (u[i, j] + sum(V[i,j,:] .* new_z)) for i=1:n, j=1:m))
        count += 1
        @info("Objective cut added.")
    end
    return count
end

rho = 1
Gamm = 5
model, x, u, V = CP_facility_model(c, f)
unset_binary.(x) # First solve a relaxation of the problem.
apply_heuristic(model, x, u, V)
optimize!(model)

# Do 25 iterations of the relaxed problem, and then tighten binary variables. 
for i=1:100
    @info("Iteration $(i).")
    xvals, uvals, Vvals = value.(x), value.(u), value.(V)
    count = find_wc_cuts(model, x, u, V, xvals, uvals, Vvals , rho, Gamm)
    if i == 25
        set_binary.(x)
    end
    if count == 0
        @info("Optimum reached.")
        @info("Optimal cost:$(objective_value(model)).")
        break
    end
    optimize!(model)
end
# Problem should converge in ~100 adversarial iterations. 
xvals, uvals, Vvals = value.(x), value.(u), value.(V)
plot_solution(model, xvals, uvals)