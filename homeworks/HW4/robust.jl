include("utils.jl")

""" Nominal facility location model. """
function facility_model(c::Matrix, f::Vector)
    n, m = size(c) 
    @assert length(f) == n
    model = Model(GLPK.Optimizer)
    
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

""" Robust facility location model, with JuMP using the robust counterpart. """
function robust_facility_model(c::Matrix, f::Vector, rho::Real, Gamm::Real)
    n, m = size(c) 
    @assert length(f) == n
    model = Model(GLPK.Optimizer)
    # VARIABLES
    @variable(model, x[1:n], Bin)                 # Facility locations
    @variable(model, u[1:n, 1:m] >= 0)            # Flow variables

    # CONSTRAINTS
    for j=1:m
        infdummy = @variable(model)
        normdummy = @variable(model, [1:m])
        y = @variable(model, [1:m])
        @constraint(model, [i=1:m], normdummy[i] >= y[i])
        @constraint(model, [i=1:m], normdummy[i] >= -y[i])
        @constraint(model, [i=1:m], infdummy >= -y[i] - P[j,i])
        @constraint(model, [i=1:m], infdummy >= y[i] + P[j,i])
        @constraint(model, sum(u[i, j] for i = 1:n) >= d[j] + rho*sum(normdummy) + Gamm * infdummy) 
    end
    @constraint(model, [i=1:n], sum(u[i, j] for j = 1:m) <= s[i] * x[i]) # capacity constraint
    # OBJECTIVE
    @objective(model, Min, 
            sum(f[j] * x[j] for j = 1:n) + sum(c[i, j] * u[i, j] for i=1:n, j=1:m))
    return model, x, u
end

# Nominal model (3.1)
model, x, y = facility_model(c, f)
optimize!(model)
plt = plot_solution(model, x, y)
@show plt

# Robust model (3.2)
rho = 1
Gamm = 5
model, x, y = robust_facility_model(c, f, rho, Gamm)
optimize!(model)
plt = plot_solution(model, x, y)
@show plt

# Different radii of influence
R_Ds = collect(0:0.05:0.75)
R_Ds[1] = 1e-5

# Unscaled P (3.3)
df = DataFrame("R_D" => [], "sumP" => [], "Gamma" => [], "f+c" => [], "f" => [], "c" => [], "nP" => [], "nx" => [])
for k=1:length(R_Ds)
    global P = [0.2*exp(-1/R_Ds[k] * LinearAlgebra.norm(customers[i, :] .- customers[j, :])[1]) for j=1:m, i=1:m];
    global P = (P .>= 0.2*exp(-1/R_Ds[k] .* R_Ds[k])) .* P
    println("Nonzero Ps: $(sum(P .> 0))")
    model, x, y = robust_facility_model(c, f, rho, Gamm)
    optimize!(model)
    push!(df, Dict("R_D" => R_Ds[k],
                     "sumP" => sum(P),
                     "Gamma" => Gamm,
                     "f+c" => objective_value(model),   
                     "f" => value(sum(f[j] * x[j] for j = 1:n)),
                     "c" => value(sum(c[i, j] * y[i, j] for i=1:n, j=1:m)),
                     "nP" => sum(P .> 0),
                     "nx" => sum(value.(x))))
end

# Column-wise scaled P (3.7)
df = DataFrame("R_D" => [], "sumP" => [], "Gamma" => [], "f+c" => [], "f" => [], "c" => [], "nP" => [], "nx" => [])
P_nominal = [0.2*exp(-1/R_D .*LinearAlgebra.norm(customers[i, :] .- customers[j, :])[1]) for j=1:m, i=1:m];
P_nominal = (P_nominal .>= 0.2*exp(-1/R_D .* R_D)) .* P_nominal 
for k=1:length(R_Ds)
    global P = [0.2*exp(-1/R_Ds[k] * LinearAlgebra.norm(customers[i, :] .- customers[j, :])[1]) for j=1:m, i=1:m];
    global P = (P .>= 0.2*exp(-1/R_Ds[k] .* R_Ds[k])) .* P
    for i=1:m
        global P[i,:] = P[i,:] .* sum(P_nominal[i,:]) ./ sum(P[i,:])
    end
    println("Nonzero Ps: $(sum(P .> 0))")
    model, x, y = robust_facility_model(c, f, rho, Gamm)
    optimize!(model)
    push!(df, Dict("R_D" => R_Ds[k],
                     "sumP" => sum(P),
                     "Gamma" => Gamm,
                     "f+c" => objective_value(model),   
                     "f" => value(sum(f[j] * x[j] for j = 1:n)),
                     "c" => value(sum(c[i, j] * y[i, j] for i=1:n, j=1:m)),
                     "nP" => sum(P .> 0),
                     "nx" => sum(value.(x))))
end