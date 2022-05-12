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