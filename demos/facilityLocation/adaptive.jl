include("utils.jl")

""" Adaptive facility location model, with JuMP using robust counterpart,
    under budget uncertainty. 
    Note: We restrict u and V to be elementwise positive, for tractability! """
function adaptive_facility_model(c::Matrix, f::Vector, rho::Real, Gamm::Real, optimizer=GLPK.Optimizer)
    n, m = size(c) 
    @assert length(f) == n
    model = Model(optimizer)

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