for i = 1:n_groups # We embed the uncertainty in the errors!
    for j = i+1:n_groups
        for l = 1:n_traits
            y = @variable(rm, [1:n_traits])
            normdummy = @variable(rm, [1:n_traits])
            @constraint(rm, normdummy .>= y)
            @constraint(rm, normdummy .>= -y)
            infdummy = @variable(rm)
            @constraint(rm, [k = 1:n_traits], infdummy >= (x[k,j] - x[k,i] - y[k]))
            @constraint(rm, [k = 1:n_traits], infdummy >= -(x[k,j] - x[k,i] - y[k]))
            @constraint(rm, M[l] * n_ppg >= 
                        sum(data[k,l] .* (x[k,j] - x[k,i]) for k=1:n_people) + 
                        ρ * sum(normdummy) + Γ*infdummy)
            y = @variable(rm, [1:n_traits])
            normdummy = @variable(rm, [1:n_traits])
            @constraint(rm, normdummy .>= y)
            @constraint(rm, normdummy .>= -y)
            infdummy = @variable(rm)
            @constraint(rm, [k = 1:n_traits], infdummy >= (x[k,j] - x[k,i] - y[k]))
            @constraint(rm, [k = 1:n_traits], infdummy >= -(x[k,j] - x[k,i] - y[k]))
            @constraint(rm, M[l] * n_ppg >= 
                        - sum(data[k,l] .* (x[k,j] - x[k,i]) for k=1:n_people) +  
                        ρ * sum(normdummy) + Γ*infdummy)
#             Sometimes you have to get creative... linearization of the change of the variance. 
            y = @variable(rm, [1:n_traits])
            normdummy = @variable(rm, [1:n_traits])
            @constraint(rm, normdummy .>= y)
            @constraint(rm, normdummy .>= -y)
            infdummy = @variable(rm)
            @constraint(rm, [k = 1:n_traits], infdummy >= (2*data[k,l]*(x[k,j] - x[k,i]) - y[k]))
            @constraint(rm, [k = 1:n_traits], infdummy >= -(2*data[k,l]*(x[k,j] - x[k,i]) - y[k]))
            @constraint(rm, V[l] * n_ppg >= 
                        sum(data[k,l].^2 .* (x[k,j] - x[k,i]) for k=1:n_people) + 
                        ρ*sum(normdummy) + Γ*infdummy)
            y = @variable(rm, [1:n_traits])
            normdummy = @variable(rm, [1:n_traits])
            @constraint(rm, normdummy .>= y)
            @constraint(rm, normdummy .>= -y)
            infdummy = @variable(rm)
            @constraint(rm, [k = 1:n_traits], infdummy >= (2*data[k,l]*(x[k,j] - x[k,i]) - y[k]))
            @constraint(rm, [k = 1:n_traits], infdummy >= -(2*data[k,l]*(x[k,j] - x[k,i]) - y[k]))
            @constraint(rm, V[l] * n_ppg >= 
                        -sum(data[k,l].^2 .* (x[k,j] - x[k,i]) for k=1:n_people) + 
                        ρ*sum(normdummy) + Γ*infdummy)
        end
    end
end