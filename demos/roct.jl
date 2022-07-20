using DataFrames, Distributions, JuMP, Random, LinearAlgebra, Gurobi, Plots

""" Uncorrected, 1d variance function. """
function var_1d(data)
    return var(data, corrected = false, dims = 1)
end
""" 1d mean function. """
function mean_1d(data)
    return mean(data, dims = 1)
end

""" Function to compute the elementwise maximum. Will be useful later! """
function elem_maximum(a...)
    na = []
    for i = 1:length(a[1])
        push!(na, maximum([elem[i] for elem in a]))
    end
    return na
end

""" Generates data from a Normal(0,1) distribution. """
function generate_random_people(n_people::Int64 = 20, n_traits::Int64 = 5, seed = 314)
    continuous_values = rand(MersenneTwister(seed), Normal(0.00, 1), (n_people, n_traits))
    return continuous_values
end

""" Optimized controlled trial model. """
function optimized_trial_model(data::Union{Matrix, DataFrame}, n_groups:: Int, n_ppg::Int; 
    target_means::Vector = zeros(size(data, 2)),
    target_variances::Vector = ones(size(data, 2)),
    regularizer::Real = 0.5, optimizer = Gurobi.Optimizer)
    m = Model(optimizer)
    n_people, n_traits = size(data)
    @variable(m, x[i=1:n_people, 1:n_groups], Bin)
    @variable(m, μ_p[i=1:n_groups, j=1:n_traits]) # Mean
    @variable(m, σ_p[i=1:n_groups, j=1:n_traits]) # Variance
    for j = 1:n_groups # Taking the mean and std deviation of parameters for each group
        @constraint(m, μ_p[j,:] .== 1/(n_ppg) * 
                        sum(data[i,:] .* x[i,j] for i=1:n_people))
        @constraint(m, σ_p[j,:] .== 1/(n_ppg) * 
                        sum((data[i,:] - target_means).^2 .* x[i,j] for i=1:n_people))
        @constraint(m, sum(x[:,j]) == n_ppg)
    end
    for i = 1:n_people
        @constraint(m, sum(x[i, :]) <= 1) # each patient only picked at most once
    end

    @variable(m, d)
    @variable(m, M[1:n_traits]) # mean error
    @variable(m, V[1:n_traits]) # variance error
    @objective(m, Min, sum(M) + regularizer*sum(V))
    for i = 1:n_groups
        @constraint(m, M[:] .>= μ_p[i,:] - target_means)
        @constraint(m, M[:] .>= -(μ_p[i,:] - target_means))
        @constraint(m, V[:] .>= σ_p[i, :] - target_variances)
        @constraint(m, V[:] .>= -(σ_p[i, :] - target_variances))
    end
    for i = 1:n_groups-1
        for j = i+1:n_groups
            @constraint(m, M[:] .>= μ_p[i, :] - μ_p[j,:])
            @constraint(m, M[:] .>= -(μ_p[i, :] - μ_p[j,:]))
            @constraint(m, V[:] .>= σ_p[i, :] - σ_p[j,:])
            @constraint(m, V[:] .>= -(σ_p[i, :] - σ_p[j,:]))
        end
    end
    return m
end

""" Robust ptimized controlled trial model, under a budget uncertainty set. """
function robust_optimized_trial_model(data::Union{Matrix, DataFrame}, n_groups:: Int, n_ppg::Int; 
    target_means::Vector = zeros(size(data, 2)),
    target_variances::Vector = ones(size(data, 2)),
    ρ::Real = 1, Γ::Real = size(data, 1)/5, 
    regularizer::Real = 0.5, optimizer = Gurobi.Optimizer)

    rm = Model(Gurobi.Optimizer)
    n_people, n_traits = size(data)
    @variable(rm, x[i=1:n_people, 1:n_groups], Bin)
    @variable(rm, μ_p[i=1:n_groups, j=1:n_traits]) # Mean
    @variable(rm, σ_p[i=1:n_groups, j=1:n_traits]) # Variance
    for j = 1:n_groups # Taking the mean and std deviation of parameters for each group
        @constraint(rm, μ_p[j,:] .== 1/n_ppg * 
                        sum(data[i,:].*x[i,j] for i=1:n_people))
        @constraint(rm, σ_p[j,:] .== 1/n_ppg * 
                        sum((data[i,:] - target_means).^2 .* x[i,j] for i=1:n_people))
        @constraint(rm, sum(x[:,j]) == n_ppg)
    end
    for i = 1:n_people
        @constraint(rm, sum(x[i, :]) <= 1)
    end
    @variable(rm, M[1:n_traits])
    @variable(rm, V[1:n_traits])
    @objective(rm, Min, sum(M) + regularizer*sum(V))

    # With the following budget uncertainty
    # @uncertain_variable(rm, z[1:n_people, 1:n_traits])
    # @constraint(rm, norm(z, 1) <= Γ)
    # @constraint(rm, -ρ .<= z .<= ρ)  

    # FIRST, THE ROBUST COUNTERPART FOR THE INTRA-GROUP ERRORS.
    for g1 = 1:n_groups-1
        for g2 = g1+1:n_groups
            for l = 1:n_traits
                y = @variable(rm, [1:n_people])
                normdummy = @variable(rm, [1:n_people])
                @constraint(rm, normdummy .>= y)
                @constraint(rm, normdummy .>= -y)
                infdummy = @variable(rm)
                @constraint(rm, [i = 1:n_people], infdummy >= (x[i,g1] - x[i,g2] - y[i]))
                @constraint(rm, [i = 1:n_people], infdummy >= -(x[i,g1] - x[i,g2] - y[i]))
                @constraint(rm, M[l] * n_ppg >= 
                            sum(data[k,l] .* (x[k,g1] - x[k,g2]) for k=1:n_people) + 
                            ρ * sum(normdummy) + Γ*infdummy)
                y = @variable(rm, [1:n_people])
                normdummy = @variable(rm, [1:n_people])
                @constraint(rm, normdummy .>= y)
                @constraint(rm, normdummy .>= -y)
                infdummy = @variable(rm)
                @constraint(rm, [i = 1:n_people], infdummy >= (x[i,g1] - x[i,g2] - y[i]))
                @constraint(rm, [i = 1:n_people], infdummy >= -(x[i,g1] - x[i,g2] - y[i]))
                @constraint(rm, M[l] * n_ppg >= 
                            -(sum(data[k,l] .* (x[k,g1] - x[k,g2]) for k=1:n_people)) + 
                            ρ * sum(normdummy) + Γ*infdummy)
                # Sometimes you have to get creative... linearization of the change of the variance. 
                y = @variable(rm, [1:n_people])
                normdummy = @variable(rm, [1:n_people])
                @constraint(rm, normdummy .>= y)
                @constraint(rm, normdummy .>= -y)
                infdummy = @variable(rm)
                @constraint(rm, [i = 1:n_people], infdummy >= (2*(data[i,l] - target_means[l])*(x[i,g1] - x[i,g2]) - y[i]))
                @constraint(rm, [i = 1:n_people], infdummy >= -(2*(data[i,l] - target_means[l])*(x[i,g1] - x[i,g2]) - y[i]))
                @constraint(rm, V[l] * n_ppg >= 
                            sum((data[k,l] - target_means[l]).^2 .* (x[k,g1] - x[k,g2]) for k=1:n_people) + 
                            ρ*sum(normdummy) + Γ*infdummy)
                y = @variable(rm, [1:n_people])
                normdummy = @variable(rm, [1:n_people])
                @constraint(rm, normdummy .>= y)
                @constraint(rm, normdummy .>= -y)
                infdummy = @variable(rm)
                @constraint(rm, [i = 1:n_people], infdummy >= (2*(data[i,l] - target_means[l])*(x[i,g1] - x[i,g2]) - y[i]))
                @constraint(rm, [i = 1:n_people], infdummy >= -(2*(data[i,l] - target_means[l])*(x[i,g1] - x[i,g2]) - y[i]))
                @constraint(rm, V[l] * n_ppg >= 
                            -(sum((data[k,l] - target_means[l]).^2 .* (x[k,g1] - x[k,g2]) for k=1:n_people)) + 
                            ρ*sum(normdummy) + Γ*infdummy)
            end
        end
    end
    # THEN, THE ROBUST COUNTERPART FOR THE INTER_POPULATION ERRORS
    for g1 = 1:n_groups
        for l = 1:n_traits
            y = @variable(rm, [1:n_people])
            normdummy = @variable(rm, [1:n_people])
            @constraint(rm, normdummy .>= y)
            @constraint(rm, normdummy .>= -y)
            infdummy = @variable(rm)
            @constraint(rm, [i = 1:n_people], infdummy >= (x[i,g1] - y[i]))
            @constraint(rm, [i = 1:n_people], infdummy >= -(x[i,g1] - y[i]))
            @constraint(rm, M[l] * n_ppg >= 
                        sum(data[k,l] .* x[k,g1] for k=1:n_people) - target_means[l] * n_ppg +
                        ρ * sum(normdummy) + Γ*infdummy)
            y = @variable(rm, [1:n_people])
            normdummy = @variable(rm, [1:n_people])
            @constraint(rm, normdummy .>= y)
            @constraint(rm, normdummy .>= -y)
            infdummy = @variable(rm)
            @constraint(rm, [i = 1:n_people], infdummy >= (x[i,g1] - y[i]))
            @constraint(rm, [i = 1:n_people], infdummy >= -(x[i,g1] - y[i]))
            @constraint(rm, M[l] * n_ppg >= 
                        -(sum(data[k,l] .* x[k,g1] for k=1:n_people) - target_means[l] * n_ppg) +
                        ρ * sum(normdummy) + Γ*infdummy)
            # Now for the variance
            y = @variable(rm, [1:n_people])
            normdummy = @variable(rm, [1:n_people])
            @constraint(rm, normdummy .>= y)
            @constraint(rm, normdummy .>= -y)
            infdummy = @variable(rm)
            @constraint(rm, [i = 1:n_people], infdummy >= (2*(data[i,l] - target_means[l])*(x[i,g1]) - y[i]))
            @constraint(rm, [i = 1:n_people], infdummy >= -(2*(data[i,l] - target_means[l])*(x[i,g1]) - y[i]))
            @constraint(rm, V[l] * n_ppg >= 
                        sum((data[k,l] - target_means[l]).^2 .* x[k,g1] for k=1:n_people) - target_variances[l] * n_ppg + 
                        ρ*sum(normdummy) + Γ*infdummy)
            y = @variable(rm, [1:n_people])
            normdummy = @variable(rm, [1:n_people])
            @constraint(rm, normdummy .>= y)
            @constraint(rm, normdummy .>= -y)
            infdummy = @variable(rm)
            @constraint(rm, [i = 1:n_people], infdummy >= (2*(data[i,l] - target_means[l])*(x[i,g1]) - y[i]))
            @constraint(rm, [i = 1:n_people], infdummy >= -(2*(data[i,l] - target_means[l])*(x[i,g1]) - y[i]))
            @constraint(rm, V[l] * n_ppg >= 
                        -(sum((data[k,l] - target_means[l]).^2 .* x[k,g1] for k=1:n_people) - target_variances[l] * n_ppg) + 
                        ρ*sum(normdummy) + Γ*infdummy)
        end
    end
    return rm
end

function compute_worst_case(rm::JuMP.Model, ctrl_idxs::Vector{Int}, vacc_idxs::Vector{Int}; silent::Bool = true)
    n_ppg = length(ctrl_idxs)
    @assert length(ctrl_idxs) == length(vacc_idxs)
    rm_copy = copy(rm)
    set_optimizer(rm_copy, Gurobi.Optimizer)
    if silent
        set_optimizer_attribute(rm_copy, "OutputFlag", 0)
    end
    constrs = [@constraint(rm_copy, rm_copy[:x][ctrl_idxs,1] .== ones(n_ppg))..., @constraint(rm_copy, rm_copy[:x][vacc_idxs, 2] .== ones(n_ppg))...]
    optimize!(rm_copy)
    return objective_value(rm_copy)
end

function group_subjects(m::JuMP.Model, n_groups::Int)
    return (findall(x -> x >= 0.8, Array(value.(m[:x][:,i]))) for i in 1:n_groups)
end

function print_details(data::Union{Matrix, DataFrame}, ctrl_idxs::Vector{Int}, vacc_idxs::Vector{Int})
    println("Control group: ", ctrl_idxs)
    println("Vaccine group: ", vacc_idxs)
    println("Mean traits of control group: ", round.(mean_1d(data[ctrl_idxs, :]); sigdigits = 4))
    println("Mean traits of vaccine group: ", round.(mean_1d(data[vacc_idxs, :]); sigdigits = 4))
    println("Var of traits of control group: ", round.(var_1d(data[ctrl_idxs, :]); sigdigits = 4))
    println("Var of traits of vaccine group: ", round.(var_1d(data[vacc_idxs, :]); sigdigits = 4))
    mean_errors = abs.(mean_1d(data[ctrl_idxs, :])' - mean_1d(data[vacc_idxs, :])')
    println("Nominal objective: ", round(
        sum(elem_maximum(abs.(mean_1d(data[ctrl_idxs, :])' .- mean_1d(data[vacc_idxs, :])'),
                         abs.(mean_1d(data[vacc_idxs, :])' .- target_means),
                         abs.(mean_1d(data[ctrl_idxs, :])' .- target_means))) + 
        0.5*sum(elem_maximum(abs.(var_1d(data[ctrl_idxs, :])' .- var_1d(data[vacc_idxs, :])'),
                             abs.(var_1d(data[ctrl_idxs, :])' .- target_variances),
                             abs.(var_1d(data[vacc_idxs, :])' .- target_variances))); sigdigits = 4))
    return
end

function plot_errors(data::Union{Matrix, DataFrame}, ctrl_idxs::Vector{Int}, vacc_idxs::Vector{Int})
    p1 = bar(1:5, vec(sum(data[ctrl_idxs, :], dims=1)/n_ppg .-  sum(data[vacc_idxs, :], dims=1)/n_ppg))
    xlabel!("Features")
    ylabel!("Intra-group mean error")
    ylims!((-0.8, 0.8))
    p2 = bar(1:5, vec(sum(data[ctrl_idxs, :], dims=1)/n_ppg) - vec(sum(data[:, :], dims=1)/n_people))
    xlabel!("Features")
    ylabel!("Control group mean error w.r.t. target")
    ylims!((-0.8, 0.8))
    p3 = bar(1:5, vec(sum(data[vacc_idxs, :], dims=1)/n_ppg) - vec(sum(data[:, :], dims=1)/n_people))
    xlabel!("Features")
    ylabel!("Experiment group mean error w.r.t. target")
    ylims!((-0.8, 0.8))
    plot(p1, p2, p3, layout = (1,3), legend = false)
end