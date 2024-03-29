# Activating Julia environment
using Pkg
Pkg.activate(".")

# Packages
using JuMP, MathOptInterface, Gurobi, Random, Distributions, LinearAlgebra, CSV, DataFrames
MOI = MathOptInterface

# CRUNCHING THE DATA (just like in lecture)
# NODES (I = International supplier; R = Regional supplier; L = Local market (both supply and deliver); D = delivery point)
N = []
N_I = []     # set of international suppliers
N_R = []     # set of regional suppliers
N_L = []     # set of local markets
N_D = []     # delivery points
dem = Dict() # set of demands
file = CSV.File("syria_nodes.csv")
for row in file
    push!(N, row.Name)
    if !ismissing(row.Demand)
        dem[row.Name] = row.Demand
    end
    if row.Type == "I"
        push!(N_I, row.Name)
    elseif row.Type == "R"
        push!(N_R, row.Name)
    elseif row.Type == "L"
        push!(N_L, row.Name) # Note: local markets supply and deliver goods at the given cost. 
    elseif row.Type == "D"
        push!(N_D, row.Name)
    else
        throw(ErrorException("rowType $(row.Type) not supported."))
    end
end

# EDGES
hc = DataFrame(CSV.File("syria_edges.csv"))

# FOOD NUTRITION AND INTERNATIONAL COSTS
fooddata = DataFrame(CSV.File("syria_foodnutrition.csv"))
intfoodcosts = select(fooddata, [:Food, :InternationalPrice])
commodities = sort(Array(intfoodcosts.Food)) # Commodities
select!(fooddata, Not([14,15]))
fooddata = Dict(fooddata.Food .=> eachrow(fooddata))

# FOOD COST ($/metric ton for regional suppliers)
pc = DataFrame(CSV.File("syria_foodcost.csv"))
pc.Food = convert.(String63, pc.Food)
for int_supply_node in N_I # adding international prices to pc for easier processing
    for row in eachrow(intfoodcosts)
        append!(pc, DataFrame(:A => N_I, :Food => row.Food, :Price => row.InternationalPrice))
    end
end
pc = unique(pc)
international_items = DataFrame([r for r in eachrow(pc) if r.A in N_I])
regional_items =  DataFrame([r for r in eachrow(pc) if r.A in N_R])

# FOOD REQUIREMENTS (avg. per person per day)
foodreqs = DataFrame(CSV.File("syria_foodreq.csv"))
select!(foodreqs, Not(:Type))
nutrients = String.(propertynames(foodreqs))
foodreqs = Dict(string(pptname) => foodreqs[1, pptname] for pptname in propertynames(foodreqs))


function WFP_model(slack::Union{Nothing, Real} = 1, pof::Real = 0.5)
    # FINALLY CREATING THE MODEL
    m = Model(Gurobi.Optimizer)
    m[:pof] = pof
    normalquant = quantile(Normal(0,1), pof)

    # Procurement and delivery
    procurement_links = unique([row.A => row.Food for row in eachrow(pc)])       # all places where we can procure food
    @variable(m, procurement[A = N, Food = commodities; (A => Food) in procurement_links] >= 0) # procurement in tons
    @variable(m, delivery[N_D, commodities] >= 0)                                               # delivery in tons
    # Total procurement cost
    @variable(m, procurement_cost >= 0)
    if normalquant == 0
        @constraint(m, procurement_cost >= sum(r[:Price] * procurement[r.A, r.Food] for r in eachrow(pc)))
    else
        @variable(m, proccert[1:size(pc, 1)])
        for i=1:size(pc,1)
            r = eachrow(pc)[i]
            if r.A in N_I
                @constraint(m, proccert[i] == 0.05*r.Price*procurement[r.A, r.Food])
            else 
                @constraint(m, proccert[i] == 0.30*r.Price*procurement[r.A, r.Food] +
                                    0.05*sum(cr.Price*procurement[cr.A, cr.Food] for cr in eachrow(pc) if cr.A == r.A))
            end
        end
        @constraint(m, [1/normalquant*(procurement_cost - sum(r[:Price] * procurement[r.A, r.Food] for r in eachrow(pc))),
                        proccert...] in MOI.SecondOrderCone(length(proccert) + 1)) 
        # Equivalent constraint in understandable format
        # @constraint(m, procurement_cost >= normalquant * norm(proccert, 2) + 
        #         sum(r[:Price] * procurement[r.A, r.Food] for r in eachrow(pc)))
    end

    # Transportation
    transportation_links = unique([row.A => row.B for row in eachrow(hc)])
    @variable(m, transportation[A = N, B = N; (A => B) in transportation_links] >= 0)
    @variable(m, F[A = N, B = N, W = commodities; (A => B) in transportation_links] >= 0) # connections F
    for r in eachrow(hc) # Linking transportation cost to total food hauled
        @constraint(m, transportation[r.A, r.B] == sum(F[r.A, r.B, commodity] for commodity in commodities))
    end
    # Total transportation cost
    @variable(m, transportation_cost >= 0)
    @constraint(m, transportation_cost >= sum(r.tCost * transportation[r.A, r.B] for r in eachrow(hc)))

    # Flow constraints
    for node in N
        valid_sources = [link.first for link in transportation_links if link.second == node]
        valid_sinks = [link.second for link in transportation_links if link.first == node]
        for commodity in commodities
            if (node =>  commodity) in procurement_links
                @constraint(m, procurement[node, commodity] + sum(F[source, node, commodity] for source in valid_sources) == 
                                    sum(F[node, sink, commodity] for sink in valid_sinks))
            elseif node in N_D
                @assert length(valid_sinks) == 0
                @constraint(m, delivery[node, commodity] == sum(F[source, node, commodity] for source in valid_sources))
            else
                @constraint(m, sum(F[source, node, commodity] for source in valid_sources) == 
                                    sum(F[node, sink, commodity] for sink in valid_sinks))
            end
        end
    end

    # Other helpers
    @variable(m, international_p_costs >= 0)
    @constraint(m, international_p_costs == sum(procurement[r.A, r.Food] * r.Price for r in eachrow(international_items)))

    # Serving demand
    @variable(m, ration_pp[commodities] >= 0) # Rations (kg/person) of commodities
    @variable(m, nutrients_pp[nutrients] >= 0) # Total nutrients per person

    # Making sure the rations are good nutritionally 
    @variable(m, nutrient_slack >= 0)
    if slack isa Real
        @constraint(m, nutrient_slack == slack)
    end
    for nutrient in nutrients # Note the factor of 10 for conversion of 100g to kg (since rations are in kg/pp)
        @constraint(m, nutrients_pp[nutrient] == 10 * sum(ration_pp[commodity] * fooddata[commodity][nutrient] for commodity in commodities))
        @constraint(m, nutrients_pp[nutrient] >= nutrient_slack * foodreqs[nutrient])
    end
    for node in N_D
        for commodity in commodities
            @constraint(m, 1000*delivery[node, commodity] >= dem[node] * ration_pp[commodity])
        end
    end

    # Diet constraints
    # Achieving a greater than 4:1 ratio by mass of carbohydrates to protein, and a greater than 4:1 ratio by mass of carbohydrates to fats
    # for the same total energy intake. The energy stored in a gram of carbohydrate, protein and fat are 4kcal/g, 4kcal/g and 9kcal/g respectively. 
    @variable(m, carbs_pp >= 0)
    @constraint(m, 4*carbs_pp == nutrients_pp["Energy(kcal)"] - 4*nutrients_pp["Protein(g)"] - 9*nutrients_pp["Fat(g)"])
    @constraint(m, carbs_pp >= 4*nutrients_pp["Protein(g)"])
    @constraint(m, carbs_pp >= 4*nutrients_pp["Fat(g)"])    

    # Setting objectives
    @objective(m, Min, procurement_cost + transportation_cost)
    return m
end

function report_results(m)
    return DataFrame("PP" => m[:pof],
                     "TotalCost" => round(value(m[:procurement_cost] + m[:transportation_cost]), sigdigits = 5),
                     "Proc/Total" => round(value(m[:procurement_cost]) / value(m[:procurement_cost] + m[:transportation_cost]), sigdigits=3),
                     "Trans/Total" => round(value(m[:transportation_cost]) / value(m[:procurement_cost] + m[:transportation_cost]), sigdigits=3),
                     "Intl/TotalProc" => round(value(m[:international_p_costs]) / value(m[:procurement_cost]), sigdigits=3), 
                     "Slack" =>  round(value(m[:nutrient_slack]), sigdigits = 3),
                     "Cost/Person" => round(value(m[:procurement_cost] + m[:transportation_cost]) / 
                                        (value(m[:nutrient_slack]) * sum(values(dem))), sigdigits=5),
                     "NumActiveProc" => sum(values(value.(m[:procurement])) .> 1e-3),
                     "NumActiveTrans" => sum(values(value.(m[:transportation])) .> 1e-3))  
end


# Solving nominal problem
m = WFP_model(1)
optimize!(m);
report_results(m)

# Maximizing nutrition (optimal solution just scales the nutrition)
nm = WFP_model(nothing, 0.5) 
# set_optimizer_attribute_(nm, "OptimalityTol", 1e-8)
@constraint(nm, nm[:procurement_cost] + nm[:transportation_cost] <= 6000)
@objective(nm, Max, nm[:nutrient_slack])
optimize!(nm);
report_results(nm)

# Robust solutions with a budget limit
pofs = [75, 85, 90, 92, 95, 96, 97, 98, 99, 99.5]./100
robustdata_lim = report_results(nm)
for pof in pofs
    m_lim = WFP_model(nothing, pof)
    @constraint(m_lim, m_lim[:procurement_cost] + m_lim[:transportation_cost] <= 6000)
    @objective(m_lim, Max, m_lim[:nutrient_slack])
    optimize!(m_lim);
    df = report_results(m_lim)
    append!(robustdata_lim, df)
end

# Robust solutions without a budget limit
m_unlim = WFP_model(1)
optimize!(m_unlim);
robustdata_unlim = report_results(m_unlim)
for pof in pofs
    m_unlim = WFP_model(1, pof)
    optimize!(m_unlim);
    df = report_results(m_unlim)
    append!(robustdata_unlim, df)
end