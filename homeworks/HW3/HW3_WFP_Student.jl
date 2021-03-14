using JuMP, JuMPeR, Gurobi, Random, Distributions, LinearAlgebra, Plots, CSV, DataFrames

# CRUNCHING THE DATA
# NODES (I = International supplier; R = Regional supplier; L = Local market (both supply and deliver); D = delivery point)
N = []
N_I = []     # set of international suppliers
N_R = []     # set of regional suppliers
N_L = []     # set of local markets
N_D = []     # delivery points
dem = Dict() # set of demands
file = CSV.File("homeworks/HW3/syria_nodes.csv")
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
hc = DataFrame(CSV.File("homeworks/HW3/syria_edges.csv"))

# FOOD NUTRITION AND INTERNATIONAL COSTS
fooddata = DataFrame(CSV.File("homeworks/HW3/syria_foodnutrition.csv"))
intfoodcosts = select(fooddata, [:Food, :InternationalPrice])
commodities = sort(Array(intfoodcosts.Food)) # Commodities
select!(fooddata, Not([14,15]))
fooddata = Dict(fooddata.Food .=> eachrow(fooddata)) 
# Note: fooddata contains the nutrients provided by 100g of a commodity!

# FOOD COST ($/metric ton for regional suppliers)
pc = DataFrame(CSV.File("homeworks/HW3/syria_foodcost.csv"))
for int_supply_node in N_I # adding international prices to pc for easier processing
    for row in eachrow(intfoodcosts)
        append!(pc, DataFrame(:A => N_I, :Food => row.Food, :Price => row.InternationalPrice))
    end
end
pc = unique(pc)
international_items = DataFrame([r for r in eachrow(pc) if r.A in N_I])
regional_items =  DataFrame([r for r in eachrow(pc) if r.A in N_R])

# FOOD REQUIREMENTS (avg. per person per day)
foodreqs = DataFrame(CSV.File("homeworks/HW3/syria_foodreq.csv"))
select!(foodreqs, Not(:Type))
nutrients = String.(propertynames(foodreqs))
foodreqs = Dict(string(pptname) => foodreqs[1, pptname] for pptname in propertynames(foodreqs))

# FINALLY CREATING THE MODEL
m = Model(solver = GurobiSolver())

# Procurement and delivery
procurement_links = unique([row.A => row.Food for row in eachrow(pc)])       # all places where we can procure food
@variable(m, procurement[A = N, Food = commodities; (A => Food) in procurement_links] >= 0) # procurement in tons
@variable(m, delivery[N_D, commodities] >= 0)                                               # delivery in tons

# Total procurement cost
@variable(m, procurement_cost >= 0)
@constraint(m, procurement_cost >= sum(r[:Price] * procurement[r.A, r.Food] for r in eachrow(pc)))

# Transportation
transportation_links = unique([row.A => row.B for row in eachrow(hc)]) # all possible edges
@variable(m, transportation[A = N, B = N; (A => B) in transportation_links] >= 0)     # transportation in tons...
@variable(m, F[A = N, B = N, W = commodities; (A => B) in transportation_links] >= 0) # linked directly to F, also in tons. 
for r in eachrow(hc) # Linking transportation cost to total food transported on an edge
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

# Serving demand
@variable(m, ration_pp[commodities] >= 0) # Rations (kg/person) of commodities
@variable(m, nutrients_pp[nutrients] >= 0) # Total nutrients per person

# Making sure the rations are good nutritionally 
for nutrient in nutrients # Note the factor of 10 for conversion of 100g to kg (since rations are in kg/pp)
    @constraint(m, nutrients_pp[nutrient] == 10 * sum(ration_pp[commodity] * fooddata[commodity][nutrient] for commodity in commodities))
    @constraint(m, nutrients_pp[nutrient] >= foodreqs[nutrient])
end
for node in N_D
    for commodity in commodities
        @constraint(m, 1000*delivery[node, commodity] >= dem[node] * ration_pp[commodity])
    end
end

# Setting objectives
@objective(m, Min, procurement_cost + transportation_cost)

# Solving
solve(m)