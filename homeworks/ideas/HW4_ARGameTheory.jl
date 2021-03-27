using JuMP, JuMPeR, Gurobi, Random, Distributions, LinearAlgebra, Plots

n_moves = 2
n_players = 3
# Some payoff matrix
payoffM = Dict(1 => zeros(2,2,2), 2 => zeros(2,2,2), 3 => zeros(2,2,2))
# Jean's payoff
payoffM[1][:,:,1] = [3 2; 1 5]
payoffM[1][:,:,2] = [2 4; 0 7]
# Enrico's payoff
payoffM[2][:,:,1] = [5 4; 3 2]
payoffM[2][:,:,2] = [4 6; 2 7]
# Guy's payoff
payoffM[3][:,:,1] = [8 1; 2 4]
payoffM[3][:,:,2] = [5 3; 1 7]

function collaborative_game()
    m = Model(solver = GurobiSolver())
    @variable(m, 0 <= strategy[1:n_moves, 1:n_moves, 1:n_moves] <= 1)
    @variable(m, payoff[1:n_players])
    @constraint(m, sum(strategy) == 1)
    for i = 1:n_players
        @constraint(m, payoff[i] == sum(payoffM[i] .* strategy))
    end
    @objective(m, Max, sum(payoff))
    return m
end

function invididual_game(ind=1)
    m = Model(solver = GurobiSolver());
    @variable(m, 0 <= move <= 1);
    @variable(m, payoff >= 0);
    all_opposing_strats = reshape(collect(Base.product(zip([1,2], [1,2])...)), (1, n_moves^(n_players-1)))
    @objective(m, Max, sum(dot(payoffM[ind][:, all_opposing_strats[i][1], all_opposing_strats[i][2]], [move, 1-move])
                for i=1:length(all_opposing_strats))); # Average return
    solve(m)
    println("Average return fixed move: $(getvalue(move)) with payoff $(getobjectivevalue(m)).")
    
    for i = 1:length(all_opposing_strats)
        @constraint(m, payoff <= sum(dot(payoffM[ind][:, all_opposing_strats[i][1], all_opposing_strats[i][2]], 
                    [move, 1-move]))) # Minimum return 
    end
    @objective(m, Max, payoff)
    solve(m)
    println("Max-min return, fixed move: $(getvalue(move)) with payoff $(getvalue(payoff)).")

    m = Model(solver = GurobiSolver());
    @variable(m, 0 <= move <= 1);
    @variable(m, strat[1:n_players-1])
    @variable(m, strat0)
    @variable(m, 0 <= opp_moves[1:n_players-1] <= 1)
    @constraint(m, move == dot(strat, opp_moves) + strat0)
    @variable(m, payoff >= 0);
    @constraint(m, payoff <= )
    @objective(m, Max, payoff)
    




