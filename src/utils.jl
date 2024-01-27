function initialization_of_π(g::game)
    return [[zeros(g.nu, g.nx) for t in 1:g.horizon], [zeros(g.nu) for t in 1:g.horizon]]
end


function initialization_of_lq_approximation(
    g::game, 
    current_op::trajectory
    )
    lq_approx = LQGame(horizon = g.horizon, 
        n_players = g.n_players, 
        nx = g.nx, 
        nu = g.nu, 
        players_u_index_list = g.players_u_index_list,
        x0 = g.x0)
    lq_approximation!(lq_approx, g, current_op)
    return lq_approx
end

function relu(x)
    # we require x to be a vector or scalar
    m = length(x)
    if m == 1
        return max(0, x[1])
    else
        return [max(0, x[ii]) for ii in 1:m]
    end
end