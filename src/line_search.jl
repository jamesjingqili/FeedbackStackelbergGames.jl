function KKT_residual_fbst(
    π,
    g::Constrained_LQGame,
    nonlinear_game::game,
    current_op::trajectory,
    )
    # step 1: construct the KKT matrix
    Mₜ, Nₜ, nₜ = constrained_fbst_lq_solver!(π, g, true)
    x = current_op.x
    u = current_op.u
    μ = current_op.μ # equality constraints
    λ = current_op.λ # dynamics
    η = current_op.η # future strategies
    ψ = current_op.ψ # follower's future strategies
    γ = current_op.γ # inequality constraints
    # ψ = current_op.ψ # leader's future strategies
    
    l = g.equality_constraints_size
    ll = g.inequality_constraints_size
    x0 = g.x0
    # extract state, total control, and each player's control dimensions
    nx, nu, m = g.nx, g.nu, length(g.players_u_index_list[1])
    T = g.horizon
    # m is the input size of agent i, and T is the horizon.
    num_player = g.n_players # number of player
    terminal_block_size = nu + l*num_player + nx*num_player + m + nx + l*num_player
    non_terminal_block_size = nu + l*num_player + nx + nx*num_player + m + (num_player-1)*nu
    M_size = size(Mₜ, 1)

    # step 2: construct the solution vector
    solution = BlockVector(zeros(M_size), vcat([non_terminal_block_size for t in 1:T-1], [terminal_block_size]))
    solution[Block(T)] = vcat(u[T], μ[T], λ[T], ψ[T], x[T+1], μ[T+1])
    loss = 0.0
    for t in 1:T-1
        solution[Block(t)] = vcat(u[t], μ[t], λ[t], η[t], ψ[t], x[t+1])
        dynamics_residual = x[t+1] - nonlinear_game.f_list[t](x[t], u[t])
        equality_constraint_residual = sum([
            nonlinear_game.equality_constraints_list[t][ii](vcat(x[t], u[t])) 
            for ii in 1:num_player])
        inequality_constraint_residual = sum([
            min(
                nonlinear_game.inequality_constraints_list[t][ii](vcat(x[t], u[t])), 
                zeros(nonlinear_game.inequality_constraints_size)
                ) 
            for ii in 1:num_player]) # G(x,u)<=0
        loss += norm(dynamics_residual, 2) + 
            norm(equality_constraint_residual, 2) + 
            norm(inequality_constraint_residual, 2)
    end
    # lastly, we consider the terminal time step:
    dynamics_residual = x[T+1] - nonlinear_game.f_list[T](x[T], u[T])
    equality_constraint_residual = sum([
        nonlinear_game.equality_constraints_list[T][ii](vcat(x[T], u[T])) 
        for ii in 1:num_player])
    inequality_constraint_residual = sum([
        min(
            nonlinear_game.inequality_constraints_list[T][ii](vcat(x[T], u[T])), 
            zeros(nonlinear_game.inequality_constraints_size)
            ) 
        for ii in 1:num_player]) # G(x,u)<=0
    loss += norm(dynamics_residual, 2) + 
        norm(equality_constraint_residual, 2) + 
        norm(inequality_constraint_residual, 2)

    # and terminal equality and inequality constraints:
    equality_constraint_residual = sum([
        nonlinear_game.terminal_equality_constraints_list[ii](x[T+1]) 
        for ii in 1:num_player])
    inequality_constraint_residual = sum([
        min(
            nonlinear_game.terminal_inequality_constraints_list[ii](x[T+1]), 
            zeros(nonlinear_game.inequality_constraints_size)
            ) 
        for ii in 1:num_player]) # G(x,u)<=0
    loss += norm(equality_constraint_residual, 2) + 
        norm(inequality_constraint_residual, 2)
    
    # step 3: multiply the KKT matrix and the solution vector!
    loss += norm(Mₜ*solution+Nₜ*x0+nₜ, 2)
    return loss
end

function KKT_residual_fbst(
    Mₜ,
    Nₜ,
    nₜ,
    g::Constrained_LQGame,
    current_op::trajectory
    )
    # step 1: construct the KKT matrix
    x = current_op.x
    u = current_op.u
    μ = current_op.μ
    λ = current_op.λ
    η = current_op.η
    ψ = current_op.ψ
    
    l = g.equality_constraints_size
    x0 = g.x0
    # extract state, total control, and each player's control dimensions
    nx, nu, m = g.nx, g.nu, length(g.players_u_index_list[1])
    T = g.horizon
    # m is the input size of agent i, and T is the horizon.
    num_player = g.n_players # number of player
    terminal_block_size = nu + l*num_player + nx*num_player + m + nx + l*num_player
    non_terminal_block_size = nu + l*num_player + nx + nx*num_player + m + (num_player-1)*nu
    M_size = size(Mₜ, 1)
    # step 2: construct the solution vector
    solution = BlockVector(zeros(M_size), vcat([non_terminal_block_size for t in 1:T-1], [terminal_block_size]))
    solution[Block(T)] = vcat(u[T], μ[T], λ[T], ψ[T], x[T+1], μ[T+1])
    for t in 1:T-1
        solution[Block(t)] = vcat(u[t], μ[t], λ[t], η[t], ψ[t], x[t+1])
    end
    # step 3: multiply the KKT matrix and the solution vector!
    loss = norm(Mₜ*solution + Nₜ*x0 + nₜ, 1)
    return loss
end


function line_search!(
    π::strategy,
    Δ::trajectory,
    current_op::trajectory, 
    lq_approx::Constrained_LQGame,
    nonlinear_game::game,
    α = 1.0, # initial step size
    β = 0.5 # step size reduction factor
    )

    # line search, update the linear policy
    loss = KKT_residual_fbst(
        π,
        lq_approx,
        nonlinear_game,
        current_op 
    )
    new_loss = loss+1
    next_op = deepcopy(current_op)
    counted_iterations = 0
    while new_loss > loss && counted_iterations < 50
        α *= β
        new_x = current_op.x + α * (Δ.x)
        new_u = current_op.u + α * (Δ.u)
        new_λ = current_op.λ + (α) * (Δ.λ - current_op.λ)
        new_η = current_op.η + (α) * (Δ.η - current_op.η)
        new_ψ = current_op.ψ + (α) * (Δ.ψ - current_op.ψ)
        new_μ = current_op.μ + (α) * (Δ.μ - current_op.μ)
        new_γ = current_op.γ + (α) * (Δ.γ - current_op.γ)
        next_op = trajectory(
            x = new_x,
            u = new_u,
            λ = new_λ,
            η = new_η,
            ψ = new_ψ,
            μ = new_μ,
            γ = new_γ
        )
        new_loss = KKT_residual_fbst(
            π,
            lq_approx,
            nonlinear_game,
            next_op
        )
        counted_iterations += 1
    end
    return α, new_loss, next_op
end


