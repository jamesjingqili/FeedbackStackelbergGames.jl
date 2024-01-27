function nw_pdip_KKT_residual_fbst(
    π,
    g::Constrained_LQGame,
    nonlinear_game::game,
    current_op::trajectory,
    ρ::Float64
    )
    norm_mode = 2
    # step 1: construct the KKT matrix
    Mₜ, Nₜ, nₜ = nw_pdip_fbst_lq_solver!(π, g, current_op, ρ, true)
    x = current_op.x # in global coordinate? no, in local coordinate
    u = current_op.u # in global coordinate? no, in local coordinate
    μ = current_op.μ # equality constraints
    λ = current_op.λ # dynamics
    η = current_op.η # future strategies
    ψ = current_op.ψ # follower's future strategies
    γ = current_op.γ # inequality constraints
    s = current_op.s # slackness variables in local coordinate
    
    l = g.equality_constraints_size
    ll = g.inequality_constraints_size
    x0 = g.x0
    # extract state, total control, and each player's control dimensions
    nx, nu, m = g.nx, g.nu, length(g.players_u_index_list[1])
    T = g.horizon
    # m is the input size of agent i, and T is the horizon.
    num_player = g.n_players # number of player
    terminal_block_size = nu + ll*num_player+ll*num_player + l*num_player + nx*num_player + m + nx + ll*num_player+ll*num_player + l*num_player
    non_terminal_block_size = nu + ll*num_player + ll*num_player + l*num_player + nx + nx*num_player + m + (num_player-1)*nu
    M_row_size = size(Mₜ, 1)
    M_col_size = size(Mₜ, 2)

    # step 2: construct the solution vector
    loss = 0.0
    min_γ = minimum([minimum(γ[t]) for t in 1:T+1])
    min_s = minimum([minimum(s[t]) for t in 1:T+1])
    if min_s < 0.0
        println("min_s = ", min_s)
        @infiltrate
    end
    min_ineq = 1e-16
    homotopy_loss = 0.0
    # complementary_slackness_loss = 0.0 # including homotopy loss is enough in pdip!

    # ----------------------------------------------------------------------------------------------
    solution = BlockVector(zeros(M_col_size), vcat([non_terminal_block_size for t in 1:T-1], [terminal_block_size]))
    solution[Block(T)] = vcat(u[T], s[T], γ[T], μ[T], λ[T], ψ[T], x[T+1], s[T+1], γ[T+1], μ[T+1])
    sum_equality_constraint_residual = 0.0
    # In what follows, we construct loss
    # Part 1:
    for t in 1:T-1
        solution[Block(t)] = vcat(u[t], s[t], γ[t], μ[t], λ[t], η[t], ψ[t], x[t+1])
        dynamics_residual = x[t+1] - nonlinear_game.f_list[t](x[t], u[t])
        equality_constraint_residual = sum([
            nonlinear_game.equality_constraints_list[t][ii](vcat(x[t], u[t])) 
            for ii in 1:num_player]
        )
        vec_ineq = vcat(
            nonlinear_game.inequality_constraints_list[t][1](vcat(x[t], u[t])), 
            nonlinear_game.inequality_constraints_list[t][2](vcat(x[t], u[t]))
        )
        min_ineq = minimum([minimum(vec_ineq), min_ineq])
        inequality_constraint_residual = norm(
            vec_ineq - current_op.s[t], norm_mode
        )
        homotopy_loss += norm(
            current_op.γ[t].*current_op.s[t] - ρ*ones(2*ll), norm_mode
        )
        # @infiltrate
        # homotopy_loss += -sum(γ[t].*log.(s[t])) # TODO: test more and decide whether to use this or not!
        loss += norm(dynamics_residual, norm_mode) + 
            norm(equality_constraint_residual, norm_mode) + 
            norm(inequality_constraint_residual, norm_mode)
        sum_equality_constraint_residual += norm(equality_constraint_residual, norm_mode)
    end
    # @infiltrate
    # Part 2: at the T-th step
    dynamics_residual = x[T+1] - nonlinear_game.f_list[T](x[T], u[T])
    equality_constraint_residual = sum([
        nonlinear_game.equality_constraints_list[T][ii](vcat(x[T], u[T])) 
        for ii in 1:num_player]
    )
    vec_ineq = vcat(
        nonlinear_game.inequality_constraints_list[T][1](vcat(x[T], u[T])), 
        nonlinear_game.inequality_constraints_list[T][2](vcat(x[T], u[T]))
    )
    min_ineq = minimum([minimum(vec_ineq), min_ineq])
    inequality_constraint_residual = norm(
        vec_ineq - current_op.s[T], norm_mode
    )
    homotopy_loss += norm(
        current_op.γ[T].*current_op.s[T] - ρ*ones(2*ll), norm_mode
    )
    # homotopy_loss += -sum(γ[T].*log.(s[T]))
    loss += norm(dynamics_residual, norm_mode) + 
        norm(equality_constraint_residual, norm_mode) + 
        norm(inequality_constraint_residual, norm_mode)
    sum_equality_constraint_residual += norm(equality_constraint_residual, norm_mode)
    # @infiltrate
    # Part 3: at the terminal time step, we have terminal equality and inequality constraints:
    equality_constraint_residual = sum([
        nonlinear_game.terminal_equality_constraints_list[ii](x[end]) 
        for ii in 1:num_player]
    )
    vec_ineq = vcat(
        nonlinear_game.terminal_inequality_constraints_list[1](x[T+1]), 
        nonlinear_game.terminal_inequality_constraints_list[2](x[T+1])
    )
    min_ineq = minimum([minimum(vec_ineq), min_ineq])
    inequality_constraint_residual = norm(
        vec_ineq - current_op.s[T+1], norm_mode
    )
    homotopy_loss += norm(
        current_op.γ[T+1].*current_op.s[T+1] - ρ*ones(2*ll), norm_mode
    )
    # homotopy_loss += -sum(γ[T+1].*log.(s[T+1]))
    loss += norm(equality_constraint_residual, norm_mode) + 
        norm(inequality_constraint_residual, norm_mode)
    sum_equality_constraint_residual += norm(equality_constraint_residual, norm_mode)
    # @infiltrate    
    # step 3: multiply the KKT matrix and the solution vector!
    loss += norm(Mₜ*solution+Nₜ*x0+nₜ, norm_mode)
    loss += homotopy_loss
    # @infiltrate
    return loss, min_s, min_γ, homotopy_loss, min_ineq, sum_equality_constraint_residual
end

function nw_pdip_line_search(
    π::strategy,
    Δ::trajectory,
    current_op::trajectory, 
    lq_approx::Constrained_LQGame,
    nonlinear_game::game,
    ρ::Float64,
    α = 1.0, # initial step size
    β = 0.5 # step size reduction factor
    )

    # line search, update the linear policy
    loss, min_s, min_γ, homotopy_loss, min_ineq, eq_loss = nw_pdip_KKT_residual_fbst(
        π,
        lq_approx,
        nonlinear_game,
        current_op,
        ρ
    )
    new_loss = loss+1 # initialize to be greater than current loss
    new_homotopy_loss = homotopy_loss+1 # initialize to be greater than current loss
    new_min_s = min_s+1 # initialize to be greater than 0
    new_min_γ = min_γ+1 # initialize to be greater than 0
    new_min_ineq = min_ineq+1 # initialize to be greater than 0
    next_op = deepcopy(current_op)
    next_lq_approx = deepcopy(lq_approx)

    line_search_success = false
    # # TODO: notice that we used simple heuristic to update γ, which is not theoretically justified!
    α_γ = 1.0

    for iteration in 1:31
        if iteration == 31
            α = 0.0
        end
        new_s = current_op.s + α * (Δ.s)#(Δ.s - current_op.s)
        min_new_s = minimum([minimum(new_s[t]) for t in 1:nonlinear_game.horizon+1])
        if min_new_s  < 0.0
            # println("min_new_s = ", min_new_s)
            α *= β
        else
            new_x = current_op.x + α * (Δ.x)
            new_u = current_op.u + α * (Δ.u)
            new_λ = current_op.λ + α * (Δ.λ - current_op.λ)
            new_η = current_op.η + α * (Δ.η - current_op.η)
            new_ψ = current_op.ψ + α * (Δ.ψ - current_op.ψ)
            new_μ = current_op.μ + α * (Δ.μ - current_op.μ)
            α_γ = α
            new_γ = current_op.γ + α_γ * (Δ.γ - current_op.γ) # the Z variable in Nocedal's book
            # new_γ = current_op.γ + α_γ * Δ.γ
            next_op = trajectory(
                x = new_x,
                u = new_u,
                λ = new_λ,
                η = new_η,
                ψ = new_ψ,
                μ = new_μ,
                γ = new_γ,
                s = new_s
            )
            # @infiltrate
            lq_approximation!(next_lq_approx, nonlinear_game, next_op) # update lq approximation?

            new_loss, new_min_s, new_min_γ, new_homotopy_loss, new_min_ineq, new_eq_loss = nw_pdip_KKT_residual_fbst(
            π,
            next_lq_approx,
            nonlinear_game,
            next_op,
            ρ
            )
            # @infiltrate
            # println("new_loss = ", new_loss)
            if new_loss < loss && new_min_γ > 0 #&& new_min_s > 0 # && new_eq_loss < 1.1eq_loss # && new_min_ineq > 0
            # if new_loss < loss && new_min_γ >= 0 && new_min_ineq > 0
                line_search_success = true
                break
            end
            α *= β
        end

    end
    # if line_search_success == false
    #     α=0.0
    # end
    return α, new_loss, next_op, next_lq_approx, min_s, min_γ, new_homotopy_loss, α_γ, min_ineq
end

function fast_nw_pdip_line_search(
    π::strategy,
    Δ::trajectory,
    current_op::trajectory, 
    lq_approx::Constrained_LQGame,
    nonlinear_game::game,
    ρ::Float64,
    α = 1.0, # initial step size
    β = 0.5, # step size reduction factor
    )

    # line search, update the linear policy
    loss, min_s, min_γ, homotopy_loss, min_ineq = nw_pdip_KKT_residual_fbst(
        π,
        lq_approx,
        nonlinear_game,
        current_op,
        ρ
    )
    new_loss = loss+1 # initialize to be greater than current loss
    new_homotopy_loss = homotopy_loss+1 # initialize to be greater than current loss
    new_min_s = min_s+1 # initialize to be greater than 0
    new_min_γ = min_γ+1 # initialize to be greater than 0
    new_min_ineq = min_ineq+1 # initialize to be greater than 0
    next_op = deepcopy(current_op)
    next_lq_approx = deepcopy(lq_approx)

    line_search_success = false
    # # TODO: notice that we used simple heuristic to update γ, which is not theoretically justified!
    α_γ = 1.0

    for iteration in 1:31
        if iteration == 31
            α = 0.0
        end
        new_x = current_op.x + α * (Δ.x)
        new_u = current_op.u + α * (Δ.u)
        new_s = current_op.s + α * (Δ.s)#(Δ.s - current_op.s)
        new_λ = current_op.λ + α * (Δ.λ - current_op.λ)
        new_η = current_op.η + α * (Δ.η - current_op.η)
        new_ψ = current_op.ψ + α * (Δ.ψ - current_op.ψ)
        new_μ = current_op.μ + α * (Δ.μ - current_op.μ)
        α_γ = α
        new_γ = current_op.γ + α_γ * (Δ.γ - current_op.γ) # the Z variable in Nocedal's book
        next_op = trajectory(
            x = new_x,
            u = new_u,
            λ = new_λ,
            η = new_η,
            ψ = new_ψ,
            μ = new_μ,
            γ = new_γ,
            s = new_s
        )
        # @infiltrate
        # lq_approximation!(next_lq_approx, nonlinear_game, next_op) # update lq approximation? # double check this only difference!

        new_loss, new_min_s, new_min_γ, new_homotopy_loss, new_min_ineq = nw_pdip_KKT_residual_fbst(
        π,
        next_lq_approx,
        nonlinear_game,
        next_op,
        ρ
        )
        # @infiltrate
        if new_loss < loss && new_min_s > 0 && new_min_γ > 0 # && new_min_ineq > 0
            line_search_success = true
            break
        end
        α *= β        
    end

    return α, new_loss, next_op, next_lq_approx, min_s, min_γ, new_homotopy_loss, α_γ, min_ineq
end
