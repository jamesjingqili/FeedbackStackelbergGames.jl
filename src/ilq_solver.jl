function solve!(
    current_op::trajectory, 
    π::strategy,
    g::game,
    solver::iLQsolver,
    lq_approx::LQGame,
    x0::Vector
    )
    converged = false
    iter = 0
    while !converged && iter < solver.max_iter
        iter += 1
        # forward simulation, forward simulate the policy
        # forward_simulation!(current_op, g) # 8/18, no need to do foward pass
        
        # backward pass, compute a new linear policy
        backward_pass!(current_op, π, g, lq_approx)
        
        # line search, update the linear policy and the current_op
        line_search!(current_op, π, g)
        
        # check convergence
        converged = check_convergence!(current_op, π, g, solver)
        
    end
    if !converged
        println("Warning: iLQ solver did not converge!")
    end
end


