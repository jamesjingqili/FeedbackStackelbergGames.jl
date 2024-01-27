function backward_pass!(
    current_op::trajectory, 
    π::strategy, 
    g::game, 
    lq_approx::LQGame
    )
    
    # backward pass, compute a new linear policy
    lq_approximatio!(lq_approx, g, current_op)
    return fbne_lq_solver!(π, lq_approx)
    
end