function forward_simulation!(
    current_op::trajectory, 
    g
    )
    # we use current_op.u to update current_op.x, i.e., ensure that the later one 
    # satisfies the dynamics of the nonlinear game
    for t in 1:g.horizon
        current_op.x[t+1] = g.f_list[t](current_op.x[t], current_op.u[t])
    end
end

function forward_simulation!(
    current_op::trajectory,
    π::strategy, 
    g
    )
    xₜ = g.x0 
    # we use current_op.u to update current_op.x, i.e., ensure that the later one 
    # satisfies the dynamics of the nonlinear game
    for t in 1:g.horizon
        # update control:
        current_op.u[t] = current_op.u[t] - π.P[t] * (current_op.x[t] - xₜ) - π.α[t]
        # update new state:
        current_op.x[t+1] = g.f_list[t](current_op.x[t], current_op.u[t])
        # update reference point xₜ:
        xₜ = current_op.x[t+1]
    end
end
