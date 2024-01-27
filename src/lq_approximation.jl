using Infiltrator
function lq_approximation!(
    lq_approx, 
    g::game, 
    current_op::trajectory
    )
    # we linearize the dynamics of the nonlinear game around the current operating point,
    # and then quadraticize the cost function of the nonlinear game around the current operating point
    # we return a LQGame object, which is a linear-quadratic game approximation of the nonlinear game

    nx = g.nx
    nu = g.nu
    for t in 1:g.horizon
        # linearize the dynamics of the nonlinear game around the current operating point
        f_jacobian = ForwardDiff.jacobian(
            g.f_for_jacobian_and_hessian_evaluation[t], 
            vcat(current_op.x[t], current_op.u[t])
        )
        f_hessian = reshape(
            ForwardDiff.jacobian(
                z -> ForwardDiff.jacobian(g.f_for_jacobian_and_hessian_evaluation[t], z), 
                vcat(current_op.x[t], current_op.u[t])
            ), 
            nx, nx+nu, nx+nu
        )
        lq_approx.A_list[t] = f_jacobian[1:g.nx, 1:g.nx]
        lq_approx.B_list[t] = f_jacobian[1:g.nx, g.nx+1:g.nx+g.nu]
        lq_approx.c_list[t] = g.f_list[t](current_op.x[t], current_op.u[t]) - current_op.x[t+1]

        for ii in 1:g.n_players
            # TODO: linerize the equality and inequality constraints around the current operating point. Done!!
            # TODO: properly define the quadraticized cost function Q matrix, S matrix, and R matrix! Done!!
            h_jacobian = ForwardDiff.jacobian(
                g.equality_constraints_list[t][ii], 
                vcat(current_op.x[t], current_op.u[t])
            )
            g_jacobian = ForwardDiff.jacobian(
                g.inequality_constraints_list[t][ii], 
                vcat(current_op.x[t], current_op.u[t])
            )
            h_hessian = reshape(
                ForwardDiff.jacobian(
                    z -> ForwardDiff.jacobian(g.equality_constraints_list[t][ii], z), 
                    vcat(current_op.x[t], current_op.u[t])
                ), 
                g.equality_constraints_size, g.nx+g.nu, g.nx+g.nu
            )
            g_hessian = reshape(
                ForwardDiff.jacobian(
                    z -> ForwardDiff.jacobian(g.inequality_constraints_list[t][ii], z), 
                    vcat(current_op.x[t], current_op.u[t])
                ),
                g.inequality_constraints_size, g.nx+g.nu, g.nx+g.nu
            )
            lq_approx.Hx_list[t][ii] = h_jacobian[1:g.equality_constraints_size, 1:g.nx]
            lq_approx.Hu_list[t][ii] = h_jacobian[1:g.equality_constraints_size, g.nx+1:g.nx+g.nu]
            lq_approx.h_list[t][ii] = g.equality_constraints_list[t][ii](vcat(current_op.x[t], current_op.u[t]))
            lq_approx.Gx_list[t][ii] = g_jacobian[1:g.inequality_constraints_size, 1:g.nx]
            lq_approx.Gu_list[t][ii] = g_jacobian[1:g.inequality_constraints_size, g.nx+1:g.nx+g.nu]
            lq_approx.g_list[t][ii] = g.inequality_constraints_list[t][ii](vcat(current_op.x[t], current_op.u[t]))

            costs_hessian = ForwardDiff.hessian(
                g.costs_for_jacobian_and_hessian_evaluation[t][ii], vcat(current_op.x[t], current_op.u[t])
            )
            lq_approx.Q_list[t][ii] = costs_hessian[1:g.nx, 1:g.nx]  +  
                sum([f_hessian[jj,1:g.nx, 1:g.nx].*current_op.λ[t][(ii-1)*nx+jj] for jj in 1:nx]) - 
                sum([h_hessian[jj,1:g.nx, 1:g.nx].*current_op.μ[t][(ii-1)*g.equality_constraints_size+jj] for jj in 1:g.equality_constraints_size]) -
                sum([g_hessian[jj,1:g.nx, 1:g.nx].*current_op.γ[t][(ii-1)*g.inequality_constraints_size+jj] for jj in 1:g.inequality_constraints_size])
            # @infiltrate
            lq_approx.S_list[t][ii] = costs_hessian[g.nx+1:g.nx+g.nu, 1:g.nx]  + 
                sum([f_hessian[jj,g.nx+1:g.nx+g.nu, 1:g.nx]*current_op.λ[t][(ii-1)*nx+jj] for jj in 1:nx]) -
                sum([h_hessian[jj,g.nx+1:g.nx+g.nu, 1:g.nx]*current_op.μ[t][(ii-1)*g.equality_constraints_size+jj] for jj in 1:g.equality_constraints_size]) -
                sum([g_hessian[jj,g.nx+1:g.nx+g.nu, 1:g.nx]*current_op.γ[t][(ii-1)*g.inequality_constraints_size+jj] for jj in 1:g.inequality_constraints_size])
            lq_approx.R_list[t][ii] = costs_hessian[g.nx+1:end, g.nx+1:end]  + 
                sum([f_hessian[jj,g.nx+1:end, g.nx+1:end]*current_op.λ[t][(ii-1)*nx+jj] for jj in 1:nx]) -
                sum([h_hessian[jj,g.nx+1:end, g.nx+1:end]*current_op.μ[t][(ii-1)*g.equality_constraints_size+jj] for jj in 1:g.equality_constraints_size]) -
                sum([g_hessian[jj,g.nx+1:end, g.nx+1:end]*current_op.γ[t][(ii-1)*g.inequality_constraints_size+jj] for jj in 1:g.inequality_constraints_size])

            costs_gradient = ForwardDiff.gradient(
                g.costs_for_jacobian_and_hessian_evaluation[t][ii],
                vcat(current_op.x[t], current_op.u[t])
            )
            lq_approx.q_list[t][ii] = costs_gradient[1:g.nx] #- 
                # current_op.λ[t][(ii-1)*nx+1:ii*nx] #- 
                # lq_approx.Gx_list[t+1][ii]'*current_op.γ[t+1][(ii-1)*g.inequality_constraints_size+1:ii*g.inequality_constraints_size] -
                # lq_approx.Hx_list[t+1][ii]'*current_op.μ[t+1][(ii-1)*g.equality_constraints_size+1:ii*g.equality_constraints_size] #+ 
                # lq_approx.A_list[t]'*current_op.λ[t+1][(ii-1)*nx+1:ii*nx]

            lq_approx.r_list[t][ii] = costs_gradient[g.nx+1:end] # + 
                # lq_approx.B_list[t]'*current_op.λ[t][(ii-1)*nx+1:ii*nx] -
                # lq_approx.Gu_list[t][ii]'*current_op.γ[t][(ii-1)*g.inequality_constraints_size+1:ii*g.inequality_constraints_size] -
                # lq_approx.Hu_list[t][ii]'*current_op.μ[t][(ii-1)*g.equality_constraints_size+1:ii*g.equality_constraints_size]
        end
    end


    # for the terminal time T+1:
    for ii in 1:g.n_players
        costs_hessian = ForwardDiff.hessian(
            g.terminal_costs_list[ii], 
            current_op.x[g.horizon+1]
        )
        costs_gradient = ForwardDiff.gradient(
            g.terminal_costs_list[ii], 
            current_op.x[g.horizon+1]
        )
        h_jacobian = ForwardDiff.jacobian(
            g.terminal_equality_constraints_list[ii], 
            current_op.x[g.horizon+1]
        )
        h_hessian = reshape(
            ForwardDiff.jacobian(
                z -> ForwardDiff.jacobian(
                    g.terminal_equality_constraints_list[ii], z), 
                current_op.x[g.horizon+1]
            ),
            g.equality_constraints_size, g.nx, g.nx
        )
        g_jacobian = ForwardDiff.jacobian(
            g.terminal_inequality_constraints_list[ii], 
            current_op.x[g.horizon+1]
        )
        g_hessian = reshape(
            ForwardDiff.jacobian(
                z -> ForwardDiff.jacobian(
                    g.terminal_inequality_constraints_list[ii], z), 
                current_op.x[g.horizon+1]
            ),
            g.inequality_constraints_size, g.nx, g.nx
        )

        lq_approx.HxT[ii] = h_jacobian[1:g.equality_constraints_size, 1:g.nx]
        lq_approx.hxT[ii] = g.terminal_equality_constraints_list[ii](current_op.x[g.horizon+1])
        lq_approx.GxT[ii] = g_jacobian[1:g.inequality_constraints_size, 1:g.nx]
        lq_approx.gxT[ii] = g.terminal_inequality_constraints_list[ii](current_op.x[g.horizon+1])
        lq_approx.Q_list[g.horizon+1][ii] = costs_hessian[1:g.nx, 1:g.nx]  -  
            sum([h_hessian[jj,:,:]'*current_op.μ[g.horizon+1][(ii-1)*g.equality_constraints_size+jj] for jj in 1:g.equality_constraints_size]) - 
            sum([g_hessian[jj,:,:]'*current_op.γ[g.horizon+1][(ii-1)*g.inequality_constraints_size+jj] for jj in 1:g.inequality_constraints_size])
        lq_approx.q_list[g.horizon+1][ii] = costs_gradient[1:g.nx] #- 
            # current_op.λ[g.horizon][(ii-1)*nx+1:ii*nx] #- 
            # lq_approx.HxT[ii]'*current_op.μ[g.horizon+1][(ii-1)*g.equality_constraints_size+1:ii*g.equality_constraints_size] -
            # lq_approx.GxT[ii]'*current_op.γ[g.horizon+1][(ii-1)*g.inequality_constraints_size+1:ii*g.inequality_constraints_size] 
    end
end


