using Infiltrator
# # We only have two players 
# # first player is leader and second player is follower
# # TODO: nonzero S matrix. R̃ₜ is not used in the code
# function fbst_lq_solver!(
#     strategies, 
#     g::LQGame
#     )

#     x0 = g.x0
#     # extract state, total control, and each player's control dimensions
#     nx, nu, m = g.nx, g.nu, length(g.players_u_index_list[1])
#     T = g.horizon
#     # m is the input size of agent i, and T is the horizon.
#     num_player = g.n_players # number of player
#     @assert length(num_player) != 2
#     M_size = nu+nx*num_player+m+nx# + (num_player-1)*nu 
#     # size of the M matrix for each time instant, will be used to define KKT matrix
#     new_M_size = nu+nx+nx*num_player+m + (num_player-1)*nu

#     # initialize some intermidiate variables in KKT conditions    
#     Mₜ,     Nₜ,      nₜ     = zeros(M_size, M_size), zeros(M_size, nx), zeros(M_size)
#     λ = zeros(T*nx*num_player)
#     η = zeros((T)*nu)
#     ψ = zeros(T*m)
    
#     record_old_Mₜ_size = M_size
#     K, k = zeros(M_size, nx), zeros(M_size)
#     Π² = zeros(m, nu)
#     π̂², π̌² = zeros(m, nx), zeros(m, m) # π²'s dependence on x and u1
    
#     Π_next = zeros(nu, nx*num_player)
#     π¹_next = zeros(m, nx)

#     Aₜ₊₁ = zeros(nx, nx)
#     Bₜ₊₁ = zeros(nx, nu)
#     K_lambda_next = zeros(2*nx, nx)
#     k_lambda_next = zeros(2*nx)
#     K_multipliers = zeros((T-1)*(2*nx+nu+m) + (2*nx+m), nx)
#     k_multipliers = zeros((T-1)*(2*nx+nu+m) + (2*nx+m), 1)
#     for t in T:-1:1 # work in backwards to construct the KKT constraint matrix
#         # convenience shorthands for the relevant quantities
#         A, B, c = g.A_list[t], g.B_list[t], g.c_list[t]
#         Âₜ₊₁, B̂ₜ = zeros(nx*num_player, nx*num_player), zeros(nx*num_player, nu)
#         Rₜ, Qₜ₊₁ = zeros(nu, nu), zeros(nx*num_player, nx)
#         rₜ, qₜ₊₁ = zeros(nu), zeros(nx*num_player)
#         B̃ₜ₊₁ = zeros(num_player*nx, (num_player-1)*nu) 
#         # R̃ₜ = zeros((num_player-1)*nu, nu) 
#         B̃² = [B[:,m+1:nu]; zeros(nx, m)]

#         if t == T
#             # We first solve for the follower, player 2
#             for (ii, udxᵢ) in enumerate(g.players_u_index_list)
#                 B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
#                 Qₜ₊₁[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = g.Q_list[t][ii], g.R_list[t][ii][udxᵢ,:]
#                 qₜ₊₁[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = g.q_list[t][ii], g.r_list[t][ii][udxᵢ]
#             end
#             N2 = zeros(m+nx+nx, nx+m)
#             M2 = zeros(m+nx+nx, m+nx+nx)
            
#             N2[m+1:m+nx, :] = [-A -B[:,1:m]]

#             M2[1:m, 1:m] = g.R_list[t][2][m+1:2*m, m+1:2*m]
#             M2[1:m, m+1:m+nx] = transpose(B[:,m+1:nu]) # B̂ₜ
#             M2[m+1:m+nx, 1:m] = -B[:,m+1:nu]
#             M2[m+1:m+nx, m+nx+1:m+nx+nx] = I(nx)
#             M2[m+nx+1:m+nx+nx, m+1:m+nx] = -I(nx)
#             M2[m+nx+1:m+nx+nx, m+nx+1:m+nx+nx] = g.Q_list[t][2]
#             inv_M2_N2 = -M2\N2
#             π̂², π̌² = inv_M2_N2[1:m, 1:nx], inv_M2_N2[1:m, nx+1:nx+m]
            
#             n2 = zeros(m+nx+nx)
#             n2[1:m] = g.r_list[t][2][m+1:2*m]
#             n2[m+1:m+nx] = -c
#             n2[m+nx+1:m+nx+nx] = g.q_list[t][2]
#             inv_M2_n2 = -M2\n2
#             βₜ² = inv_M2_n2[1:m] # offset term in the follower's strategy

#             Π²[:,1:m] = π̌²

#             # we then solve for the leader, player 1
#             Nₜ[nu+1:nu+nx,:] = -A

#             Mₜ[1:nu, 1:nu] = Rₜ
#             Mₜ[1:nu, nu+1:nu+nx*num_player] = transpose(B̂ₜ)
#             Mₜ[nu+1:nu+nx, 1:nu] = -B
#             Mₜ[nu+1:nu+nx, M_size-nx+1:M_size] = I(nx)
#             Mₜ[nu+nx+1:M_size-m, nu+1:nu+nx*num_player] = -I(nx*num_player)
#             Mₜ[nu+nx+1:M_size-m, M_size-nx+1:M_size] = Qₜ₊₁
#             Mₜ[M_size-m+1:M_size, nu+1:nu+2*nx] = B̃²'
#             Mₜ[M_size-m+1:M_size, nu+2*nx+1:nu+2*nx+m] = -I(m)

#             Mₜ[1:nu, nu+2*nx+1:nu+2*nx+m] = Π²'
#             nₜ[1:nu], nₜ[M_size-nx*num_player+1-m:M_size-m] = rₜ, qₜ₊₁
            
            
#             # M_next, N_next, n_next = Mₜ, Nₜ, nₜ
#             K, k = -Mₜ\Nₜ, -Mₜ\nₜ
#             π¹_next = K[1:m,:]
#             Π_next[1:m,1:nx] = π̂² # πₜ₊₁
#             Π_next[m+1:nu, nx+1:2*nx] = π¹_next
            
#             strategies.P[t] = -K[1:nu, :]
#             strategies.α[t] = -k[1:nu]
            
#             Aₜ₊₁ = A
#             Bₜ₊₁ = B
#             K_lambda_next = -K[nu+1:nu+nx+nx, :]
#             k_lambda_next = -k[nu+1:nu+nx+nx, :]
#             K_multipliers[(t-1)*(2*nx+nu+m)+1:end, :] = K[nu+1:nu+2*nx+m, :]
#             k_multipliers[(t-1)*(2*nx+nu+m)+1:end] = k[nu+1:nu+2*nx+m, :]
#             @infiltrate
#         else
#             # when t < T, we first solve for the follower
#             for (ii, udxᵢ) in enumerate(g.players_u_index_list)
#                 Âₜ₊₁[(ii-1)*nx+1:ii*nx, (ii-1)*nx+1:ii*nx] = Aₜ₊₁
#                 B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
#                 Qₜ₊₁[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = g.Q_list[t][ii], g.R_list[t][ii][udxᵢ,:]
#                 qₜ₊₁[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = g.q_list[t][ii], g.r_list[t][ii][udxᵢ]
#                 udxᵢ_complement = setdiff(1:1:nu, udxᵢ)
                
#                 B̃ₜ₊₁[(ii-1)*nx+1:ii*nx, (ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m] = Bₜ₊₁[:,udxᵢ_complement] # 
#                 # R̃ₜ[(ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m, :] = cost[ii].R[udxᵢ_complement,:] # not used actually 
#             end
#             # below is the construction of the KKT matrix for follower
#             # we don't need to compute the offset term here
#             tmp_Pₜ¹ = inv(
#                 [g.R_list[t][2][m+1:2*m, m+1:2*m]  transpose(B[:,m+1:nu])  zeros(m,m)  zeros(m,nx);
#                 -B[:,m+1:nu]  zeros(nx, nx+m)  I(nx);
#                 zeros(nx, m)  -I(nx)  π¹_next'  g.Q_list[t][2]-Aₜ₊₁'*K_lambda_next[nx+1:end,:];
#                 zeros(m, m+nx)  -I(m)  -Bₜ₊₁[:,1:m]'*K_lambda_next[nx+1:end,:]]
#             )
            
#             Nₜ² = zeros(size(tmp_Pₜ¹,1), nx+m)
#             Nₜ²[m+1:m+nx, :] = [-A -B[:,1:m]]
#             inv_tmp_Pₜ¹_Nₜ² = -tmp_Pₜ¹*Nₜ²
#             π̂², π̌² = inv_tmp_Pₜ¹_Nₜ²[1:m, 1:nx], inv_tmp_Pₜ¹_Nₜ²[1:m, nx+1:nx+m]
#             Π²[:,1:m] = π̌²
#             # @infiltrate
#             # we then solve the leader
            
#             # Nₜ = zeros(new_M_size, nx)
#             # Nₜ[nu+1:nu+nx,:] = -A # Nₜ is defined here!
#             # @infiltrate
#             Pₜ¹ = inv([Rₜ  B̂ₜ'  zeros(nu, nu)  Π²'  zeros(nu, nx);
#                 -B  zeros(nx, 2*nx+nu+m)  I(nx);
#                 zeros(2*nx, nu)  -I(2*nx)  Π_next'  zeros(2*nx,m)   Qₜ₊₁-Âₜ₊₁'*K_lambda_next;
#                 zeros(m, nu)  B̃²'  zeros(m, nu)  -I(m)  zeros(m, nx);
#                 zeros(nu,nu)  zeros(nu, 2*nx)  -I(nu)  zeros(nu, m)   -B̃ₜ₊₁'*K_lambda_next ])
#             Pₜ²nₜ₊₁ = -Pₜ¹* [zeros(nu+nx, 1); Âₜ₊₁'*k_lambda_next; zeros(m,1); B̃ₜ₊₁'*k_lambda_next ]
            
#             K, k = -Pₜ¹*[zeros(nu,nx);-A;zeros(new_M_size-nu-nx,nx)], -Pₜ¹*[rₜ; -c; qₜ₊₁; zeros(m+nu,1)] - Pₜ²nₜ₊₁
#             π¹_next = K[1:m,:] # update π¹_next
#             Π_next[1:m,1:nx] = π̂²
#             Π_next[m+1:nu, nx+1:2*nx] = π¹_next
            
#             strategies.P[t] = -K[1:nu, :]
#             strategies.α[t] = -k[1:nu]
            
#             Aₜ₊₁ = A
#             Bₜ₊₁ = B
#             K_lambda_next = -K[nu+1:nu+nx+nx, :]
#             k_lambda_next = -k[nu+1:nu+nx+nx, :]
#             K_multipliers[(t-1)*(2*nx+nu+m)+1:t*(2*nx+nu+m), :] = K[nu+1:nu+2*nx+nu+m, :]
#             k_multipliers[(t-1)*(2*nx+nu+m)+1:t*(2*nx+nu+m)] = k[nu+1:nu+2*nx+nu+m, :]
            
#         end
#     end
#     # solution = K*x0+k
#     x = [x0 for t in 1:T+1]
#     u = [zeros(nu) for t in 1:T]
#     λ = [zeros(nx*num_player) for t in 1:T]
#     η = [zeros(nu) for t in 1:T-1]
#     ψ = [zeros(m) for t in 1:T]
#     for t in 1:1:T
#         if t == T
#             λ[t] = K_multipliers[(t-1)*(2*nx+nu+m)+1:end-m, :]*x[t] + k_multipliers[(t-1)*(2*nx+nu+m)+1:end-m]
#             ψ[t] = K_multipliers[end-m+1:end, :]*x[t] + k_multipliers[end-m+1:end]
#             u[t] = -strategies.P[t]*x[t] - strategies.α[t]
#             x[t+1] = g.A_list[t]*x[t] + g.B_list[t]*u[t] + g.c_list[t] # update x
#         else
#             λ[t] = vec(K_multipliers[ (t-1)*(2*nx+nu+m)+1:(t-1)*(2*nx+nu+m)+2*nx, : ]*x[t] + k_multipliers[(t-1)*(2*nx+nu+m)+1:(t-1)*(2*nx+nu+m)+2*nx, : ])
#             η[t] = vec(K_multipliers[(t-1)*(2*nx+nu+m)+2*nx+1:(t-1)*(2*nx+nu+m)+2*nx+nu, :]*x[t] + k_multipliers[ (t-1)*(2*nx+nu+m)+2*nx+1:(t-1)*(2*nx+nu+m)+2*nx+nu, : ])
#             ψ[t] = vec(K_multipliers[(t-1)*(2*nx+nu+m)+2*nx+nu+1:t*(2*nx+nu+m), : ]*x[t] + k_multipliers[(t-1)*(2*nx+nu+m)+2*nx+nu+1:t*(2*nx+nu+m)])
#             u[t] = -strategies.P[t]*x[t] - strategies.α[t]
#             x[t+1] = g.A_list[t]*x[t] + g.B_list[t]*u[t] + g.c_list[t] # update x
#         end
#     end
#     # @infiltrate
#     return x, u, λ, η, ψ
# end








function fbst_lq_solver!(
    strategies, 
    g
    )

    x0 = g.x0
    # extract state, total control, and each player's control dimensions
    nx, nu, m = g.nx, g.nu, length(g.players_u_index_list[1])
    T = g.horizon
    # m is the input size of agent i, and T is the horizon.
    num_player = g.n_players # number of player
    @assert length(num_player) != 2
    M_size = nu+nx*num_player+m+nx# + (num_player-1)*nu 
    # size of the M matrix for each time instant, will be used to define KKT matrix
    new_M_size = nu+nx+nx*num_player+m + (num_player-1)*nu

    # initialize some intermidiate variables in KKT conditions    
    Mₜ = BlockArray(zeros(M_size, M_size), [nu, nx, nx*num_player, m], [nu, nx*num_player, m, nx] )
    Nₜ = BlockArray(zeros(M_size, nx), [nu, nx, nx*num_player, m], [nx])
    nₜ = BlockArray(zeros(M_size), [nu, nx, nx*num_player, m])
    λ = [zeros(nx*num_player) for t in 1:T]
    η = [zeros(nu) for t in 1:T-1]
    ψ = [zeros(m) for t in 1:T]
    

    record_old_Mₜ_size = M_size
    K = BlockArray(zeros(M_size, nx), [nu, nx, nx*num_player, m], [nx])
    k = BlockArray(zeros(M_size, 1), [nu, nx, nx*num_player, m], [1])
    Π² = zeros(m, nu)
    π̂², π̌² = zeros(m, nx), zeros(m, m) # π²'s dependence on x and u1
    
    Π_next = BlockArray(zeros(nu, nx*num_player), [nu], [nx for ii in 1:num_player])
    π¹_next = zeros(m, nx)

    Aₜ₊₁ = zeros(nx, nx)
    Bₜ₊₁ = zeros(nx, nu)
    K_lambda_next = zeros(2*nx, nx)
    k_lambda_next = zeros(2*nx)
    K_multipliers = zeros((T-1)*(2*nx+nu+m) + (2*nx+m), nx)
    k_multipliers = zeros((T-1)*(2*nx+nu+m) + (2*nx+m), 1)
    for t in T:-1:1 # work in backwards to construct the KKT constraint matrix
        # convenience shorthands for the relevant quantities
        A, B, c = g.A_list[t], g.B_list[t], g.c_list[t]
        Âₜ₊₁, B̂ₜ = zeros(nx*num_player, nx*num_player), zeros(nx*num_player, nu)
        Rₜ, Qₜ₊₁ = zeros(nu, nu), zeros(nx*num_player, nx)
        rₜ, qₜ₊₁ = zeros(nu), zeros(nx*num_player)
        B̃ₜ₊₁ = zeros(num_player*nx, (num_player-1)*nu) 
        # R̃ₜ = zeros((num_player-1)*nu, nu) 
        B̃² = [B[:,m+1:nu]; zeros(nx, m)]

        if t == T
            # We first solve for the follower, player 2
            for (ii, udxᵢ) in enumerate(g.players_u_index_list)
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                Qₜ₊₁[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = g.Q_list[t+1][ii], g.R_list[t][ii][udxᵢ,:]
                qₜ₊₁[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = g.q_list[t+1][ii], g.r_list[t][ii][udxᵢ]
            end
            N2 = BlockArray(zeros(m+nx+nx, nx+m), [m, nx, nx], [nx+m])
            M2 = BlockArray(zeros(m+nx+nx, m+nx+nx), [m, nx, nx], [m, nx, nx])
            
            N2[Block(2,1)] = [-A -B[:,1:m]]

            M2[Block(1,1)] = g.R_list[t][2][m+1:2*m, m+1:2*m]
            M2[Block(1,2)] = transpose(B[:,m+1:nu]) # B̂ₜ
            M2[Block(2,1)] = -B[:,m+1:nu]
            M2[Block(2,3)] = I(nx)
            M2[Block(3,2)] = -I(nx)
            M2[Block(3,3)] = g.Q_list[t+1][2]
            inv_M2_N2 = -Array(M2)\N2
            π̂², π̌² = inv_M2_N2[1:m, 1:nx], inv_M2_N2[1:m, nx+1:nx+m]
            
            n2 = BlockArray(zeros(m+nx+nx), [m, nx, nx])
            # @infiltrate
            n2[Block(1,1)] = g.r_list[t][2][m+1:2*m]
            n2[Block(2,1)] = -c
            n2[Block(3,1)] = g.q_list[t+1][2]
            inv_M2_n2 = -Array(M2)\n2
            # βₜ² = inv_M2_n2[1:m] # offset term in the follower's strategy

            Π²[:,1:m] = π̌²

            # we then solve for the leader, player 1
            Nₜ[nu+1:nu+nx,:] = -A

            Mₜ[Block(1,1)] = Rₜ
            Mₜ[Block(1,2)] = transpose(B̂ₜ)
            Mₜ[Block(1,3)] = Π²'
            Mₜ[Block(2,1)] = -B
            Mₜ[Block(2,4)] = I(nx)
            Mₜ[Block(3,2)] = -I(nx*num_player)
            Mₜ[Block(3,4)] = Qₜ₊₁
            Mₜ[Block(4,2)] = B̃²'
            Mₜ[Block(4,3)] = -I(m)

            nₜ[Block(1,1)], nₜ[Block(2,1)], nₜ[Block(3,1)] = rₜ, -c, qₜ₊₁
            
            
            # M_next, N_next, n_next = Mₜ, Nₜ, nₜ
            K, k = -Array(Mₜ)\Nₜ, -Array(Mₜ)\nₜ
            π¹_next = K[1:m,:]
            Π_next[1:m,1:nx] = π̂² # πₜ₊₁
            Π_next[m+1:nu, nx+1:2*nx] = π¹_next
            
            strategies.P[t] = -K[1:nu, :]
            strategies.α[t] = -k[1:nu]
            
            Aₜ₊₁ = A
            Bₜ₊₁ = B
            K_lambda_next = -K[nu+1:nu+nx+nx, :]
            k_lambda_next = -k[nu+1:nu+nx+nx, :]
            K_multipliers[(t-1)*(2*nx+nu+m)+1:end, :] = K[nu+1:nu+2*nx+m, :]
            k_multipliers[(t-1)*(2*nx+nu+m)+1:end] = k[nu+1:nu+2*nx+m, :]
            # @infiltrate
        else
            # when t < T, we first solve for the follower
            for (ii, udxᵢ) in enumerate(g.players_u_index_list)
                Âₜ₊₁[(ii-1)*nx+1:ii*nx, (ii-1)*nx+1:ii*nx] = Aₜ₊₁
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                Qₜ₊₁[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = g.Q_list[t+1][ii], g.R_list[t][ii][udxᵢ,:]
                qₜ₊₁[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = g.q_list[t+1][ii], g.r_list[t][ii][udxᵢ]
                udxᵢ_complement = setdiff(1:1:nu, udxᵢ)
                
                B̃ₜ₊₁[(ii-1)*nx+1:ii*nx, (ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m] = Bₜ₊₁[:,udxᵢ_complement] # 
                # R̃ₜ[(ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m, :] = cost[ii].R[udxᵢ_complement,:] # not used actually 
            end
            # below is the construction of the KKT matrix for follower
            # we don't need to compute the offset term here
            follower_Pₜ¹ = BlockArray(zeros(m+nx+nx+m, m+nx+m+nx), [m, nx, nx, m], [m, nx, m, nx])
            follower_Pₜ¹[Block(1,1)] = g.R_list[t][2][m+1:2*m, m+1:2*m]
            follower_Pₜ¹[Block(1,2)] = transpose(B[:,m+1:nu])
            follower_Pₜ¹[Block(2,1)] = -B[:,m+1:nu]
            follower_Pₜ¹[Block(2,4)] = I(nx)
            follower_Pₜ¹[Block(3,2)] = -I(nx)
            follower_Pₜ¹[Block(3,3)] = π¹_next'
            follower_Pₜ¹[Block(3,4)] = g.Q_list[t+1][2]-Aₜ₊₁'*K_lambda_next[nx+1:end,:]
            follower_Pₜ¹[Block(4,3)] = -I(m)
            follower_Pₜ¹[Block(4,4)] = -Bₜ₊₁[:,1:m]'*K_lambda_next[nx+1:end,:]
            tmp_Pₜ¹ = inv(follower_Pₜ¹)
            # tmp_Pₜ¹ = inv(
            #     [g.R_list[t][2][m+1:2*m, m+1:2*m]  transpose(B[:,m+1:nu])  zeros(m,m)  zeros(m,nx);
            #     -B[:,m+1:nu]  zeros(nx, nx+m)  I(nx);
            #     zeros(nx, m)  -I(nx)  π¹_next'  g.Q_list[t][2]-Aₜ₊₁'*K_lambda_next[nx+1:end,:];
            #     zeros(m, m+nx)  -I(m)  -Bₜ₊₁[:,1:m]'*K_lambda_next[nx+1:end,:]]
            # )
            
            Nₜ² = BlockArray(zeros(size(tmp_Pₜ¹,1), nx+m), [m, nx, nx, m], [nx+m])
            Nₜ²[Block(2,1)] = [-A -B[:,1:m]]
            inv_tmp_Pₜ¹_Nₜ² = -tmp_Pₜ¹*Nₜ²
            π̂², π̌² = inv_tmp_Pₜ¹_Nₜ²[1:m, 1:nx], inv_tmp_Pₜ¹_Nₜ²[1:m, nx+1:nx+m]
            Π²[:,1:m] = π̌²
            # @infiltrate
            # we then solve the leader
            
            # Nₜ = zeros(new_M_size, nx)
            # Nₜ[nu+1:nu+nx,:] = -A # Nₜ is defined here!
            # @infiltrate
            leader_Pₜ¹ = BlockArray(zeros(new_M_size, new_M_size), [nu, nx, nx*num_player, m, nu], [nu, nx*num_player, nu, m, nx])
            leader_Pₜ¹[Block(1,1)] = Rₜ
            leader_Pₜ¹[Block(1,2)] = transpose(B̂ₜ)
            leader_Pₜ¹[Block(1,4)] = Π²'
            leader_Pₜ¹[Block(2,1)] = -B
            leader_Pₜ¹[Block(2,5)] = I(nx)
            leader_Pₜ¹[Block(3,2)] = -I(nx*num_player)
            leader_Pₜ¹[Block(3,3)] = Π_next'
            leader_Pₜ¹[Block(3,5)] = Qₜ₊₁-Âₜ₊₁'*K_lambda_next
            leader_Pₜ¹[Block(4,2)] = B̃²'
            leader_Pₜ¹[Block(4,4)] = -I(m)
            leader_Pₜ¹[Block(5,3)] = -I(nu)
            leader_Pₜ¹[Block(5,5)] = -B̃ₜ₊₁'*K_lambda_next

            Pₜ¹ = inv(leader_Pₜ¹)
            # Pₜ¹ = inv([Rₜ  B̂ₜ'  zeros(nu, nu)  Π²'  zeros(nu, nx);
            #     -B  zeros(nx, 2*nx+nu+m)  I(nx);
            #     zeros(2*nx, nu)  -I(2*nx)  Π_next'  zeros(2*nx,m)   Qₜ₊₁-Âₜ₊₁'*K_lambda_next;
            #     zeros(m, nu)  B̃²'  zeros(m, nu)  -I(m)  zeros(m, nx);
            #     zeros(nu,nu)  zeros(nu, 2*nx)  -I(nu)  zeros(nu, m)   -B̃ₜ₊₁'*K_lambda_next ])
            Pₜ²nₜ₊₁ = -Pₜ¹* [zeros(nu+nx, 1); Âₜ₊₁'*k_lambda_next; zeros(m,1); B̃ₜ₊₁'*k_lambda_next ]
            
            K, k = -Pₜ¹*[zeros(nu,nx);-A;zeros(new_M_size-nu-nx,nx)], -Pₜ¹*[rₜ; -c; qₜ₊₁; zeros(m+nu,1)] - Pₜ²nₜ₊₁
            π¹_next = K[1:m,:] # update π¹_next
            Π_next[1:m,1:nx] = π̂²
            Π_next[m+1:nu, nx+1:2*nx] = π¹_next
            
            strategies.P[t] = -K[1:nu, :]
            strategies.α[t] = -k[1:nu]
            
            Aₜ₊₁ = A
            Bₜ₊₁ = B
            K_lambda_next = -K[nu+1:nu+nx+nx, :]
            k_lambda_next = -k[nu+1:nu+nx+nx, :]
            K_multipliers[(t-1)*(2*nx+nu+m)+1:t*(2*nx+nu+m), :] = K[nu+1:nu+2*nx+nu+m, :]
            k_multipliers[(t-1)*(2*nx+nu+m)+1:t*(2*nx+nu+m)] = k[nu+1:nu+2*nx+nu+m, :]
            
        end
    end
    # solution = K*x0+k
    x = [x0 for t in 1:T+1]
    u = [zeros(nu) for t in 1:T]
    
    for t in 1:1:T
        if t == T
            λ[t] = K_multipliers[(t-1)*(2*nx+nu+m)+1:end-m, :]*x[t] + k_multipliers[(t-1)*(2*nx+nu+m)+1:end-m]
            ψ[t] = K_multipliers[end-m+1:end, :]*x[t] + k_multipliers[end-m+1:end]
            u[t] = -strategies.P[t]*x[t] - strategies.α[t]
            x[t+1] = g.A_list[t]*x[t] + g.B_list[t]*u[t] + g.c_list[t] # update x
        else
            λ[t] = vec(K_multipliers[ (t-1)*(2*nx+nu+m)+1:(t-1)*(2*nx+nu+m)+2*nx, : ]*x[t] + k_multipliers[(t-1)*(2*nx+nu+m)+1:(t-1)*(2*nx+nu+m)+2*nx, : ])
            η[t] = vec(K_multipliers[(t-1)*(2*nx+nu+m)+2*nx+1:(t-1)*(2*nx+nu+m)+2*nx+nu, :]*x[t] + k_multipliers[ (t-1)*(2*nx+nu+m)+2*nx+1:(t-1)*(2*nx+nu+m)+2*nx+nu, : ])
            ψ[t] = vec(K_multipliers[(t-1)*(2*nx+nu+m)+2*nx+nu+1:t*(2*nx+nu+m), : ]*x[t] + k_multipliers[(t-1)*(2*nx+nu+m)+2*nx+nu+1:t*(2*nx+nu+m)])
            u[t] = -strategies.P[t]*x[t] - strategies.α[t]
            x[t+1] = g.A_list[t]*x[t] + g.B_list[t]*u[t] + g.c_list[t] # update x
        end
    end
    # @infiltrate
    return x, u, λ, η, ψ
end