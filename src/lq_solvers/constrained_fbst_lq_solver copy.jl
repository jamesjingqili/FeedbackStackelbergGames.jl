using Infiltrator
function constrained_fbst_lq_solver!(
    strategies, 
    g::Constrained_LQGame, # this is the lq approximation of the original game
    output_only_KKT_matrix_for_KKT_residual_loss = false,
    simulate_the_trajectory_in_global_coordinate = false,
    output_ground_truth_KKT_matrices = false
    )
    l = g.equality_constraints_size
    
    x0 = g.x0
    # extract state, total control, and each player's control dimensions
    nx, nu, m = g.nx, g.nu, length(g.players_u_index_list[1])
    T = g.horizon
    # m is the input size of agent i, and T is the horizon.
    num_player = g.n_players # number of player
    @assert length(num_player) != 2
    M_size = nu + l*num_player + nx*num_player + m + nx + l*num_player # + (num_player-1)*nu 
    # size of the M matrix for each time instant, will be used to define KKT matrix
    new_M_size = nu + l*num_player + nx + nx*num_player + m + (num_player-1)*nu

    # initialize some intermidiate variables in KKT conditions    
    Mₜ = BlockArray(zeros(M_size, M_size), 
        [nu, l*num_player, nx, nx*num_player, l*num_player, m], 
        [nu, l*num_player, nx*num_player, m, nx, l*num_player] )
    Nₜ = BlockArray(zeros(M_size, nx),  [nu, l*num_player, nx, nx*num_player, l*num_player, m], [nx])
    nₜ = BlockArray(zeros(M_size),      [nu, l*num_player, nx, nx*num_player, l*num_player, m])
    μ = [zeros(l*num_player) for t in 1:T+1]
    λ = [zeros(nx*num_player) for t in 1:T]
    η = [zeros(nu) for t in 1:T-1]
    ψ = [zeros(m) for t in 1:T]
    
    hₜ = BlockArray(zeros(l*num_player), [l, l]) # (2)
    H_uₜ² = BlockArray(zeros(l*num_player, m), [l, l], [m]) # (2,1)
    H_uₜ¹ = BlockArray(zeros(l*num_player, m), [l, l], [m])# (2,1)

    H_xₜ = BlockArray(zeros(l*num_player, nx), [l, l], [nx]) # (2,1)
    H_uₜ = BlockArray(zeros(l*num_player, nu), [l, l], [nu]) # (2,1)
    H_x_terminal = BlockArray(zeros(l*num_player, nx), [l, l], [nx]) # (2,1)
    h_x_terminal = BlockArray(zeros(l*num_player), [l, l]) # (2)

    Ĥ_uₜ = BlockArray(zeros(l*num_player, m*num_player), [l, l], [m, m]) # (2,2)
    Ĥ_xₜ₊₁ = BlockArray(zeros(l*num_player, nx*num_player), 
        [l, l], [nx, nx]) # (2,2)
    
    H̃_uₜ₊₁ = BlockArray(zeros(l*num_player, m*num_player), [l, l], [m, m]) # (2,2)

    # K = BlockArray(zeros(M_size, nx), [nu, 2*l, nx*num_player, nx, 2*l, m], [nx]) # (4,1)
    # k = BlockArray(zeros(M_size, 1), [nu, 2*l, nx*num_player, nx, 2*l, m], [1]) # (4,1)
    Π² = zeros(m, nu) 
    π̂², π̌² = zeros(m, nx), zeros(m, m) # π²'s dependence on x and u1
    
    Π_next = BlockArray(zeros(nu, nx*num_player), [nu], [nx for ii in 1:num_player]) # (1,2)
    π¹_next = zeros(m, nx) 

    Aₜ₊₁ = zeros(nx, nx)
    Bₜ₊₁ = zeros(nx, nu)
    K_next = zeros(nu, nx)
    k_next = zeros(nu)
    K_lambda_next = zeros(2*nx, nx)
    k_lambda_next = zeros(2*nx)
    K_mu_next = zeros(2*l, nx)
    k_mu_next = zeros(2*l)

    K_multipliers = BlockArray(zeros((T-1)*(l*num_player + 2*nx + nu + m) + (l*num_player + 2*nx + m + 2*l), nx), 
        vcat([2*l + 2*nx + nu + m for t in 1:T-1], [2*l + 2*nx + m + 2*l]),     [nx])
    k_multipliers = BlockArray(zeros((T-1)*(l*num_player + 2*nx + nu + m) + (l*num_player + 2*nx + m + 2*l), 1), 
        vcat([2*l + 2*nx + nu + m for t in 1:T-1], [2*l + 2*nx + m + 2*l]),     [1])
    
    KKT_M = deepcopy(Mₜ)
    KKT_N = deepcopy(Nₜ)
    KKT_n = deepcopy(nₜ)

    for t in T:-1:1 # work in backwards to construct the KKT constraint matrix
        # convenience shorthands for the relevant quantities
        A, B, c = g.A_list[t], g.B_list[t], g.c_list[t]
        Âₜ₊₁, B̂ₜ = zeros(nx*num_player, nx*num_player), zeros(nx*num_player, nu)
        Rₜ, Qₜ₊₁ = zeros(nu, nu), zeros(nx*num_player, nx)
        rₜ, qₜ₊₁ = zeros(nu), zeros(nx*num_player)
        
        B̃ₜ₊₁ = zeros(num_player*nx, (num_player-1)*nu) 
        # R̃ₜ = zeros((num_player-1)*nu, nu) 
        B̃² = [B[:,m+1:nu]; zeros(nx, m)]
        Ĥ_x_terminal = BlockArray(zeros(2*l, 2*nx), [l,l], [nx,nx])

        H_uₜ²[Block(1,1)] = g.Hu_list[t][1][:,m+1:nu]
        H_uₜ²[Block(2,1)] = g.Hu_list[t][2][:,m+1:nu]
        H_uₜ¹[Block(1,1)] = g.Hu_list[t][1][:,1:m]
        H_uₜ¹[Block(2,1)] = g.Hu_list[t][2][:,1:m]

        H_xₜ[Block(1,1)] = g.Hx_list[t][1]
        H_xₜ[Block(2,1)] = g.Hx_list[t][2]
        H_uₜ[Block(1,1)] = g.Hu_list[t][1]
        H_uₜ[Block(2,1)] = g.Hu_list[t][2]

        Ĥ_uₜ[Block(1,1)] = g.Hu_list[t][1][:,1:m]
        Ĥ_uₜ[Block(2,2)] = g.Hu_list[t][2][:,m+1:nu]

        H_x_terminal[Block(1,1)] = g.HxT[1]
        H_x_terminal[Block(2,1)] = g.HxT[2]
        Ĥ_x_terminal[Block(1,1)] = g.HxT[1]
        Ĥ_x_terminal[Block(2,2)] = g.HxT[2]
        h_x_terminal[Block(1)]= g.hxT[1]
        h_x_terminal[Block(2)]= g.hxT[2]
        hₜ[Block(1)]= g.h_list[t][1]
        hₜ[Block(2)]= g.h_list[t][2]


        if t == T
            # We first solve for the follower, player 2
            
            for (ii, udxᵢ) in enumerate(g.players_u_index_list)
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                Qₜ₊₁[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = g.Q_list[t+1][ii], g.R_list[t][ii][udxᵢ,:]
                qₜ₊₁[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = g.q_list[t+1][ii], g.r_list[t][ii][udxᵢ]
            end
            N2 = BlockArray(zeros(m+l+nx+nx+l, nx+m), 
                [m, l, nx, nx, l], [nx+m])
            M2 = BlockArray(zeros(m+l+nx+nx+l, m+l+nx+nx+l), 
                [m, l, nx, nx, l], [m, l, nx, nx, l])

            N2[Block(1,1)] = [g.S_list[t][2][m+1:nu,:] g.R_list[t][2][m+1:nu,1:m]] # double check
            N2[Block(2,1)] = [g.Hx_list[t][2] g.Hu_list[t][2][:,1:m]]
            N2[Block(3,1)] = [-A -B[:,1:m]]

            M2[Block(1,1)] = g.R_list[t][2][m+1:2*m, m+1:2*m]
            M2[Block(1,2)] = -g.Hu_list[t][2][:,m+1:nu]'
            M2[Block(1,3)] = transpose(B[:,m+1:nu]) # B̂ₜ
            M2[Block(2,1)] = g.Hu_list[t][2][:,m+1:nu]
            M2[Block(3,1)] = -B[:,m+1:nu]
            M2[Block(3,4)] = I(nx)
            M2[Block(4,3)] = -I(nx)
            M2[Block(4,4)] = g.Q_list[t+1][2]
            M2[Block(4,5)] = -g.HxT[2]'
            M2[Block(5,4)] = g.HxT[2]
            
            inv_M2_N2 = -pinv(Array(M2))*Array(N2)
            π̂², π̌² = inv_M2_N2[1:m, 1:nx], inv_M2_N2[1:m, nx+1:nx+m] # uₜ² = π̂²*xₜ + π̌²*uₜ¹
            
            n2 = BlockArray(zeros(m+l+nx+nx+l), [m, l, nx, nx, l])
            n2[Block(1,1)] = g.r_list[t][2][m+1:2*m]
            n2[Block(2,1)] = g.h_list[t][2]
            n2[Block(3,1)] = -c
            n2[Block(4,1)] = g.q_list[t+1][2]
            n2[Block(5,1)] = g.hxT[2]
            # inv_M2_n2 = -Array(M2)\n2 # commented out because we didn't use!
            # βₜ² = inv_M2_n2[1:m] # offset term in the follower's strategy

            Π²[:,1:m] = π̌² # uₜ² = π̂²*xₜ + Π²*uₜ + βₜ², i.e., 0 = π̂²*xₜ + Π²*uₜ + βₜ² - uₜ² 

            # we then solve for the leader, player 1
            Nₜ[Block(1,1)] = [g.S_list[t][1][1:m,:]; g.S_list[t][2][m+1:nu,:]] # double check
            Nₜ[Block(2,1)] = H_xₜ
            Nₜ[Block(3,1)] = -A

            Mₜ[Block(1,1)] = Rₜ
            Mₜ[Block(1,2)] = -Ĥ_uₜ'
            Mₜ[Block(1,3)] = transpose(B̂ₜ)
            Mₜ[Block(1,4)] = Π²'
            Mₜ[Block(2,1)] = H_uₜ
            Mₜ[Block(3,1)] = -B
            Mₜ[Block(3,5)] = I(nx)
            Mₜ[Block(4,3)] = -I(nx*num_player)
            Mₜ[Block(4,5)] = Qₜ₊₁
            Mₜ[Block(4,6)] = -Ĥ_x_terminal'
            Mₜ[Block(5,5)] = H_x_terminal
            Mₜ[Block(6,3)] = B̃²'
            Mₜ[Block(6,4)] = -I(m)

            nₜ[Block(1,1)], nₜ[Block(2,1)], nₜ[Block(3,1)] = rₜ, hₜ, -c
            nₜ[Block(4,1)], nₜ[Block(5,1)] = qₜ₊₁, h_x_terminal
            
            K, k = -pinv(Array(Mₜ))*Array(Nₜ), -pinv(Array(Mₜ))*Array(nₜ)
            π¹_next = K[1:m,:]
            Π_next[1:m,1:nx] = π̂² # πₜ₊₁
            Π_next[m+1:nu, nx+1:2*nx] = π¹_next # true policy for player 1, from x
            
            strategies.P[t] = -K[1:nu, :]
            strategies.α[t] = -k[1:nu]
            
            Aₜ₊₁ = A
            Bₜ₊₁ = B
            K_next = -K[1:nu, :]
            k_next = -k[1:nu, :]
            K_mu_next = -K[nu+1 : nu+2*l, :]
            k_mu_next = -k[nu+1 : nu+2*l, :]
            K_lambda_next = -K[nu+2*l+1 : nu+2*l+2*nx, :]
            k_lambda_next = -k[nu+2*l+1 : nu+2*l+2*nx, :]
            
            K_multipliers[Block(t,1)] = Array(K[vcat(nu+1:nu+2*l+2*nx+m, nu+2*l+2*nx+m+nx+1:nu+2*l+2*nx+m+nx+2*l), :])
            k_multipliers[Block(t,1)] = Array(k[vcat(nu+1:nu+2*l+2*nx+m, nu+2*l+2*nx+m+nx+1:nu+2*l+2*nx+m+nx+2*l), :])

            # in what follows, we manipulate the KKT matrices for KKT residual loss:
            KKT_M = Mₜ
            KKT_N = Nₜ
            KKT_n = nₜ
            # remove all the second order related matrices:
            if output_ground_truth_KKT_matrices == false
                KKT_M[Block(1,1)] = 0.0*Rₜ
                KKT_M[Block(4,5)] = 0.0*Qₜ₊₁
                KKT_N[Block(1,1)] = zeros(nu, nx)
                
                KKT_M[Block(2,1)] = KKT_M[Block(2,1)]*0.0
                KKT_M[Block(3,1)] = KKT_M[Block(3,1)]*0.0
                KKT_M[Block(3,5)] = KKT_M[Block(3,5)]*0.0
                KKT_M[Block(5,5)] = KKT_M[Block(5,5)]*0.0
                KKT_N[Block(2,1)] = KKT_N[Block(2,1)]*0.0
                KKT_N[Block(3,1)] = KKT_N[Block(3,1)]*0.0
                KKT_N[Block(5,1)] = KKT_N[Block(5,1)]*0.0
                KKT_n[Block(2,1)] = KKT_n[Block(2,1)]*0.0
                KKT_n[Block(3,1)] = KKT_n[Block(3,1)]*0.0
                KKT_n[Block(5,1)] = KKT_n[Block(5,1)]*0.0
            end
        else
            # when t < T, we first solve for the follower
            
            Ĥ_xₜ₊₁[Block(1,1)] = g.Hx_list[t+1][1]
            Ĥ_xₜ₊₁[Block(2,2)] = g.Hx_list[t+1][2]
            H̃_uₜ₊₁[Block(1,1)] = g.Hu_list[t+1][1][:,m+1:nu]
            H̃_uₜ₊₁[Block(2,2)] = g.Hu_list[t+1][2][:,1:m]
            
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
            follower_Pₜ¹ = BlockArray(zeros(m+l+nx+nx+m, m+l+nx+m+nx), [m, l, nx, nx, m], [m, l, nx, m, nx])
            follower_Pₜ¹[Block(1,1)] = g.R_list[t][2][m+1:2*m, m+1:2*m]
            follower_Pₜ¹[Block(1,2)] = -g.Hu_list[t][2][:,m+1:nu]'
            follower_Pₜ¹[Block(1,3)] = transpose(B[:,m+1:nu])
            follower_Pₜ¹[Block(2,1)] = g.Hu_list[t][2][:,m+1:nu]
            follower_Pₜ¹[Block(3,1)] = -B[:,m+1:nu]
            follower_Pₜ¹[Block(3,5)] = I(nx)
            follower_Pₜ¹[Block(4,3)] = -I(nx)
            follower_Pₜ¹[Block(4,4)] = π¹_next'
            follower_Pₜ¹[Block(4,5)] = g.Q_list[t+1][2] - 
                g.S_list[t+1][2]'*K_next +
                g.Hx_list[t+1][2]'*K_mu_next[l+1:end,:] -
                Aₜ₊₁'*K_lambda_next[nx+1:end,:] # double check
            follower_Pₜ¹[Block(5,4)] = -I(m)
            # @infiltrate
            follower_Pₜ¹[Block(5,5)] = g.S_list[t+1][2][1:m,:]  - 
                g.R_list[t+1][2][1:m,:]*K_next  + 
                g.Hu_list[t+1][2][:,1:m]'*K_mu_next[l+1:end,:] -
                Bₜ₊₁[:,1:m]'*K_lambda_next[nx+1:end,:] # double check
            tmp_Pₜ¹ = pinv(follower_Pₜ¹)
            
            Nₜ² = BlockArray(zeros(size(tmp_Pₜ¹,1), nx+m), [m, l, nx, nx, m], [nx+m])
            Nₜ²[Block(1,1)] = [g.S_list[t][2][m+1:nu,:] g.R_list[t+1][2][m+1:nu,1:m]] # double check
            Nₜ²[Block(2,1)] = [g.Hx_list[t][2]  g.Hu_list[t][2][:,1:m]]
            Nₜ²[Block(3,1)] = [-A  -B[:,1:m]]
            inv_tmp_Pₜ¹_Nₜ² = -tmp_Pₜ¹*Nₜ²
            π̂², π̌² = inv_tmp_Pₜ¹_Nₜ²[1:m, 1:nx], inv_tmp_Pₜ¹_Nₜ²[1:m, nx+1:nx+m]
            Π²[:,1:m] = π̌² # uₜ² = π̂²*xₜ + π̌²*uₜ¹

            # we then solve the leader
            S̃_uₜ₊₁ = [g.S_list[t+1][1]'; g.S_list[t+1][2]'] # double check
            R̃ₜ₊₁ = [g.R_list[t+1][1][m+1:nu,:]; g.R_list[t+1][2][1:m,:]] # double check
            S̃ₓ₊₁ = [g.S_list[t+1][1][m+1:nu,:]; g.S_list[t+1][2][1:m,:]] # double check

            leader_Pₜ¹ = BlockArray(zeros(new_M_size, new_M_size), 
                [nu, 2*l, nx, 2*nx, nu, m], 
                [nu, 2*l, 2*nx, nu, m, nx])
            leader_Pₜ¹[Block(1,1)] = Rₜ
            leader_Pₜ¹[Block(1,2)] = -Ĥ_uₜ'
            leader_Pₜ¹[Block(1,3)] = transpose(B̂ₜ)
            leader_Pₜ¹[Block(1,5)] = Π²'
            leader_Pₜ¹[Block(2,1)] = H_uₜ
            leader_Pₜ¹[Block(3,1)] = -B
            leader_Pₜ¹[Block(3,6)] = I(nx)
            leader_Pₜ¹[Block(4,3)] = -I(nx*num_player)
            leader_Pₜ¹[Block(4,4)] = Π_next'
            leader_Pₜ¹[Block(4,6)] = Qₜ₊₁ - 
                S̃_uₜ₊₁*K_next + 
                Ĥ_xₜ₊₁'*K_mu_next  -
                Âₜ₊₁'*K_lambda_next # double check
            leader_Pₜ¹[Block(5,4)] = -I(nu)
            leader_Pₜ¹[Block(5,6)] = S̃ₓ₊₁ - 
                R̃ₜ₊₁*K_next + 
                H̃_uₜ₊₁'*K_mu_next -
                B̃ₜ₊₁'*K_lambda_next # double check
            leader_Pₜ¹[Block(6,3)] = B̃²'
            leader_Pₜ¹[Block(6,5)] = -I(m)

            Pₜ¹ = pinv(leader_Pₜ¹)
            Pₜ²nₜ₊₁ = Array(-Pₜ¹* [zeros(nu+2*l+nx, 1); 
                S̃_uₜ₊₁*k_next-Ĥ_xₜ₊₁'*k_mu_next+Âₜ₊₁'*k_lambda_next;  # double check
                R̃ₜ₊₁*k_next-H̃_uₜ₊₁'*k_mu_next+B̃ₜ₊₁'*k_lambda_next;  # double check
                zeros(m,1) ])
            
            K, k = -Pₜ¹*[[g.S_list[t][1][1:m,:]; g.S_list[t][2][m+1:nu,:]]; H_xₜ; -A; zeros(new_M_size-nu-2*l-nx,nx)], -Pₜ¹*[rₜ; hₜ; -c; qₜ₊₁; zeros(nu+m,1)] - Pₜ²nₜ₊₁ # double check
            π¹_next = K[1:m,:] # update π¹_next
            Π_next[1:m,1:nx] = π̂²
            Π_next[m+1:nu, nx+1:2*nx] = π¹_next # uₜ² = π̂²*xₜ + π̌²*uₜ¹
            
            strategies.P[t] = -K[1:nu, :]
            strategies.α[t] = -k[1:nu]
            
            Aₜ₊₁ = A
            Bₜ₊₁ = B
            K_next = -K[1:nu, :]
            k_next = -k[1:nu, :]
            K_mu_next = -K[nu+1 : nu+2*l, :]
            k_mu_next = -k[nu+1 : nu+2*l, :]
            K_lambda_next = -K[nu+2*l+1 : nu+2*l+2*nx,:]
            k_lambda_next = -k[nu+2*l+1 : nu+2*l+2*nx,:]
            K_multipliers[Block(t,1)] = Array(K[nu+1 : nu+2*l+2*nx+nu+m, :])
            k_multipliers[Block(t,1)] = Array(k[nu+1 : nu+2*l+2*nx+nu+m, :])



            # in what follows, we construct the KKT matrix for the next time instant. Related to KKT residual loss
            M_size = M_size + new_M_size
            D1 = deepcopy(leader_Pₜ¹)
            D2 = BlockArray(
                zeros(new_M_size, M_size-new_M_size), 
                [nu, 2*l, nx, 2*nx, nu, m], 
                [nu, 2*l, 2*nx, (M_size-new_M_size)-nu-2*l-2*nx]
            )
            D2[Block(4,2)] = -Ĥ_xₜ₊₁'
            D2[Block(4,3)] = Âₜ₊₁'
            D2[Block(5,2)] = -H̃_uₜ₊₁'
            D2[Block(5,3)] = B̃ₜ₊₁'
            # remove all the second order related matrices:
            if output_ground_truth_KKT_matrices == false
                D1[Block(1,1)] = 0.0*Rₜ
                D1[Block(4,6)] = 0.0*Qₜ₊₁
                D1[Block(5,6)] = zeros(nu, nx)
                D2[Block(4,1)] = zeros(2*nx, nu)
                D2[Block(5,1)] = zeros(nu, nu)

                D1[Block(2,1)] = D1[Block(2,1)]*0.0
                D1[Block(3,1)] = D1[Block(3,1)]*0.0
                D1[Block(3,6)] = D1[Block(3,6)]*0.0
            end
            KKT_M = [
                D1 D2 ; 
                zeros(M_size - new_M_size, new_M_size-nx)  KKT_N  KKT_M 
            ]
            # construct KKT_N and KKT_n
            KKT_N = [zeros(nu,nx); H_xₜ; -A; zeros(M_size - nu-2*l-nx, nx)] # remove the second order related matrices
            KKT_n = [rₜ; hₜ; -c; qₜ₊₁; zeros(nu+m); KKT_n]
            if output_ground_truth_KKT_matrices == false
                KKT_N[1:nu+2*l+nx,:] = KKT_N[1:nu+2*l+nx,:]*0.0
                KKT_n[nu+1:nu+2*l+nx] = KKT_n[nu+1:nu+2*l+nx]*0.0
            end

        end
    end
    # solution = K*x0+k


    # meeting notes: 8/29. introduction, motivated by GFNE. 
    if output_only_KKT_matrix_for_KKT_residual_loss == false
        if simulate_the_trajectory_in_global_coordinate == false
            Δx = [0.0*x0 for t in 1:T+1]
            Δu = [zeros(nu) for t in 1:T]
        else
            Δx = [x0 for t in 1:T+1] # in global coordinate
            Δu = [zeros(nu) for t in 1:T]
        end
        for t in 1:1:T
            if t == T
                μ[t] = K_multipliers[Block(t,1)][1:2*l,:]*Δx[t] + k_multipliers[Block(t,1)][1:2*l]
                λ[t] = K_multipliers[Block(t,1)][2*l+1:2*l+2*nx,:]*Δx[t] + k_multipliers[Block(t,1)][2*l+1:2*l+2*nx]
                ψ[t] = K_multipliers[Block(t,1)][2*l+2*nx+1:2*l+2*nx+m,:]*Δx[t] + k_multipliers[Block(t,1)][2*l+2*nx+1:2*l+2*nx+m]
                μ[t+1] = K_multipliers[Block(t,1)][end-2*l+1:end,:]*Δx[t] + k_multipliers[Block(t,1)][end-2*l+1:end]

                Δu[t] = -strategies.P[t]*Δx[t] - strategies.α[t]
                Δx[t+1] = g.A_list[t]*Δx[t] + g.B_list[t]*Δu[t] + g.c_list[t] # update x
            else
                μ[t] = K_multipliers[Block(t,1)][1:2*l,:]*Δx[t] + k_multipliers[Block(t,1)][1:2*l]
                λ[t] = K_multipliers[Block(t,1)][2*l+1:2*l+2*nx,:]*Δx[t] + k_multipliers[Block(t,1)][2*l+1:2*l+2*nx]
                η[t] = K_multipliers[Block(t,1)][2*l+2*nx+1:2*l+2*nx+nu,:]*Δx[t] + k_multipliers[Block(t,1)][2*l+2*nx+1:2*l+2*nx+nu]
                ψ[t] = K_multipliers[Block(t,1)][end-m+1:end,:]*Δx[t] + k_multipliers[Block(t,1)][end-m+1:end]
                
                Δu[t] = - strategies.P[t]*Δx[t] - strategies.α[t]
                Δx[t+1] = g.A_list[t]*Δx[t] + g.B_list[t]*Δu[t] + g.c_list[t] # update x
            end
        end
        return Δx, Δu, μ, λ, η, ψ, KKT_M, KKT_N, KKT_n
    else
        return KKT_M, KKT_N, KKT_n
    end
end