using Infiltrator
function nw_pdip_fbst_lq_solver!(
    strategies, # this is updated in this function
    g::Constrained_LQGame, # this is the lq approximation of the original game
    current_op::trajectory, # this is not updated in this function
    ρ::Float64, # PDIP penalty parameter, set to 0.0 for non-inequality constrained problems, ρ = 1/t
    output_only_KKT_matrix_for_KKT_residual_loss = false,
    simulate_the_trajectory_in_global_coordinate = false,
    output_ground_truth_KKT_matrices = false,
    have_nontrivial_inequailty_constraints_in_intermediate_steps = true,
    have_nontrivial_terminal_inequality_constraints = true,
    regularization_turn_on = false
    )
    regularization_weight = 1e-3
    l = g.equality_constraints_size
    ll = g.inequality_constraints_size
    
    x0 = g.x0
    # extract state, total control, and each player's control dimensions
    nx, nu, m = g.nx, g.nu, length(g.players_u_index_list[1])
    T = g.horizon
    # m is the input size of agent i, and T is the horizon.
    num_player = g.n_players # number of player
    @assert length(num_player) != 2
    M_size = nu + ll*num_player + ll*num_player + l*num_player + nx*num_player + m + nx + ll*num_player + ll*num_player + l*num_player
    # size of the M matrix for each time instant, will be used to define KKT matrix
    new_M_size = nu + ll*num_player + ll*num_player + l*num_player + nx*num_player + (num_player-1)*nu + m + nx
    # new_M_row_size = nu + ll*num_player + l*num_player + nx + nx*num_player + ll*num_player + (num_player-1)*nu + m

    # initialize some intermidiate variables in KKT conditions    
    Mₜ = BlockArray(zeros(M_size, M_size), 
        [nu, ll*num_player, ll*num_player, l*num_player, nx, nx*num_player, ll*num_player, ll*num_player, l*num_player, m], 
        [nu, ll*num_player, ll*num_player, l*num_player, nx*num_player, m, nx, ll*num_player, ll*num_player, l*num_player] )
    Nₜ = BlockArray(zeros(M_size, nx),  
        [nu, ll*num_player, ll*num_player, l*num_player, nx, nx*num_player, ll*num_player, ll*num_player, l*num_player, m], [nx])
    nₜ = BlockArray(zeros(M_size),      
        [nu, ll*num_player, ll*num_player, l*num_player, nx, nx*num_player, ll*num_player, ll*num_player, l*num_player, m])
    s = [zeros(ll*num_player) for t in 1:T+1]
    γ = [zeros(ll*num_player) for t in 1:T+1]
    μ = [zeros(l*num_player) for t in 1:T+1]
    λ = [zeros(nx*num_player) for t in 1:T]
    η = [zeros(nu) for t in 1:T-1]
    ψ = [zeros(m) for t in 1:T]
    
    gₜₖ = BlockArray(zeros(ll*num_player), [ll, ll])
    ĝₜₖ = BlockArray(zeros(ll*num_player, ll*num_player), [ll, ll], [ll, ll]) # diagonalization of the above variable
    gₜ₊₁ₖ = BlockArray(zeros(ll*num_player), [ll, ll])
    ĝₜ₊₁ₖ = BlockArray(zeros(ll*num_player, ll*num_player), [ll, ll], [ll, ll]) # diagonalization of the above variable
    g_terminal = BlockArray(zeros(ll*num_player), [ll, ll])
    ĝ_terminal = BlockArray(zeros(ll*num_player, ll*num_player), [ll, ll], [ll, ll]) # diagonalization of the above variable
    γ̂ₜₖ = BlockArray(zeros(ll*num_player, ll*num_player), [ll, ll], [ll, ll]) # diagonalization of γₜₖ
    γ̂ₜ₊₁ₖ = BlockArray(zeros(ll*num_player, ll*num_player), [ll, ll], [ll, ll]) # diagonalization of γₜ₊₁ₖ
    γ̂_terminal = BlockArray(zeros(ll*num_player, ll*num_player), [ll, ll], [ll, ll]) 
    
    ŝₜₖ = BlockArray(zeros(ll*num_player, ll*num_player), [ll, ll], [ll, ll]) # diagonalization of sₜₖ
    ŝₜ₊₁ₖ = BlockArray(zeros(ll*num_player, ll*num_player), [ll, ll], [ll, ll]) # diagonalization of sₜ₊₁ₖ
    ŝ_terminal = BlockArray(zeros(ll*num_player, ll*num_player), [ll, ll], [ll, ll]) # diagonalization of s_terminal

    if begin 
        norm(g.Gx_list[1][1]) == 0.0 && 
        norm(g.Gx_list[1][2]) == 0.0 && 
        norm(g.Gu_list[1][1]) == 0.0 &&
        norm(g.Gu_list[1][2]) == 0.0 &&
        norm(g.g_list[1][1]) == 0.0 && 
        norm(g.g_list[1][2]) == 0.0
        end
        have_nontrivial_inequailty_constraints_in_intermediate_steps = false
    end
    if norm(g.GxT[1]) == 0.0 && norm(g.GxT[2]) == 0.0 && norm(g.gxT[1]) == 0.0 && norm(g.gxT[2]) == 0.0
        have_nontrivial_terminal_inequality_constraints = false
    end

    normalization_term = BlockArray(zeros(ll*num_player, ll*num_player), [ll, ll], [ll, ll])
    normalization_term_terminal = BlockArray(zeros(ll*num_player, ll*num_player), [ll, ll], [ll, ll])

    if have_nontrivial_terminal_inequality_constraints
        normalization_term_terminal[Block(1,1)] = diagm(ones(ll))
        normalization_term_terminal[Block(2,2)] = diagm(ones(ll))
    end
    if have_nontrivial_inequailty_constraints_in_intermediate_steps
        normalization_term[Block(1,1)] = diagm(ones(ll))
        normalization_term[Block(2,2)] = diagm(ones(ll))
    end
    G_x_terminal = BlockArray(zeros(ll*num_player, nx), [ll, ll], [nx])
    Ĝ_x_terminal = BlockArray(zeros(ll*num_player, nx*num_player), [ll, ll], [nx, nx])
    G_xₜ = BlockArray(zeros(ll*num_player, nx), [ll, ll], [nx])
    G_uₜ = BlockArray(zeros(ll*num_player, nu), [ll, ll], [nu])
    G_xₜ₊₁ = BlockArray(zeros(ll*num_player, nx), [ll, ll], [nx])
    Ĝ_uₜ = BlockArray(zeros(ll*num_player, m*num_player), [ll, ll], [m, m])

    Ĝ_xₜ₊₁ = BlockArray(zeros(2*ll, 2*nx), [ll,ll], [nx,nx])
    G̃_uₜ₊₁ = BlockArray(zeros(2*ll, nu), [ll,ll], [m,m])


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
    K_gamma_next = zeros(2*ll, nx)
    k_gamma_next = zeros(2*ll)
    K_lambda_next = zeros(2*nx, nx)
    k_lambda_next = zeros(2*nx)
    K_mu_next = zeros(2*l, nx)
    k_mu_next = zeros(2*l)

    K_multipliers = BlockArray(zeros((T-1)*(2*ll+2*ll + l*num_player + 2*nx + nu + m) + 
        (2*ll+2*ll + 2*l + 2*nx + m +2*ll+ 2*ll + 2*l), nx), 
        vcat([2*ll+2*ll + 2*l + 2*nx + nu + m for t in 1:T-1], [2*ll+2*ll + 2*l + 2*nx + m + 2*ll+2*ll + 2*l]), [nx])
    k_multipliers = BlockArray(zeros((T-1)*(2*ll+2*ll + l*num_player + 2*nx + nu + m) + 
        (2*ll+2*ll + 2*l + 2*nx + m +2*ll+ 2*ll + 2*l), 1), 
        vcat([2*ll+2*ll + 2*l + 2*nx + nu + m for t in 1:T-1], [2*ll+2*ll + 2*l + 2*nx + m + 2*ll+2*ll + 2*l]), [1])
    
    KKT_M = deepcopy(Mₜ)
    KKT_N = deepcopy(Nₜ)
    KKT_n = deepcopy(nₜ)

    for t in T:-1:1 # work in backwards to construct the KKT constraint matrix
        # convenience shorthands for the relevant quantities
        ŝₜₖ[Block(1,1)] = diagm(current_op.s[t][1:ll])
        ŝₜₖ[Block(2,2)] = diagm(current_op.s[t][ll+1:end])
        A, B, c = g.A_list[t], g.B_list[t], g.c_list[t]
        Âₜ₊₁, B̂ₜ = zeros(nx*num_player, nx*num_player), zeros(nx*num_player, nu)
        Rₜ, Qₜ₊₁ = zeros(nu, nu), zeros(nx*num_player, nx)
        rₜ, qₜ₊₁ = zeros(nu), zeros(nx*num_player)
        
        B̃ₜ₊₁ = zeros(num_player*nx, (num_player-1)*nu) 
        B̃² = [B[:,m+1:nu]; zeros(nx, m)]

        G_xₜ[Block(1,1)] = g.Gx_list[t][1]
        G_xₜ[Block(2,1)] = g.Gx_list[t][2]
        G_uₜ[Block(1,1)] = g.Gu_list[t][1]
        G_uₜ[Block(2,1)] = g.Gu_list[t][2]
        Ĝ_uₜ[Block(1,1)] = g.Gu_list[t][1][:,1:m]
        Ĝ_uₜ[Block(2,2)] = g.Gu_list[t][2][:,m+1:nu]
        gₜₖ[Block(1,1)] = g.g_list[t][1]
        gₜₖ[Block(2,1)] = g.g_list[t][2]
        ĝₜₖ[Block(1,1)] = diagm(g.g_list[t][1])
        ĝₜₖ[Block(2,2)] = diagm(g.g_list[t][2])
        γ̂ₜₖ[Block(1,1)] = diagm(current_op.γ[t][1:ll])
        γ̂ₜₖ[Block(2,2)] = diagm(current_op.γ[t][ll+1:end])

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

        hₜ[Block(1)]= g.h_list[t][1]
        hₜ[Block(2)]= g.h_list[t][2]

        R̃ₜ¹² = g.R_list[t][1][m+1:nu,:]
        Sₜ¹² = g.S_list[t][1][m+1:nu,:]
        G̃ᵤₜ¹² = [g.Gu_list[t][1][:,m+1:nu]; zeros(ll, m) ]
        H̃ᵤₜ¹² = [g.Hu_list[t][1][:,m+1:nu]; zeros(l, m) ]

        if t == T
            ŝ_terminal[Block(1,1)] = diagm(current_op.s[T+1][1:ll])
            ŝ_terminal[Block(2,2)] = diagm(current_op.s[T+1][ll+1:end])
            g_terminal[Block(1,1)] = g.gxT[1]
            g_terminal[Block(2,1)] = g.gxT[2]
            ĝ_terminal[Block(1,1)] = diagm(g.gxT[1])
            ĝ_terminal[Block(2,2)] = diagm(g.gxT[2])
            G_x_terminal[Block(1,1)] = g.GxT[1]
            G_x_terminal[Block(2,1)] = g.GxT[2]
            Ĝ_x_terminal[Block(1,1)] = g.GxT[1]
            Ĝ_x_terminal[Block(2,2)] = g.GxT[2]
            
            γ̂_terminal[Block(1,1)] = diagm(current_op.γ[t+1][1:ll])
            γ̂_terminal[Block(2,2)] = diagm(current_op.γ[t+1][ll+1:end])

            Ĥ_x_terminal = BlockArray(zeros(2*l, 2*nx), [l,l], [nx,nx])
            H_x_terminal[Block(1,1)] = g.HxT[1]
            H_x_terminal[Block(2,1)] = g.HxT[2]
            Ĥ_x_terminal[Block(1,1)] = g.HxT[1]
            Ĥ_x_terminal[Block(2,2)] = g.HxT[2]
            h_x_terminal[Block(1)]= g.hxT[1]
            h_x_terminal[Block(2)]= g.hxT[2]
            # We first solve for the follower, player 2
            
            for (ii, udxᵢ) in enumerate(g.players_u_index_list)
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                # if regularization_turn_on
                #     g.Q_list[t+1][ii] = g.Q_list[t+1][ii]+regularization_weight*I(nx)
                #     g.R_list[t][ii] = g.R_list[t][ii]+regularization_weight*I(nu)
                # end
                Qₜ₊₁[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = g.Q_list[t+1][ii], g.R_list[t][ii][udxᵢ,:]
                qₜ₊₁[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = g.q_list[t+1][ii], g.r_list[t][ii][udxᵢ]
            end
            N2 = BlockArray(zeros(m+ll+ll+l+nx+nx+ll+ll+l, nx+m), 
                [m, ll, ll, l, nx, nx, ll, ll, l], [nx+m])
            M2 = BlockArray(zeros(m+ll+ll+l+nx+nx+ll+ll+l, m+ll+ll+l+nx+nx+ll+ll+l), 
                [m, ll, ll, l, nx, nx, ll, ll, l], [m, ll, ll, l, nx, nx, ll, ll, l])

            N2[Block(1,1)] = [g.S_list[t][2][m+1:nu,:] g.R_list[t][2][m+1:nu,1:m]] # double check
            N2[Block(3,1)] = normalization_term[Block(2,2)]*[g.Gx_list[t][2] g.Gu_list[t][2][:,1:m]] # double check
            N2[Block(4,1)] = [g.Hx_list[t][2] g.Hu_list[t][2][:,1:m]]
            N2[Block(5,1)] = [-A -B[:,1:m]]

            M2[Block(1,1)] = g.R_list[t][2][m+1:2*m, m+1:2*m]
            M2[Block(1,3)] = -g.Gu_list[t][2][:,m+1:nu]'
            M2[Block(1,4)] = -g.Hu_list[t][2][:,m+1:nu]'
            M2[Block(1,5)] = transpose(B[:,m+1:nu]) # B̂ₜ
            
            M2[Block(2,2)] = normalization_term[Block(2,2)]*γ̂ₜₖ[Block(2,2)]#/ρ
            M2[Block(2,3)] = normalization_term[Block(2,2)]*ŝₜₖ[Block(2,2)]#/ρ

            M2[Block(3,1)] = normalization_term[Block(2,2)]*g.Gu_list[t][2][:,m+1:nu]
            M2[Block(3,2)] = normalization_term[Block(2,2)]*(-I(ll))
            M2[Block(4,1)] = g.Hu_list[t][2][:,m+1:nu]
            M2[Block(5,1)] = -B[:,m+1:nu]
            M2[Block(5,6)] = I(nx)
            M2[Block(6,5)] = -I(nx)
            M2[Block(6,6)] = g.Q_list[t+1][2]
            M2[Block(6,8)] = -g.GxT[2]'
            M2[Block(6,9)] = -g.HxT[2]'
            M2[Block(7,7)] = normalization_term_terminal[Block(2,2)]*γ̂_terminal[Block(2,2)]#/ρ
            M2[Block(7,8)] = normalization_term_terminal[Block(2,2)]*ŝ_terminal[Block(2,2)]#/ρ
            M2[Block(8,6)] = normalization_term_terminal[Block(2,2)]*g.GxT[2]
            M2[Block(8,7)] = normalization_term_terminal[Block(2,2)]*(-I(ll))
            M2[Block(9,6)] = g.HxT[2]
            
            inv_M2_N2 = -pinv(Array(M2))*Array(N2)
            π̂², π̌² = inv_M2_N2[1:m, 1:nx], inv_M2_N2[1:m, nx+1:nx+m] # uₜ² = π̂²*xₜ + π̌²*uₜ¹
            Π²[:,1:m] = π̌² # uₜ² = π̂²*xₜ + Π²*uₜ + βₜ², i.e., 0 = π̂²*xₜ + Π²*uₜ + βₜ² - uₜ² 

            # we then solve for the leader, player 1
            Nₜ[Block(1,1)] = [g.S_list[t][1][1:m,:]; g.S_list[t][2][m+1:nu,:]] # double check
            Nₜ[Block(3,1)] = normalization_term*G_xₜ # double check
            Nₜ[Block(4,1)] = H_xₜ
            Nₜ[Block(5,1)] = -A
            Nₜ[Block(10,1)] = Sₜ¹²

            Mₜ[Block(1,1)] = Rₜ
            Mₜ[Block(1,3)] = -Ĝ_uₜ'
            Mₜ[Block(1,4)] = -Ĥ_uₜ'
            Mₜ[Block(1,5)] = transpose(B̂ₜ)
            Mₜ[Block(1,6)] = Π²'

            Mₜ[Block(2,2)] = normalization_term*γ̂ₜₖ#/ρ
            Mₜ[Block(2,3)] = normalization_term*ŝₜₖ#/ρ
            Mₜ[Block(3,1)] = normalization_term*G_uₜ
            
            Mₜ[Block(3,2)] = Matrix(normalization_term)*(-I(ll*num_player))
            Mₜ[Block(4,1)] = H_uₜ
            Mₜ[Block(5,1)] = -B
            Mₜ[Block(5,7)] = I(nx)
            Mₜ[Block(6,5)] = -I(nx*num_player)
            Mₜ[Block(6,7)] = Qₜ₊₁
            Mₜ[Block(6,9)] = -Ĝ_x_terminal'
            Mₜ[Block(6,10)] = -Ĥ_x_terminal'
            
            Mₜ[Block(7,8)] = normalization_term_terminal*γ̂_terminal#/ρ
            Mₜ[Block(7,9)] = normalization_term_terminal*ŝ_terminal#/ρ
            Mₜ[Block(8,7)] = normalization_term_terminal*G_x_terminal
            Mₜ[Block(8,8)] = Matrix(normalization_term_terminal)*(-I(ll*num_player))
            Mₜ[Block(9,7)] = H_x_terminal
            Mₜ[Block(10,1)] = R̃ₜ¹²
            Mₜ[Block(10,3)] = -G̃ᵤₜ¹²'
            Mₜ[Block(10,4)] = -H̃ᵤₜ¹²'
            Mₜ[Block(10,5)] = B̃²'
            Mₜ[Block(10,6)] = -I(m)

            nₜ[Block(1,1)] = rₜ #+ Π²'*current_op.ψ[t] # double check!!!!!!
            nₜ[Block(2,1)] = normalization_term*( - ρ * ones(ll*num_player))#/ρ#normalization_term*(γ̂ₜₖ*current_op.s[t] - ρ * ones(ll*num_player))
            nₜ[Block(3,1)] = normalization_term*(gₜₖ - current_op.s[t]  )
            nₜ[Block(4,1)] = hₜ
            nₜ[Block(5,1)] = -c
            nₜ[Block(6,1)] = qₜ₊₁ #- Ĝ_x_terminal'*current_op.γ[t+1] - Ĥ_x_terminal'*current_op.μ[t+1] # double check!!!!!!
            nₜ[Block(7,1)] = normalization_term_terminal*(- ρ * ones(ll*num_player))#/ρ#normalization_term_terminal*(γ̂_terminal*current_op.s[end] - ρ * ones(ll*num_player))
            nₜ[Block(8,1)] = normalization_term_terminal*(g_terminal - current_op.s[end])
            nₜ[Block(9,1)] = h_x_terminal
            nₜ[Block(10,1)] = g.r_list[t][1][m+1:nu] #- current_op.ψ[t] # double check!!!!!!
            if regularization_turn_on
                Mₜ = Mₜ+regularization_weight*I(size(Array(Mₜ))[1])
            end
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
            K_gamma_next = -K[nu+2*ll+1 : nu+4*ll, :]
            k_gamma_next = -k[nu+2*ll+1 : nu+4*ll, :]
            K_mu_next = -K[nu+4*ll+1 : nu+4*ll+2*l, :]
            k_mu_next = -k[nu+4*ll+1 : nu+4*ll+2*l, :]
            K_lambda_next = -K[nu+4*ll+2*l+1 : nu+4*ll+2*l+2*nx, :]
            k_lambda_next = -k[nu+4*ll+2*l+1 : nu+4*ll+2*l+2*nx, :]
            # @infiltrate
            K_multipliers[Block(t,1)] = Array(
                K[vcat(nu+1 : nu+4*ll+2*l+2*nx+m, nu+4*ll+2*l+2*nx+m+nx+1 : nu+4*ll+2*l+2*nx+m+nx+2*l+4*ll), :])
            k_multipliers[Block(t,1)] = Array(
                k[vcat(nu+1 : nu+4*ll+2*l+2*nx+m, nu+4*ll+2*l+2*nx+m+nx+1 : nu+4*ll+2*l+2*nx+m+nx+2*l+4*ll), :])

            # in what follows, we manipulate the KKT matrices for KKT residual loss:
            # @infiltrate
            KKT_M = Mₜ
            KKT_N = Nₜ
            KKT_n = nₜ
            if output_ground_truth_KKT_matrices == false
                # TODO: check the line search part!
                # remove all the second order related matrices:
                KKT_M[Block(1,1)] = 0.0*Rₜ # check
                KKT_M[Block(6,7)] = 0.0*Qₜ₊₁ # check
                KKT_M[Block(10,1)] = 0.0*R̃ₜ¹² # check
                KKT_N[Block(1,1)] = zeros(nu, nx) # check
                KKT_N[Block(10,1)] = zeros(m, nx) # check

                KKT_M[nu+1 : nu+4*ll+2*l+nx, :] = KKT_M[nu+1 : nu+4*ll+2*l+nx, :]*0.0 # inequality, equality, dynamics constraint
                KKT_M[nu+4*ll+2*l+nx+2*nx+1 : nu+4*ll+2*l+nx+2*nx+4*ll+2*l, :]=KKT_M[nu+4*ll+2*l+nx+2*nx+1 : nu+4*ll+2*l+nx+2*nx+4*ll+2*l, :]*0.0 # ineq + eq

                KKT_N[Block(3,1)] = KKT_N[Block(3,1)]*0.0
                KKT_N[Block(4,1)] = KKT_N[Block(4,1)]*0.0
                KKT_N[Block(5,1)] = KKT_N[Block(5,1)]*0.0

                KKT_n[nu+1 : nu+4*ll+2*l+nx, :] = KKT_n[nu+1 : nu+4*ll+2*l+nx, :]*0.0
                KKT_n[nu+4*ll+2*l+nx+2*nx+1 : nu+4*ll+2*l+nx+2*nx+4*ll+2*l, :]=KKT_n[nu+4*ll+2*l+nx+2*nx+1 : nu+4*ll+2*l+nx+2*nx+4*ll+2*l, :]*0.0
            end
        else
            # when t < T, we first solve for the follower
            ŝₜ₊₁ₖ[Block(1,1)] = diagm(current_op.s[t+1][1:ll])
            ŝₜ₊₁ₖ[Block(2,2)] = diagm(current_op.s[t+1][ll+1:end])
            G_xₜ₊₁[Block(1,1)] = g.Gx_list[t+1][1]
            G_xₜ₊₁[Block(2,1)] = g.Gx_list[t+1][2]
            Ĝ_xₜ₊₁[Block(1,1)] = g.Gx_list[t+1][1]
            Ĝ_xₜ₊₁[Block(2,2)] = g.Gx_list[t+1][2]

            gₜ₊₁ₖ[Block(1,1)] = g.g_list[t+1][1]
            gₜ₊₁ₖ[Block(2,1)] = g.g_list[t+1][2]
            ĝₜ₊₁ₖ[Block(1,1)] = diagm(g.g_list[t+1][1])
            ĝₜ₊₁ₖ[Block(2,2)] = diagm(g.g_list[t+1][2])
            γ̂ₜ₊₁ₖ[Block(1,1)] = diagm(current_op.γ[t+1][1:ll])
            γ̂ₜ₊₁ₖ[Block(2,2)] = diagm(current_op.γ[t+1][ll+1:end])
            
            G̃_uₜ₊₁[Block(1,1)] = g.Gu_list[t+1][1][:,m+1:nu]
            G̃_uₜ₊₁[Block(2,2)] = g.Gu_list[t+1][2][:,1:m]

            Ĥ_xₜ₊₁[Block(1,1)] = g.Hx_list[t+1][1]
            Ĥ_xₜ₊₁[Block(2,2)] = g.Hx_list[t+1][2]
            H̃_uₜ₊₁[Block(1,1)] = g.Hu_list[t+1][1][:,m+1:nu]
            H̃_uₜ₊₁[Block(2,2)] = g.Hu_list[t+1][2][:,1:m]
            
            for (ii, udxᵢ) in enumerate(g.players_u_index_list)
                Âₜ₊₁[(ii-1)*nx+1:ii*nx, (ii-1)*nx+1:ii*nx] = Aₜ₊₁
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                # if regularization_turn_on
                #     g.Q_list[t+1][ii] = g.Q_list[t+1][ii]+regularization_weight*I(nx)
                #     g.R_list[t][ii] = g.R_list[t][ii]+regularization_weight*I(nu)
                # end
                Qₜ₊₁[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = g.Q_list[t+1][ii], g.R_list[t][ii][udxᵢ,:]
                qₜ₊₁[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = g.q_list[t+1][ii], g.r_list[t][ii][udxᵢ]
                udxᵢ_complement = setdiff(1:1:nu, udxᵢ)
                
                B̃ₜ₊₁[(ii-1)*nx+1:ii*nx, (ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m] = Bₜ₊₁[:,udxᵢ_complement]
            end
            # below is the construction of the KKT matrix for follower
            # we don't need to compute the offset term here
            follower_Pₜ¹ = BlockArray(
                zeros(m+ll+ll+l+nx+nx+m, m+ll+ll+l+nx+m+nx), [m, ll, ll, l, nx, nx, m], [m, ll, ll, l, nx, m, nx]
            )
            follower_Pₜ¹[Block(1,1)] = g.R_list[t][2][m+1:2*m, m+1:2*m]
            follower_Pₜ¹[Block(1,3)] = -g.Gu_list[t][2][:,m+1:nu]'
            follower_Pₜ¹[Block(1,4)] = -g.Hu_list[t][2][:,m+1:nu]'
            follower_Pₜ¹[Block(1,5)] = transpose(B[:,m+1:nu])
            follower_Pₜ¹[Block(2,2)] = normalization_term[Block(2,2)]*γ̂ₜₖ[Block(2,2)]#/ρ
            follower_Pₜ¹[Block(2,3)] = normalization_term[Block(2,2)]*ŝₜₖ[Block(2,2)]#/ρ
            follower_Pₜ¹[Block(3,1)] = normalization_term[Block(2,2)]*g.Gu_list[t][2][:,m+1:nu]
            follower_Pₜ¹[Block(3,2)] = normalization_term[Block(2,2)]*(-I(ll))
            follower_Pₜ¹[Block(4,1)] = g.Hu_list[t][2][:,m+1:nu]
            follower_Pₜ¹[Block(5,1)] = -B[:,m+1:nu]
            follower_Pₜ¹[Block(5,7)] = I(nx)
            follower_Pₜ¹[Block(6,5)] = -I(nx)
            
            follower_Pₜ¹[Block(6,6)] = π¹_next'
            follower_Pₜ¹[Block(6,7)] = g.Q_list[t+1][2] - 
                g.S_list[t+1][2]'*K_next +
                g.Gx_list[t+1][2]'*K_gamma_next[ll+1:end,:] +
                g.Hx_list[t+1][2]'*K_mu_next[l+1:end,:] -
                Aₜ₊₁'*K_lambda_next[nx+1:end,:] # double check

            follower_Pₜ¹[Block(7,6)] = -I(m)
            
            follower_Pₜ¹[Block(7,7)] = g.S_list[t+1][2][1:m,:]  - 
                g.R_list[t+1][2][1:m,:]*K_next  + 
                g.Gu_list[t+1][2][:,1:m]'*K_gamma_next[ll+1:end,:] +
                g.Hu_list[t+1][2][:,1:m]'*K_mu_next[l+1:end,:] -
                Bₜ₊₁[:,1:m]'*K_lambda_next[nx+1:end,:] # double check
            if regularization_turn_on
                follower_Pₜ¹ = follower_Pₜ¹+regularization_weight*I(size(Array(follower_Pₜ¹))[1])
            end
            tmp_Pₜ¹ = pinv(follower_Pₜ¹)
            
            Nₜ² = BlockArray(zeros(size(follower_Pₜ¹,1), nx+m), [m, ll, ll, l, nx, nx, m], [nx+m])
            Nₜ²[Block(1,1)] = [g.S_list[t][2][m+1:nu,:] g.R_list[t+1][2][m+1:nu,1:m]] # double check
            Nₜ²[Block(3,1)] = normalization_term[Block(2,2)]*[g.Gx_list[t][2] g.Gu_list[t][2][:,1:m]] # double check
            Nₜ²[Block(4,1)] = [g.Hx_list[t][2]  g.Hu_list[t][2][:,1:m]]
            Nₜ²[Block(5,1)] = [-A  -B[:,1:m]]
            inv_tmp_Pₜ¹_Nₜ² = -tmp_Pₜ¹*Nₜ²
            π̂², π̌² = inv_tmp_Pₜ¹_Nₜ²[1:m, 1:nx], inv_tmp_Pₜ¹_Nₜ²[1:m, nx+1:nx+m]
            Π²[:,1:m] = π̌² # uₜ² = π̂²*xₜ + π̌²*uₜ¹

            # we then solve the leader
            S̃_uₜ₊₁ = [g.S_list[t+1][1]'; g.S_list[t+1][2]'] # double check
            R̃ₜ₊₁ = [g.R_list[t+1][1][m+1:nu,:]; g.R_list[t+1][2][1:m,:]] # double check
            S̃ₓ₊₁ = [g.S_list[t+1][1][m+1:nu,:]; g.S_list[t+1][2][1:m,:]] # double check

            leader_Pₜ¹ = BlockArray(zeros(new_M_size, new_M_size), 
                [nu, 2*ll, 2*ll, 2*l, nx, 2*nx, nu, m], 
                [nu, 2*ll, 2*ll, 2*l, 2*nx, nu, m, nx])
            leader_Pₜ¹[Block(1,1)] = Rₜ
            leader_Pₜ¹[Block(1,3)] = -Ĝ_uₜ'
            leader_Pₜ¹[Block(1,4)] = -Ĥ_uₜ'
            leader_Pₜ¹[Block(1,5)] = transpose(B̂ₜ)
            leader_Pₜ¹[Block(1,7)] = Π²'
            leader_Pₜ¹[Block(2,2)] = normalization_term*γ̂ₜₖ#/ρ
            leader_Pₜ¹[Block(2,3)] = normalization_term*ŝₜₖ#/ρ
            leader_Pₜ¹[Block(3,1)] = normalization_term*G_uₜ
            leader_Pₜ¹[Block(3,2)] = Matrix(normalization_term)*(-I(2*ll))
            leader_Pₜ¹[Block(4,1)] = H_uₜ
            leader_Pₜ¹[Block(5,1)] = -B
            leader_Pₜ¹[Block(5,8)] = I(nx)
            leader_Pₜ¹[Block(6,5)] = -I(nx*num_player)
            leader_Pₜ¹[Block(6,6)] = Π_next'
            leader_Pₜ¹[Block(6,8)] = Qₜ₊₁ - 
                S̃_uₜ₊₁*K_next + 
                Ĝ_xₜ₊₁'*K_gamma_next +
                Ĥ_xₜ₊₁'*K_mu_next  -
                Âₜ₊₁'*K_lambda_next # double check
            
            # leader_Pₜ¹[Block(6,7)] = γ̂ₜ₊₁ₖ*G_xₜ₊₁ - 
            #     ĝₜ₊₁ₖ*K_gamma_next
            
            leader_Pₜ¹[Block(7,6)] = -I(nu)
            # @infiltrate
            leader_Pₜ¹[Block(7,8)] = S̃ₓ₊₁ - 
                R̃ₜ₊₁*K_next + 
                G̃_uₜ₊₁'*K_gamma_next +
                H̃_uₜ₊₁'*K_mu_next -
                B̃ₜ₊₁'*K_lambda_next # double check
            leader_Pₜ¹[Block(8,1)] = R̃ₜ¹²
            leader_Pₜ¹[Block(8,3)] = -G̃ᵤₜ¹²'
            leader_Pₜ¹[Block(8,4)] = -H̃ᵤₜ¹²'
            leader_Pₜ¹[Block(8,5)] = B̃²'
            leader_Pₜ¹[Block(8,7)] = -I(m)
            if regularization_turn_on
                leader_Pₜ¹ = leader_Pₜ¹+regularization_weight*I(size(Array(leader_Pₜ¹))[1])
            end
            Pₜ¹ = pinv(leader_Pₜ¹)
            
            Pₜ²nₜ₊₁ = Array(-Pₜ¹* [zeros(nu+2*ll+2*ll+2*l+nx, 1); 
                S̃_uₜ₊₁*k_next-Ĝ_xₜ₊₁'*k_gamma_next-Ĥ_xₜ₊₁'*k_mu_next+Âₜ₊₁'*k_lambda_next;  # double check
                R̃ₜ₊₁*k_next-G̃_uₜ₊₁'*k_gamma_next-H̃_uₜ₊₁'*k_mu_next+B̃ₜ₊₁'*k_lambda_next;  # double check
                zeros(m,1) ])
            
            K = -Pₜ¹*[
                [g.S_list[t][1][1:m,:]; g.S_list[t][2][m+1:nu,:]];
                zeros(2*ll,nx);
                normalization_term*G_xₜ; # double check
                H_xₜ; 
                -A; 
                # zeros(new_M_size-nu-4*ll-2*l-nx, nx)] 
                zeros(2*nx+nu, nx);
                Sₜ¹²;
            ]
            # @infiltrate
            k = -Pₜ¹*[
                    rₜ ;#+ Π²'*current_op.ψ[t]; # double check!!!!!!
                    normalization_term*(-ρ*ones(2*ll));#/ρ;#normalization_term*(γ̂ₜₖ*current_op.s[t]-ρ*ones(2*ll));
                    normalization_term*(gₜₖ - current_op.s[t]); 
                    hₜ; 
                    -c; 
                    qₜ₊₁ ;#+ Π_next'*current_op.η[t] + Âₜ₊₁'*current_op.λ[t+1] - Ĝ_xₜ₊₁'*current_op.γ[t+1] - Ĥ_xₜ₊₁'*current_op.μ[t+1] ; # double check!!!!!!
                    # normalization_term_terminal*(γ̂ₜ₊₁ₖ*gₜ₊₁ₖ-ρ*ones(2*ll));
                    g.r_list[t+1][1][m+1:nu];# - current_op.η[t][1:m] ; # zeros(nu+m,1) # double double check!!!!!!
                    g.r_list[t+1][2][1:m];# - current_op.η[t][m+1:nu] ; # double check!!!!!!
                    g.r_list[t][1][m+1:nu];# - current_op.ψ[t]; # double check!!!!!!
                ] - Pₜ²nₜ₊₁ # double check
            π¹_next = K[1:m,:] # update π¹_next
            Π_next[1:m,1:nx] = π̂²
            Π_next[m+1:nu, nx+1:2*nx] = π¹_next # uₜ² = π̂²*xₜ + π̌²*uₜ¹
            
            strategies.P[t] = -K[1:nu, :]
            strategies.α[t] = -k[1:nu]
            
            Aₜ₊₁ = A
            Bₜ₊₁ = B
            K_next = -K[1:nu, :]
            k_next = -k[1:nu, :]
            K_gamma_next = -K[nu+2*ll+1 : nu+4*ll, :]
            k_gamma_next = -k[nu+2*ll+1 : nu+4*ll, :]
            K_mu_next = -K[nu+4*ll+1 : nu+4*ll+2*l, :]
            k_mu_next = -k[nu+4*ll+1 : nu+4*ll+2*l, :]
            K_lambda_next = -K[nu+4*ll+2*l+1 : nu+4*ll+2*l+2*nx,:]
            k_lambda_next = -k[nu+4*ll+2*l+1 : nu+4*ll+2*l+2*nx,:]
            K_multipliers[Block(t,1)] = Array(K[nu+1 : nu+4*ll+2*l+2*nx+nu+m, :])
            k_multipliers[Block(t,1)] = Array(k[nu+1 : nu+4*ll+2*l+2*nx+nu+m, :])



            # in what follows, we construct the KKT matrix for the next time instant. 
            # Related to KKT residual loss
            M_size = M_size + new_M_size
            # M_row_size = M_row_size + new_M_size
            D1 = deepcopy(leader_Pₜ¹)
            D2 = BlockArray(
                zeros(new_M_size, M_size-new_M_size), 
                [nu, 2*ll, 2*ll, 2*l, nx, 2*nx, nu, m], 
                [nu, 2*ll, 2*ll, 2*l, 2*nx, (M_size-new_M_size)-nu-2*ll-2*ll-2*l-2*nx]
            )
            D2[Block(6,3)] = -Ĝ_xₜ₊₁'
            D2[Block(6,4)] = -Ĥ_xₜ₊₁'
            D2[Block(6,5)] = Âₜ₊₁'
            # D2[Block(6,2)] = ĝₜ₊₁ₖ
            D2[Block(7,3)] = -G̃_uₜ₊₁'
            D2[Block(7,4)] = -H̃_uₜ₊₁'
            D2[Block(7,5)] = B̃ₜ₊₁'
            # remove all the second order related matrices:
            if output_ground_truth_KKT_matrices == false
                D1[Block(1,1)] = 0.0*Rₜ # check
                D1[Block(6,8)] = 0.0*Qₜ₊₁ # check 
                D1[Block(8,1)] = 0.0*R̃ₜ¹² # check 
                D1[Block(7,8)] = zeros(nu, nx) # check S̃ₓ₊₁

                D2[Block(6,1)] = zeros(2*nx, nu) # check Sᵤₜ₊₁
                D2[Block(7,1)] = zeros(nu, nu) # check R̃ₜ₊₁

                D1[nu+1:nu+2*ll+2*ll+2*l+nx,:] = D1[nu+1:nu+2*ll+2*ll+2*l+nx,:]*0.0 # ineq + eq + dyn

            end
            KKT_M = [
                D1 D2 ; 
                zeros(M_size - new_M_size, new_M_size-nx)  KKT_N  KKT_M 
            ]
            # construct KKT_N and KKT_n
            KKT_N = [
                [g.S_list[t][1][1:m,:]; g.S_list[t][2][m+1:nu,:]] ;  # check
                zeros(2*ll,nx);
                normalization_term*G_xₜ;
                H_xₜ; 
                -A; 
                # zeros(M_size - nu-2*ll-2*ll-2*l-nx, nx)
                zeros(2*nx+nu, nx);
                Sₜ¹²;
                zeros(M_size - nu-2*ll-2*ll-2*l-nx-2*nx-nu-m, nx)
            ] # remove the second order related matrices
            KKT_n = [rₜ ;#+ Π²'*current_op.ψ[t] ; 
                normalization_term*(-ρ*ones(2*ll));#/ρ;#normalization_term*(γ̂ₜₖ*current_op.s[t]-ρ*ones(2*ll)); 
                normalization_term*(gₜₖ-current_op.s[t]); 
                hₜ; 
                -c; 
                qₜ₊₁ ;#+ Π_next'*current_op.η[t] + Âₜ₊₁'*current_op.λ[t+1] - Ĝ_xₜ₊₁'*current_op.γ[t+1] - Ĥ_xₜ₊₁'*current_op.μ[t+1] ; 
                g.r_list[t+1][1][m+1:nu] ;#- current_op.η[t][1:m] ; # zeros(nu+m,1) # double double check!!!!!!
                g.r_list[t+1][2][1:m] ;#- current_op.η[t][m+1:nu] ;
                g.r_list[t][1][m+1:nu] ;#- current_op.ψ[t] ;
                KKT_n
            ]
            if output_ground_truth_KKT_matrices == false
                # KKT_N[nu+1:nu+2*ll+2*ll+2*l+nx,:] = KKT_N[nu+1:nu+2*ll+2*ll+2*l+nx,:]*0.0 # double check
                KKT_N = KKT_N*0.0
                KKT_n[nu+1:nu+2*ll+2*ll+2*l+nx] = KKT_n[nu+1:nu+2*ll+2*ll+2*l+nx]*0.0
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
                s[t] = K_multipliers[Block(t,1)][1:2*ll,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][1:2*ll]
                
                γ[t] = K_multipliers[Block(t,1)][2*ll+1:4*ll,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][2*ll+1:4*ll]
                
                μ[t] = K_multipliers[Block(t,1)][4*ll+1:4*ll+2*l,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][4*ll+1:4*ll+2*l]
                
                λ[t] = K_multipliers[Block(t,1)][4*ll+2*l+1:4*ll+2*l+2*nx,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][4*ll+2*l+1:4*ll+2*l+2*nx]
            
                ψ[t] = K_multipliers[Block(t,1)][4*ll+2*l+2*nx+1:4*ll+2*l+2*nx+m,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][4*ll+2*l+2*nx+1:4*ll+2*l+2*nx+m]
                
                s[t+1] = K_multipliers[Block(t,1)][end-4*ll-2*l+1:end-2*ll-2*l,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][end-4*ll-2*l+1:end-2*ll-2*l]
                
                γ[t+1] = K_multipliers[Block(t,1)][end-2*ll-2*l+1:end-2*l,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][end-2*ll-2*l+1:end-2*l]
            
                μ[t+1] = K_multipliers[Block(t,1)][end-2*l+1:end,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][end-2*l+1:end]

                Δu[t] = -strategies.P[t]*Δx[t] - strategies.α[t]
                Δx[t+1] = g.A_list[t]*Δx[t] + g.B_list[t]*Δu[t] + g.c_list[t] # update x
            else
                s[t] = K_multipliers[Block(t,1)][1:2*ll,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][1:2*ll]
            
                γ[t] = K_multipliers[Block(t,1)][2*ll+1:4*ll,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][2*ll+1:4*ll]
        
                μ[t] = K_multipliers[Block(t,1)][4*ll+1:4*ll+2*l,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][4*ll+1:4*ll+2*l]
    
                λ[t] = K_multipliers[Block(t,1)][4*ll+2*l+1:4*ll+2*l+2*nx,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][4*ll+2*l+1:4*ll+2*l+2*nx]
        
                η[t] = K_multipliers[Block(t,1)][4*ll+2*l+2*nx+1:4*ll+2*l+2*nx+nu,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][4*ll+2*l+2*nx+1:4*ll+2*l+2*nx+nu]
        
                ψ[t] = K_multipliers[Block(t,1)][end-m+1:end,:]*Δx[t] + 
                    k_multipliers[Block(t,1)][end-m+1:end]
                
                Δu[t] = - strategies.P[t]*Δx[t] - strategies.α[t]
                Δx[t+1] = g.A_list[t]*Δx[t] + g.B_list[t]*Δu[t] + g.c_list[t] # update x
            end
        end
        return Δx, Δu, s, γ, μ, λ, η, ψ, KKT_M, KKT_N, KKT_n
    else
        return KKT_M, KKT_N, KKT_n
    end
end