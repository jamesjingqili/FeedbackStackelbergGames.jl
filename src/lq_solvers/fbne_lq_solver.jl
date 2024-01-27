function fbne_lq_solver!(
    π::strategy, 
    g
    )

    # for more information about the definitions of strategies and LQGame,
    # please refer to Constrained_iLQGames.jl 
    # initializting the optimal cost to go representation for DP
    # quadratic cost to go
    Z = [g.Q_list[end][ii] for ii in 1:g.n_players]
    ζ = [g.q_list[end][ii] for ii in 1:g.n_players]

    # Setup the S and Y matrix of the S * P = Y matrix equation
    S = @MMatrix zeros(g.nu, g.nu)
    YP = @MMatrix zeros(g.nu, g.nx)
    Yα = @MVector zeros(g.nu)
    
    # working backwards in time to solve the dynamic program
    for t in g.horizon:-1:1
        A = g.A_list[t]
        B = g.B_list[t]
        for ii in 1:g.n_players
            udxᵢ = g.players_u_index_list[ii]
            # @infiltrate
            BᵢZᵢ = B[:, udxᵢ]' * Z[ii]
            # the current set of rows that we construct for player ii
            S[udxᵢ, :] = g.R_list[t][ii][udxᵢ, :] + BᵢZᵢ*B
            # append the fully constructed row to the full S-Matrix
            YP[udxᵢ, :] = BᵢZᵢ*A
            Yα[udxᵢ] = B[:, udxᵢ]'*ζ[ii] + g.r_list[t][ii][udxᵢ]
        end

        Sinv = inv(SMatrix(S))
        P = Sinv * SMatrix(YP)
        α = Sinv * SVector(Yα)
        # compute F and β as intermediate result for estimating the cost to go
        F = A - B * P
        β = g.c_list[t] - B * α

        # update Z and ζ (cost to go representation for the next step backwards
        # in time)
        for ii in 1:g.n_players
            PRᵢ = P' * g.R_list[t][ii]
            ζ[ii] = F' * (ζ[ii] + Z[ii] * β) + g.q_list[t][ii] + PRᵢ * α - P' * g.r_list[t][ii]
            Z[ii] = F' * Z[ii] * F + g.Q_list[t][ii] + PRᵢ * P
        end
        π.P[t] = P
        π.α[t] = α
    end
end