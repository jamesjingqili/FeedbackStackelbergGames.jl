module iLQGames_discrete_time
    using LinearAlgebra
    using Infiltrator
    using StaticArrays
    using ForwardDiff
    using BlockArrays
    # Overview: we define the following 5 structs:
    # for list, the first coordinate is time, the second coordinate is player


    # we define a struct for the linear-quadratic game:
    # note that a linear quadratic approximation of a nonlinear will
    # be a linear-quadratic game

    # We assume that each player has the same dimension of the state, control, and constraints.
    
    Base.@kwdef struct Constrained_LQGame
        horizon::Int 
        n_players::Int
        nx::Int
        nu::Int 
        players_u_index_list::Vector 
        A_list::Vector = [zeros(nx, nx) for t in 1:horizon]
        B_list::Vector = [zeros(nx, nu) for t in 1:horizon]
        c_list::Vector = [zeros(nx) for t in 1:horizon]
        Q_list::Vector = [[zeros(nx, nx) for ii in 1:n_players] for t in 1:horizon+1]
        S_list::Vector = [[zeros(nu, nx) for ii in 1:n_players] for t in 1:horizon]
        R_list::Vector = [[zeros(nu, nu) for ii in 1:n_players] for t in 1:horizon]
        q_list::Vector = [[zeros(nx) for ii in 1:n_players] for t in 1:horizon+1]
        r_list::Vector = [[zeros(nu) for ii in 1:n_players] for t in 1:horizon]
        equality_constraints_size::Int = 1
        Hx_list::Vector = [[zeros(equality_constraints_size, nx) for ii in 1:num_players] for t in 1:horizon]
        Hu_list::Vector = [[zeros(equality_constraints_size, nu) for ii in 1:num_players] for t in 1:horizon]
        h_list::Vector = [[zeros(equality_constraints_size) for ii in 1:num_players] for t in 1:horizon]
        HxT::Any = [zeros(equality_constraints_size, nx) for ii in 1:num_players]
        hxT::Any = [zeros(equality_constraints_size) for ii in 1:num_players]
        inequality_constraints_size::Int = 1
        Gx_list::Vector = [[zeros(inequality_constraints_size, nx) for ii in 1:num_players] for t in 1:horizon]
        Gu_list::Vector = [[zeros(inequality_constraints_size, nu) for ii in 1:num_players] for t in 1:horizon]
        g_list::Vector = [[zeros(inequality_constraints_size) for ii in 1:num_players] for t in 1:horizon]
        GxT::Any = [zeros(inequality_constraints_size, nx) for ii in 1:num_players]
        gxT::Any = [zeros(inequality_constraints_size) for ii in 1:num_players]
        x0::Vector = zeros(nx)
    end
    
    Base.@kwdef struct LQGame
        horizon::Int 
        n_players::Int
        nx::Int
        nu::Int
        players_u_index_list::Vector
        A_list::Vector = [zeros(nx, nx) for t in 1:horizon]
        B_list::Vector = [zeros(nx, nu) for t in 1:horizon]
        c_list::Vector = [zeros(nx) for t in 1:horizon]
        Q_list::Vector = [[zeros(nx, nx) for ii in 1:n_players] for t in 1:horizon+1]
        S_list::Vector = [[zeros(nu, nx) for ii in 1:n_players] for t in 1:horizon]
        R_list::Vector = [[zeros(nu, nu) for ii in 1:n_players] for t in 1:horizon]
        q_list::Vector = [[zeros(nx) for ii in 1:n_players] for t in 1:horizon+1]
        r_list::Vector = [[zeros(nu) for ii in 1:n_players] for t in 1:horizon]
        x0::Vector = zeros(nx)
    end


    # we define a struct for the nonlinear game:
    # note that we define f to be time-varying, and 
    # therfore f is a list of functions at each time step
    Base.@kwdef struct game
        horizon::Int 
        n_players::Int
        nx::Int
        nu::Int
        players_u_index_list::Vector 
        f_list::Vector # dynamics model
        costs_list::Vector
        terminal_costs_list::Vector
        x0::Vector
        equality_constraints_list::Vector
        terminal_equality_constraints_list::Vector
        inequality_constraints_list::Vector # G(x,u) <= 0
        terminal_inequality_constraints_list::Vector
        # this conbines [x,u] as joint input to f:
        f_for_jacobian_and_hessian_evaluation::Vector = [z -> f_list[t](z[1:nx], z[nx+1:end])
                                                            for t in 1:horizon ] 
        # this conbines [x,u] as joint input to costs:
        costs_for_jacobian_and_hessian_evaluation::Vector = [[z -> costs_list[t][ii](z[1:nx], z[nx+1:end]) 
                                                for ii in 1:n_players] for t in 1:horizon] 
        equality_constraints_size::Int = 1
        inequality_constraints_size::Int = 1
    end

    struct strategy
        P::Vector
        α::Vector
    end

    Base.@kwdef struct trajectory
        x::Vector
        u::Vector
        # Lagrange multiplier for the dynamics:
        λ::Vector 
        # Lagrange multiplier for the strategy:
        η::Vector 
        # Lagrange multiplier for the strategy of the follower player in Stackelberg game:
        ψ::Vector = []
        # Lagrange multiplier for the equality constraints:
        μ::Vector = []
        # Lagrange multiplier for the inequality constraints:
        γ::Vector = []
        # slackness variables for the inequality constraints:
        s::Vector = []
    end

    # define the struct for the iLQ solver, which stores solver parameters
    struct iLQsolver
        max_iter::Int
        tol::Float64
        g::game
        x0::Vector
        verbose::Bool
    end
    include("lq_solvers/fbne_lq_solver.jl")
    include("lq_solvers/fbst_lq_solver.jl")
    include("lq_solvers/constrained_fbst_lq_solver.jl")
    include("lq_solvers/pdip_fbst_lq_solver.jl")
    include("lq_solvers/nw_pdip_fbst_lq_solver.jl")
    include("nw_pdip_line_search.jl")
    include("lq_approximation.jl")
    include("utils.jl")
    include("line_search.jl")
    include("pdip_line_search.jl")
    include("ilq_solver.jl")
    include("forward_simulation.jl")
    include("backward_pass.jl")
    # include("lq_solvers/ground_truth_fbst.jl")
    greet() = print("Hello World!")
    export greet
    export LQGame
    export Constrained_LQGame
    export strategy
    export game
    export trajectory
    export fbne_lq_solver!
    export fbst_lq_solver!
    export pdip_fbst_lq_solver!
    export nw_pdip_fbst_lq_solver!
    export nw_pdip_KKT_residual_fbst
    export nw_pdip_line_search
    export fast_nw_pdip_line_search
    export pdip_line_search
    # export ground_truth_fbst!
    export constrained_fbst_lq_solver!
    export initialization_of_π
    export lq_approximation!
    export forward_simulation!
    export backward_pass!
    export solve!
    export line_search!
    export KKT_residual_fbst
    export relu
end

# TODO: 1. ForwardDiff + lq_approx; 2. KKT residual -> line search



