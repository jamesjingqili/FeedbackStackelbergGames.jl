using Distributed
using SharedArrays
using Statistics
num_samples = 30;
addprocs(num_samples)

x0 = SharedVector([
    0.9; # x1
    1.2; # y1
    3.5; # v1 1.2
    0.0; # θ1
    0.5; # x2
    0.6; # y2 # 0.4 works well
    3.8; # v2 1.6
    0.0; # θ2
]);
list_loss_list = SharedArray(Array{Float64,3}(undef, num_samples, 10, 10));
list_time_list = SharedArray(Array{Float64,3}(undef, num_samples, 10, 10));
x0_list = SharedArray{Float64}(8,num_samples)
for ii in 1:num_samples
    x0_list[:,ii] = SharedVector(x0 + 0.2*[1,1,1,1,1,1,1,1].*(rand(8).-0.5))
end
# x0_list = [x0 + 0.1*[1;1;1;1;1;1;1;1].*(rand(8).-0.5) for ii in 1:num_samples];

@everywhere begin
using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time
using Infiltrator
using JLD2
using Dates
using Plots
using LaTeXStrings
using SharedArrays
using Distributed
# this file is certified to be correct.

# x0 = [
#     0.9; # x1
#     1.2; # y1
#     3.2; # v1 1.2
#     0.0; # θ1
#     0.5; # x2
#     0.4; # y2
#     3.8; # v2 1.6
#     0.0; # θ2
# ];

# create a list of x0 by adding gaussian noise to x0

horizon = 20;#50;
n_players = 2;
nx = 8;
nu = 4; # sum of dim of each player's control
m = Int(nu/2); # dim of each player's control
players_u_index_list = [1:2, 3:4];

# lower: (x_right_corner, y_right_corner), upper: (base_x, road_length)
#   0.5       1.0
#  | *     \
#  |        \
#  |         \
#  |          \
#  |           \
#  |        x1  |
#  |            |
#  | x2         |
x_right_corner = 1.1; 
y_right_corner = 2.0;
base_x = 0.7; 
road_length = 4.0;
ineq_constraint_eps = 1e-16;
segment_length = sqrt((road_length - y_right_corner)^2 + (x_right_corner - base_x)^2);
segment_angle = atan((x_right_corner - base_x)/(road_length - y_right_corner));
radius = 1/4*segment_length / sin(segment_angle);

upper_circle_x = base_x + radius;
upper_circle_y = road_length;
lower_circle_x = x_right_corner - radius;
lower_circle_y = y_right_corner;

function road_constraints(x,y,road_length)
    if y > road_length
        return base_x - x
    elseif y > y_right_corner 
        angle_to_upper_center = atan((upper_circle_y - y)/(upper_circle_x - x))
        if angle_to_upper_center < 2*segment_angle
            distance_to_upper_center = sqrt((x - upper_circle_x)^2 + (y - upper_circle_y)^2)
            return distance_to_upper_center - radius
        else
            distance_to_lower_center = sqrt((x - lower_circle_x)^2 + (y - lower_circle_y)^2)
            return radius - distance_to_lower_center
        end
    else
        return x_right_corner - x
    end

end

inequality_constraints_size = 7 + 8 - 2;
equality_constraints_size = 2 ;

# initializing the strategy:
π = strategy([zeros(nu,nx) for t in 1:horizon], [zeros(nu) for t in 1:horizon]);



dt = 0.05; #0.02
f_list = [(x,u)-> 
    [
        x[1] + dt*x[3]*sin(x[4]);
        x[2] + dt*x[3]*cos(x[4]);
        x[3] + 1.0*dt*u[1]; # controlling the acceleration
        x[4] + 1.0*dt*u[2]; # controlling the angular velocity
        x[5] + dt*x[7]*sin(x[8]);
        x[6] + dt*x[7]*cos(x[8]);
        x[7] + 1.0*dt*u[3]; # controlling the acceleration
        x[8] + 1.0*dt*u[4]; # controlling the angular velocity
    ]
    for t in 1:horizon
];
costs_list = [
    [
        (x, u) -> (10*(x[1]-0.4)^2 + 6*(x[3]-x[7])^2  + 2*u[1]^2+2*u[2]^2 ) # + 2*x[4]^2, 4 times control cost
        (x, u) -> ( 0*(x[5]-0.5)^2 + 0*(x[3]-x[7])^2 + 1*x[8]^4 + 2*u[3]^2+2*u[4]^2 ) # 2*(x[5]-0.5)^2 +, 4 times control cost
    ] 
    for t in 1:horizon
];
terminal_costs_list = [
    (x) -> (10*(x[1]-0.4)^2 + 6*(x[3]-x[7])^2 ) # + 2*x[4]^2
    (x) -> ( 0*(x[5]-0.5)^2 + 0*(x[3]-x[7])^2 + 1*x[8]^4)
];
equality_constraints_list = [
    [
        (z) -> [0.0;   0.0;]#; 0.0; 0.0] 0.0;
        (z) -> [0.0;   0.0;]#; 0.0; 0.0] 0.0;
    ] 
    for t in 1:horizon
];
terminal_equality_constraints_list = [
    (x) -> [x[4];    x[3]-x[7];]#; x[5]-0.5]#*0.0 x[1] - 0.5;  x[5] - 0.5;  x[3]-x[7];
    (x) -> [x[4];    x[3]-x[7];]#; x[5]-0.5]#*0.0 x[1] - 0.5;  x[5] - 0.5;  x[3]-x[7];
];
collision_avoidance(x1,y1,x2,y2) = 2*((x1-x2)^2 + (y1-y2)^2 - 0.4^2); # 0.25 works well
max_a = 0.5*2;
min_a = -0.5*2;
max_w = 2.0; # 1.0 works well
min_w = -2.0;
inequality_constraints_list = [
    [
        (z) -> [
            z[1] - 0.25;
            z[5] - 0.25;
            road_constraints(z[1],z[2],road_length);
            road_constraints(z[5],z[6],road_length);
            collision_avoidance(z[1],z[2],z[5],z[6])*1.0;
            max_a - z[8+1];
            z[8+1] - (min_a);
            max_w - z[8+2];
            z[8+2] - (min_w);
            max_a - z[8+3];
            z[8+3] - (min_a);
            max_w - z[8+4];
            z[8+4] - (min_w);
        ]
        for ii in 1:n_players
    ] 
    for t in 1:horizon
];
terminal_inequality_constraints_list = [
    (x) -> [
        x[1] - 0.25;
        x[5] - 0.25;
        # x_right_corner - x[1];
        # x_right_corner - x[5];
        road_constraints(x[1],x[2],road_length);
        road_constraints(x[5],x[6],road_length);
        collision_avoidance(x[1],x[2],x[5],x[6])*1.0;
        5; # TODO: check that if it's ok to set trivial inequality constraints in this way
        5;
        5;
        5;
        5;
        5;
        5;
        5;
    ]
    for ii in 1:n_players 
];


# global Mₜ, Nₜ, nₜ, Δ, α, ρ, new_loss, next_op, current_op, lq_approx, early_stop_counter, calibrated_γ, calibrated_s, scale_s
# list_loss_list = [[] for ii in 1:num_samples];
end



@distributed for sample_iter = 1:num_samples
    # global Mₜ, Nₜ, nₜ, Δ, α, ρ, new_loss, next_op, current_op, lq_approx, early_stop_counter, calibrated_γ, calibrated_s, scale_s, list_loss_list
    g = Constrained_LQGame(
        horizon,
        n_players,
        nx,
        nu,
        players_u_index_list,
        [Matrix(1.0*I(nx)) for t in 1:horizon], # A_list
        [zeros(nx, nu) for t in 1:horizon], # B_list
        [zeros(nx) for t in 1:horizon], # c_list
        [[Matrix(1.0*I(nx)) for ii in 1:n_players] for t in 1:horizon+1], # Q_list
        [[zeros(nu, nx) for ii in 1:n_players] for t in 1:horizon], # S_list
        [[Matrix(1.0*I(nu)) for ii in 1:n_players] for t in 1:horizon], # R_list
        [[zeros(nx) for ii in 1:n_players] for t in 1:horizon+1], # q_list
        [[zeros(nu) for ii in 1:n_players] for t in 1:horizon], # r_list
        equality_constraints_size,
        [[zeros(equality_constraints_size, nx) for ii in 1:n_players] for t in 1:horizon], # Hx_list
        [[zeros(equality_constraints_size, nu) for ii in 1:n_players] for t in 1:horizon], # Hu_list
        [[zeros(equality_constraints_size) for ii in 1:n_players] for t in 1:horizon], # h_list
        [zeros(equality_constraints_size, nx) for ii in 1:n_players], # HxT
        [zeros(equality_constraints_size) for ii in 1:n_players], # hxT
        inequality_constraints_size,
        [[zeros(inequality_constraints_size, nx) for ii in 1:n_players] for t in 1:horizon], # Gx_list
        [[zeros(inequality_constraints_size, nu) for ii in 1:n_players] for t in 1:horizon], # Gu_list
        [[zeros(inequality_constraints_size) for ii in 1:n_players] for t in 1:horizon], # g_list
        [zeros(inequality_constraints_size, nx) for ii in 1:n_players], # GxT
        [zeros(inequality_constraints_size) for ii in 1:n_players], # gxT
        x0_list[:,sample_iter]
    );

    # define a general game:
    nonlinear_g = game(
        horizon = horizon,
        n_players = n_players,
        nx = nx,
        nu = nu,
        players_u_index_list = players_u_index_list,
        f_list = f_list,
        costs_list = costs_list,
        terminal_costs_list = terminal_costs_list,
        x0 = x0_list[:,sample_iter],
        equality_constraints_list = equality_constraints_list,
        terminal_equality_constraints_list = terminal_equality_constraints_list,
        inequality_constraints_list = inequality_constraints_list,
        terminal_inequality_constraints_list = terminal_inequality_constraints_list,
        equality_constraints_size = equality_constraints_size,
        inequality_constraints_size = inequality_constraints_size
    );

    # define the current_op
    ρ₀ = 1/1.0;
    # TODO: we should think about initializing γ by considering γ .* inequality_constraints_list = ρ
    initial_op = trajectory(
        x = [x0_list[:,sample_iter] for t in 1:horizon+1],
        u = [zeros(nu) for t in 1:horizon],
        λ = [zeros(nx*n_players) for t in 1:horizon],
        η = [zeros(nu) for t in 1:horizon-1],
        ψ = [zeros(m) for t in 1:horizon],
        μ = [zeros(equality_constraints_size*n_players) for t in 1:horizon+1],
        γ = [5*ones(inequality_constraints_size*n_players) for t in 1:horizon+1],
        s = [5*ones(inequality_constraints_size*n_players) for t in 1:horizon+1]
    );
    forward_simulation!(initial_op, nonlinear_g);
    calibrated_γ = initial_op.γ;
    calibrated_s = initial_op.s;
    current_op = deepcopy(initial_op);

    # initializing the LQ approximation:
    lq_approx = deepcopy(g);

    # we linearize the dynamics of the nonlinear game around the current operating point!
    lq_approximation!(lq_approx, nonlinear_g, current_op);



    # initialize the lq approximation and all the solution variables
    x = current_op.x;
    u = current_op.u;
    γ = current_op.γ;
    μ = current_op.μ;
    λ = current_op.λ;
    s = current_op.s;
    η = current_op.η;
    ψ = current_op.ψ;
    Mₜ = [zeros(nx, nx) for t in 1:horizon];
    Nₜ = [zeros(nu, nx) for t in 1:horizon];
    nₜ = [zeros(nx) for t in 1:horizon];




    # we do an outer iteration, where we periodically update ρ
    # in inner iterations, we fix ρ and update the linear policy till convergence
    inner_convergence_tol = 1e-6;

    σ = 2;
    num_outer_iter = 10;
    num_iter = 10;
    ρ = ρ₀;
    ρ_list = [ρ/σ^(ii-1) for ii in 1:num_outer_iter];

    # ρ_list = range(1.0, 0.01, length=num_outer_iter);


    # loss_list = [zeros(num_iter) for ii in 1:num_outer_iter];
    # loss_list = [[] for ii in 1:num_outer_iter];
    loss_list = zeros(num_outer_iter, num_iter);
    time_list = zeros(num_outer_iter, num_iter);
    # homotopy_loss = [zeros(num_iter) for ii in 1:num_outer_iter];
    homotopy_loss = [[] for ii in 1:num_outer_iter];
    # α_list = [zeros(num_iter) for ii in 1:num_outer_iter];
    α_list = [[] for ii in 1:num_outer_iter];
    # α_γ_list = [zeros(num_iter) for ii in 1:num_outer_iter];
    α_γ_list = [[] for ii in 1:num_outer_iter];
    # min_s_list = [zeros(num_iter) for ii in 1:num_outer_iter];
    min_s_list = [[] for ii in 1:num_outer_iter];
    # min_γ_list = [zeros(num_iter) for ii in 1:num_outer_iter];
    min_γ_list = [[] for ii in 1:num_outer_iter];
    # min_ineq_list = [zeros(num_iter) for ii in 1:num_outer_iter];
    min_ineq_list = [[] for ii in 1:num_outer_iter];
    current_op_list = [[] for ii in 1:num_outer_iter];
    Δ_list = [[] for ii in 1:num_outer_iter];
    M_list = [[] for ii in 1:num_outer_iter];
    N_list = [[] for ii in 1:num_outer_iter];
    n_list = [[] for ii in 1:num_outer_iter];


    early_stop_counter = 0;
    maximum_allowed_γ = 1e4;
    scale_s = 1.0;
    lq_approximation!(lq_approx, nonlinear_g, current_op);
    for outer_iter = 1:num_outer_iter
        # global x, u, s, γ, μ, λ, η, ψ, Mₜ, Nₜ, nₜ, Δ, α, ρ, new_loss, next_op, current_op, lq_approx, early_stop_counter, calibrated_γ, calibrated_s
        # global Mₜ, Nₜ, nₜ, Δ, α, ρ, new_loss, next_op, current_op, lq_approx, early_stop_counter, calibrated_γ, calibrated_s, scale_s
        # if outer_iter == 1
        #     ρ_list[outer_iter] = ρ;
        # else
        #     ρ_list[outer_iter] = ρ_list[outer_iter-1]*1/σ;
        # end
        # tmp_s = current_op.s
        # for t = 1:horizon+1
        #     # tmp_s[t] = max.(tmp_s[t], 10.0*ones(2*inequality_constraints_size))
        #     calibrated_γ[t] = diagm(current_op.s[t])^(-1)*(ρ_list[outer_iter].*ones(2*inequality_constraints_size));
        #     calibrated_γ[t] = min.(calibrated_γ[t], maximum_allowed_γ);
        # end
        current_op = trajectory(
            x = current_op.x,
            u = current_op.u,
            λ = current_op.λ,
            η = current_op.η,
            ψ = current_op.ψ,
            μ = current_op.μ,
            # γ = calibrated_γ*1,
            γ = current_op.γ*scale_s/2,
            s = current_op.s*scale_s #+ [ones(inequality_constraints_size*n_players) for t in 1:horizon+1]
            # s = tmp_s
        );
        regularization_turn_on = false;
        lq_approximation!(lq_approx, nonlinear_g, current_op);
        println("outer iter, ρ = ", ρ_list[outer_iter])

        for iter in 1:num_iter
            time = @elapsed begin
                # lq_approximation!(lq_approx, nonlinear_g, current_op)
                Δx, Δu, Δs, Δγ, Δμ, Δλ, Δη, Δψ, Mₜ, Nₜ, nₜ = nw_pdip_fbst_lq_solver!(
                    π, 
                    lq_approx, 
                    current_op, 
                    ρ_list[outer_iter], 
                    false, 
                    false,
                    false,
                    true,
                    true,
                    regularization_turn_on
                )
                Δ = trajectory(
                    x = Δx,
                    u = Δu,
                    λ = Δλ,
                    η = Δη,
                    ψ = Δψ,
                    μ = Δμ,
                    γ = Δγ,
                    s = Δs
                );
                
                # Δ_list[outer_iter] = [Δ_list[outer_iter]; Δ];
                # M_list[outer_iter] = [M_list[outer_iter]; [Mₜ]];
                # N_list[outer_iter] = [N_list[outer_iter]; [Nₜ]];
                # n_list[outer_iter] = [n_list[outer_iter]; [nₜ]];
                current_op_list[outer_iter] = [current_op_list[outer_iter]; current_op];

                α, new_loss, next_op, lq_approx, min_s, min_γ, new_homotopy_loss, α_γ, min_ineq = nw_pdip_line_search(
                    π,
                    Δ,
                    current_op, 
                    lq_approx,
                    nonlinear_g,
                    ρ_list[outer_iter],
                    1.0, # initial step size
                    0.5 # step size reduction factor
                );
                # @infiltrate

                current_op = next_op;
                # loss_list[outer_iter][iter] = new_loss;
                loss_list[outer_iter, iter] = new_loss; # changed the way to store loss
                # homotopy_loss[outer_iter][iter] = new_homotopy_loss;
                homotopy_loss[outer_iter] = [homotopy_loss[outer_iter]; new_homotopy_loss];
                # α_list[outer_iter][iter] = α;
                α_list[outer_iter] = [α_list[outer_iter]; α];
                # α_γ_list[outer_iter][iter] = α_γ;
                α_γ_list[outer_iter] = [α_γ_list[outer_iter]; α_γ];
                # min_s_list[outer_iter][iter] = min_s;
                min_s_list[outer_iter] = [min_s_list[outer_iter]; min_s];
                # min_γ_list[outer_iter][iter] = min_γ; 
                min_γ_list[outer_iter] = [min_γ_list[outer_iter]; min_γ];
                # min_ineq_list[outer_iter][iter] = min_ineq;
                min_ineq_list[outer_iter] = [min_ineq_list[outer_iter]; min_ineq];
                println("iter = $iter, loss = $new_loss, homotopy loss = $new_homotopy_loss, α = $α, min_s = $min_s, min_γ = $min_γ, min_ineq = $min_ineq")
                if α == 0.0 && regularization_turn_on == false
                    break
                    regularization_turn_on = true;
                    scale_s = 10.0;
                    println("turn on regularization")
                end
                # if α == 0.0 || loss_list[outer_iter][iter] < inner_convergence_tol
                #     break
                # #     current_op = trajectory(
                # #         x = next_op.x + [0.01*randn(nx) for t in 1:horizon+1],
                # #         u = next_op.u + [0.01*randn(nu) for t in 1:horizon],
                # #         λ = next_op.λ + [0.01*randn(nx*n_players) for t in 1:horizon],
                # #         η = next_op.η + [0.01*randn(nu) for t in 1:horizon-1],
                # #         ψ = next_op.ψ + [0.01*randn(m) for t in 1:horizon],
                # #         μ = next_op.μ + [0.01*randn(equality_constraints_size*n_players) for t in 1:horizon+1],
                # #         γ = next_op.γ + [0.01*randn(inequality_constraints_size*n_players) for t in 1:horizon+1],
                # #         s = next_op.s + [0.01*randn(inequality_constraints_size*n_players) for t in 1:horizon+1]
                # #     );
                # end

                # if loss_list[outer_iter][iter] < 1e-8
                #     break
                # end
            end
        time_list[outer_iter, iter] = time;
        end
    end
    # append loss_list to list_loss_list
    list_loss_list[sample_iter, :,:] = SharedArray{Float64}(loss_list);
    list_time_list[sample_iter, :,:] = SharedArray{Float64}(time_list);
    # loss_list

end



