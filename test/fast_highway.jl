using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time
using Infiltrator
using JLD2
using Dates
using Plots

# this file is certified to be correct.

x0 = [
    0.9; # x1
    1.2; # y1
    1.2; # v1
    0.0; # θ1
    0.5; # x2
    0.4; # y2
    1.6; # v2
    0.0; # θ2
];
horizon = 100;
n_players = 2;
nx = 8;
nu = 4; # sum of dim of each player's control
m = Int(nu/2); # dim of each player's control
players_u_index_list = [1:2, 3:4];

# (x_right_corner, y_right_corner), (base_x, road_length)
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
function road_constraints(x,y,road_length)
    if y < road_length
        return road_length - y - (road_length - y_right_corner)/(x_right_corner - base_x)*(x - base_x)
    else
        return base_x - x
    end
end
# function road_constraints(x,y,road_length)
#     if x > base_x + ineq_constraint_eps
#         # return ineq_constraint_eps
#         return road_length - y - (road_length - y)/(1.5 - base_x)*(x - base_x)
#     else
#         if y > road_length
#             return base_x - x
#         else
#             return ineq_constraint_eps
#         end
#     end
# end

# equality constraint: x1 change lane successfully, can be dropped if hard to solve
# inequality constraint: x1 and x2 avoid collision 
# inequality constraint: x1 stays in lane
# inequality constraint: x2 stays in lane
# we can define lane to be a function of x1 and x2. For example, 
# x1,x2 > 0.0, 
# (y1-2*x1)<-2 if y1<4 else x1<1.0
# (y2-2*x2)<-2 if y2<4 else x2<1.0

inequality_constraints_size = 7;
equality_constraints_size = 3;

# initializing the strategy:
π = strategy([zeros(nu,nx) for t in 1:horizon], [zeros(nu) for t in 1:horizon]);

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
    x0
);


dt = 0.02;
f_list = [(x,u)-> 
    [
        x[1] + dt*x[3]*sin(x[4]);
        x[2] + dt*x[3]*cos(x[4]);
        x[3] + dt*u[1];
        x[4] + 1.0*dt*u[2];
        x[5] + dt*x[7]*sin(x[8]);
        x[6] + dt*x[7]*cos(x[8]);
        x[7] + dt*u[3];
        x[8] + 1.0*dt*u[4];
    ]
    for t in 1:horizon
];

costs_list = [
    [
        (x, u) -> (4*(x[1]-0.5)^2 + (x[3]-x[7])^2 + u[1]^2+u[2]^2 ) # + 2*x[4]^2
        (x, u) -> (2*(x[5]-0.5)^2 + 0*(x[7]-x0[7])^2 + 2*x[8]^2 + u[3]^2+u[4]^2 )
    ] 
    for t in 1:horizon
];
terminal_costs_list = [
    (x) -> (4*(x[1]-0.5)^2 + (x[3]-x[7])^2) #  + 2*x[4]^2
    (x) -> (2*(x[5]-0.5)^2 + 0*(x[7]-x0[7])^2 + 2*x[8]^2)
];
equality_constraints_list = [
    [
        (z) -> [0.0; 0.0; 0.0] #  0.0; 0.0
        (z) -> [0.0; 0.0; 0.0] #  0.0; 0.0
    ] 
    for t in 1:horizon
];
terminal_equality_constraints_list = [
    (x) -> [ x[4];  x[8]; x[3]-x[7]]#*0.0 x[1]-0.5; x[5]-0.5;
    (x) -> [ x[4];  x[8]; x[3]-x[7]]#*0.0 x[1]-0.5; x[5]-0.5;
];



collision_avoidance(x1,y1,x2,y2) = sqrt((x1-x2)^2 + (y1-y2)^2) - 0.5;

inequality_constraints_list = [
    [
        (z) -> [
            z[1] - 0.25;
            z[5] - 0.25;
            x_right_corner - z[1];
            x_right_corner - z[5];
            road_constraints(z[1],z[2],road_length);
            road_constraints(z[5],z[6],road_length);
            collision_avoidance(z[1],z[2],z[5],z[6])*1.0;
        ]
        for ii in 1:n_players
    ] 
    for t in 1:horizon
];
terminal_inequality_constraints_list = [
    (x) -> [
        x[1] - 0.25;
        x[5] - 0.25;
        x_right_corner - x[1];
        x_right_corner - x[5];
        road_constraints(x[1],x[2],road_length);
        road_constraints(x[5],x[6],road_length);
        collision_avoidance(x[1],x[2],x[5],x[6])*1.0;
    ]
    for ii in 1:n_players 
];



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
    x0 = x0,
    equality_constraints_list = equality_constraints_list,
    terminal_equality_constraints_list = terminal_equality_constraints_list,
    inequality_constraints_list = inequality_constraints_list,
    terminal_inequality_constraints_list = terminal_inequality_constraints_list,
    equality_constraints_size = equality_constraints_size,
    inequality_constraints_size = inequality_constraints_size
);




# define the current_op
ρ₀ = 1/4.0;
# TODO: we should think about initializing γ by considering γ .* inequality_constraints_list = ρ
initial_op = trajectory(
    x = [x0 for t in 1:horizon+1],
    u = [zeros(nu) for t in 1:horizon],
    λ = [zeros(nx*n_players) for t in 1:horizon],
    η = [zeros(nu) for t in 1:horizon-1],
    ψ = [zeros(m) for t in 1:horizon],
    μ = [zeros(equality_constraints_size*n_players) for t in 1:horizon+1],
    γ = [10*ones(inequality_constraints_size*n_players) for t in 1:horizon+1],
    s = [10*ones(inequality_constraints_size*n_players) for t in 1:horizon+1]
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

σ = 4.0;
num_outer_iter = 10;
num_iter = 40;
ρ = ρ₀;
ρ_list = [ρ for ii in 1:num_outer_iter];

loss_list = [zeros(num_iter) for ii in 1:num_outer_iter];
homotopy_loss = [zeros(num_iter) for ii in 1:num_outer_iter];
α_list = [zeros(num_iter) for ii in 1:num_outer_iter];
α_γ_list = [zeros(num_iter) for ii in 1:num_outer_iter];
min_s_list = [zeros(num_iter) for ii in 1:num_outer_iter];
min_γ_list = [zeros(num_iter) for ii in 1:num_outer_iter];
min_ineq_list = [zeros(num_iter) for ii in 1:num_outer_iter];
current_op_list = [[] for ii in 1:num_outer_iter];
Δ_list = [[] for ii in 1:num_outer_iter];
M_list = [[] for ii in 1:num_outer_iter];
N_list = [[] for ii in 1:num_outer_iter];
n_list = [[] for ii in 1:num_outer_iter];


early_stop_counter = 0;
maximum_allowed_γ = 1e4;

lq_approximation!(lq_approx, nonlinear_g, current_op);
for outer_iter = 1:num_outer_iter
    # global x, u, s, γ, μ, λ, η, ψ, Mₜ, Nₜ, nₜ, Δ, α, ρ, new_loss, next_op, current_op, lq_approx, early_stop_counter, calibrated_γ, calibrated_s
    global Mₜ, Nₜ, nₜ, Δ, α, ρ, new_loss, next_op, current_op, lq_approx, early_stop_counter, calibrated_γ, calibrated_s
    if outer_iter == 1
        ρ_list[outer_iter] = ρ;
    else
        ρ_list[outer_iter] = ρ_list[outer_iter-1]*1/σ;
    end
    
    for t = 1:horizon+1
        calibrated_γ[t] = diagm(current_op.s[t])^(-1)*(ρ_list[outer_iter].*ones(2*inequality_constraints_size));
        calibrated_γ[t] = min.(calibrated_γ[t], maximum_allowed_γ);
    end
    current_op = trajectory(
        x = current_op.x,
        u = current_op.u,
        λ = current_op.λ,
        η = current_op.η,
        ψ = current_op.ψ,
        μ = current_op.μ,
        γ = calibrated_γ,
        # γ = current_op.γ,
        s = current_op.s
    );
    # lq_approximation!(lq_approx, nonlinear_g, current_op); # this is moved to the beginning of the inner loop!
    println("outer iter, ρ = ", ρ_list[outer_iter])
    for iter in 1:num_iter
        lq_approximation!(lq_approx, nonlinear_g, current_op) # Double check!

        Δx, Δu, Δs, Δγ, Δμ, Δλ, Δη, Δψ, Mₜ, Nₜ, nₜ = nw_pdip_fbst_lq_solver!(
            π, 
            lq_approx, 
            current_op, 
            ρ_list[outer_iter], 
            false, 
            false
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

        α, new_loss, next_op, lq_approx, min_s, min_γ, new_homotopy_loss, α_γ, min_ineq = fast_nw_pdip_line_search(
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
        loss_list[outer_iter][iter] = new_loss;
        homotopy_loss[outer_iter][iter] = new_homotopy_loss;
        α_list[outer_iter][iter] = α;
        α_γ_list[outer_iter][iter] = α_γ;
        min_s_list[outer_iter][iter] = min_s;
        min_γ_list[outer_iter][iter] = min_γ;
        min_ineq_list[outer_iter][iter] = min_ineq;
        println("loss = $new_loss, homotopy loss = $new_homotopy_loss, α = $α, min_s = $min_s, min_γ = $min_γ, min_ineq = $min_ineq")
        if α == 0.0
            break
        end
        if loss_list[outer_iter][iter] < 1e-8
            break
        end
    end
end
loss_list





now_str = Dates.format(now(), "yyyy-mm-dd-HH:MM:SS")
marker_size_list = 6*[0.98^(horizon+1-t) for t in 1:horizon+1]
alpha_list = [0.98^(horizon+1-t) for t in 1:horizon];
x_st = current_op_list[end][end].x;

x_inclined_lines = [base_x, x_right_corner]
y_inclined_lines = [road_length, y_right_corner]
x_upper_right_lines = [base_x, base_x]
y_upper_right_lines = [road_length, 5.0]
x_lower_right_lines = [x_right_corner, x_right_corner]
y_lower_right_lines = [y_right_corner, 0.0]
x_left_lines = [0.25, 0.25]
y_left_lines = [0.0, 5.0]
scatter([x_st[t][1] for t in 1:horizon], [x_st[t][2] for t in 1:horizon],markershape=:circle, color=:red,label="", alpha=alpha_list)
scatter!([x_st[t][5] for t in 1:horizon], [x_st[t][6] for t in 1:horizon],markershape=:circle,color=:blue,label="", alpha=alpha_list)
scatter!([x_st[end][1]], [x_st[end][2]],markershape=:circle, label="player 1", color = :red)
scatter!([x_st[end][5]], [x_st[end][6]],markershape=:circle, label="player 2", color = :blue)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "road edge", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = (300, 600))
# scatter!([x_ne[t][1] for t in 1:horizon+1], [x_ne[t][2] for t in 1:horizon+1],markershape=:circle, label="player 1 fbne", markersize=marker_size_list)
# scatter!([x_ne[t][5] for t in 1:horizon+1], [x_ne[t][6] for t in 1:horizon+1],markershape=:circle, label="player 2 fbne", markersize=marker_size_list)
# title!("J1_st = $total_cost_J1_st, J2_st = $total_cost_J2_st, J1_ne = $total_cost_J1_ne, J2_ne = $total_cost_J2_ne")
savefig("fast_highway_st_$now_str.png")


# animation for fbst:

anim1 = @animate for t = 1:horizon+1
    scatter([x_st[t][1] for t in 1:t], [x_st[t][2] for t in 1:t],markershape=:circle, label="player 1 fbst", markersize=marker_size_list[1:t])
    scatter!([x_st[t][5] for t in 1:t], [x_st[t][6] for t in 1:t],markershape=:circle, label="player 2 fbst", markersize=marker_size_list[1:t])
    xlims!(0.0, 1.5)
    ylims!(0.0, 5.0)
    plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
    plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
    plot!(x_left_lines, y_left_lines, label = "road edge", color = :black, linewidth = 2, linestyle = :solid)
    plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
    title!("t = $t")
end
gif(anim1, "fast_highway_st_$now_str.gif", fps = 10)



# for storing data:


# create some data to save
data = Dict(
    "loss_list" => loss_list,
    "homotopy_loss" => homotopy_loss,
    "α_list" => α_list,
    "α_γ_list" => α_γ_list,
    "min_s_list" => min_s_list,
    "min_γ_list" => min_γ_list,
    "min_ineq_list" => min_ineq_list,
    "current_op_list" => current_op_list,
    "nonlinear_g" => nonlinear_g,
    "lq_approx" => lq_approx,
    "x0" => x0,
    "road_length" => road_length,
    "base_x" => base_x,
    "ineq_constraint_eps" => ineq_constraint_eps,
    "ρ_list" => ρ_list
)
# save the data to a JLD2 file
save("fast_highway_$now_str.jld2", "data", data)



