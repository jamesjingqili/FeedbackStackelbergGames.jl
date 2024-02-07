using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time
using Infiltrator
using JLD2
using Dates
using Plots
using LaTeXStrings
# this file is certified to be correct.

using FilePathsBase

ρ₀ = 1; # initial penalty parameter
num_iter = 10; # number of iterations for each outer iteration
num_outer_iter = 1; # number of outer iterations

# initial state:
x0 = [
    0.9; # x1
    1.2; # y1
    3.5; # v1 1.2
    0.0; # θ1
    0.5; # x2
    0.6; # y2 # 0.4 works well
    3.8; # v2 1.6
    0.0; # θ2
];
horizon = 20; # 50
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


dt = 0.05;# 0.02
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



# collision_avoidance(x1,y1,x2,y2) = sqrt((x1-x2)^2 + (y1-y2)^2) - 0.4;
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
            # x_right_corner - z[1];
            # x_right_corner - z[5];
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
# ρ₀ =  1/2^12 # 1/1.0;
# num_iter = 20;


# TODO: we should think about initializing γ by considering γ .* inequality_constraints_list = ρ
initial_op = trajectory(
    x = [x0 for t in 1:horizon+1],
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
ρ = ρ₀;
ρ_list = [ρ/σ^(ii-1) for ii in 1:num_outer_iter];

# ρ_list = range(1.0, 0.01, length=num_outer_iter);


# loss_list = [zeros(num_iter) for ii in 1:num_outer_iter];
loss_list = [[] for ii in 1:num_outer_iter];
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
@time for outer_iter = 1:num_outer_iter
    # global x, u, s, γ, μ, λ, η, ψ, Mₜ, Nₜ, nₜ, Δ, α, ρ, new_loss, next_op, current_op, lq_approx, early_stop_counter, calibrated_γ, calibrated_s
    global Mₜ, Nₜ, nₜ, Δ, α, ρ, new_loss, next_op, current_op, lq_approx, early_stop_counter, calibrated_γ, calibrated_s, scale_s
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
        loss_list[outer_iter] = [loss_list[outer_iter]; new_loss]; # changed the way to store loss
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
            regularization_turn_on = true;
            scale_s = 10.0;
            println("turn on regularization")
            continue
        end
        if new_loss < 1e-6
            break
        end
    end
end

# # save loss_list to a file named by the "highway_data"+current timestamp
# file_name = "highway_data" * string(Dates.now()) * ".jld2"
# save(file_name, "loss_list", loss_list, "x0", x0)




now_str = Dates.format(now(), "yyyy-mm-dd-HH:MM:SS")
marker_size_list = 6*[0.98^(horizon+1-t) for t in 1:horizon+1]
alpha_list = [0.95^(horizon+1-t) for t in 1:horizon];
x_st = current_op.x;

folder_name = "log"
# for storing data:
data = Dict(
    "loss_list" => loss_list,
    "homotopy_loss" => homotopy_loss,
    "α_list" => α_list,
    "α_γ_list" => α_γ_list,
    "min_s_list" => min_s_list,
    "min_γ_list" => min_γ_list,
    "min_ineq_list" => min_ineq_list,
    "x0" => x0,
    "road_length" => road_length,
    "base_x" => base_x,
    "ineq_constraint_eps" => ineq_constraint_eps,
    "ρ_list" => ρ_list
)
# save the data to a JLD2 file
save(folder_name * "/" *"simple_data_highway.jld2", "data", data)






plot_size = (200, 400)

player1_color = :red
player2_color = :blue
player1_shape = :square
player2_shape = :square

upper_angle_list = LinRange(0, 2*segment_angle, 20)
lower_angle_list = LinRange(2*segment_angle, 0, 20)

upper_curve_x = base_x .+ radius .* (1 .- cos.(upper_angle_list))
upper_curve_y = road_length .- radius .* sin.(upper_angle_list)
lower_curve_x = x_right_corner .- radius .* (1 .- cos.(lower_angle_list))
lower_curve_y = y_right_corner .+ radius .* sin.(lower_angle_list)

x_inclined_lines = [upper_curve_x; lower_curve_x]
y_inclined_lines = [upper_curve_y; lower_curve_y]

x_upper_right_lines = [base_x, base_x]
y_upper_right_lines = [road_length, 5.0]
x_lower_right_lines = [x_right_corner, x_right_corner]
y_lower_right_lines = [y_right_corner, 0.0]
x_left_lines = [0.25, 0.25]
y_left_lines = [0.0, 5.0]
scatter([x_st[t][1] for t in 1:horizon], [x_st[t][2] for t in 1:horizon],markershape=player1_shape, markercolor=player1_color, label="", alpha=alpha_list)
scatter!([x_st[t][5] for t in 1:horizon], [x_st[t][6] for t in 1:horizon],markershape=player2_shape, markercolor=player2_color,label="", alpha=alpha_list)
scatter!([x_st[end][1]], [x_st[end][2]],markershape=player1_shape, label="player 1", markercolor=player1_color)
scatter!([x_st[end][5]], [x_st[end][6]],markershape=player2_shape, label="player 2", markercolor=player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "road edge", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L" ")
savefig(folder_name * "/" *"highway_st.pdf")


# plot initial trajectory:
x_st_init = current_op_list[1][1].x;
scatter([x_st_init[t][1] for t in 1:horizon], [x_st_init[t][2] for t in 1:horizon],markershape=player1_shape, color=player1_color,label="", alpha=alpha_list)
scatter!([x_st_init[t][5] for t in 1:horizon], [x_st_init[t][6] for t in 1:horizon],markershape=player2_shape,color=player2_color,label="", alpha=alpha_list)
scatter!([x_st_init[end][1]], [x_st_init[end][2]],markershape=player1_shape, label="", color = player1_color)
scatter!([x_st_init[end][5]], [x_st_init[end][6]],markershape=player2_shape, label="", color = player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L"p_y")
savefig(folder_name * "/" *"highway_st_init.pdf")

# plot second trajectory:
x_st_second = current_op_list[1][2].x;
scatter([x_st_second[t][1] for t in 1:horizon], [x_st_second[t][2] for t in 1:horizon],markershape=player1_shape, color=player1_color,label="", alpha=alpha_list)
scatter!([x_st_second[t][5] for t in 1:horizon], [x_st_second[t][6] for t in 1:horizon],markershape=player2_shape,color=player2_color,label="", alpha=alpha_list)
scatter!([x_st_second[end][1]], [x_st_second[end][2]],markershape=player1_shape, label="", color = player1_color)
scatter!([x_st_second[end][5]], [x_st_second[end][6]],markershape=player2_shape, label="", color = player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L" ")
savefig(folder_name * "/" *"highway_st_second.pdf")


# plot third trajectory:
x_st_third = current_op_list[1][3].x;
scatter([x_st_third[t][1] for t in 1:horizon], [x_st_third[t][2] for t in 1:horizon],markershape=player1_shape, color=player1_color,label="", alpha=alpha_list)
scatter!([x_st_third[t][5] for t in 1:horizon], [x_st_third[t][6] for t in 1:horizon],markershape=player2_shape,color=player2_color,label="", alpha=alpha_list)
scatter!([x_st_third[end][1]], [x_st_third[end][2]],markershape=player1_shape, label="", color = player1_color)
scatter!([x_st_third[end][5]], [x_st_third[end][6]],markershape=player2_shape, label="", color = player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L" ")
savefig(folder_name * "/" *"highway_st_third.pdf")

# plot the fourth trajectory:
x_st_fourth = current_op_list[1][4].x;
scatter([x_st_fourth[t][1] for t in 1:horizon], [x_st_fourth[t][2] for t in 1:horizon],markershape=player1_shape, color=player1_color,label="", alpha=alpha_list)
scatter!([x_st_fourth[t][5] for t in 1:horizon], [x_st_fourth[t][6] for t in 1:horizon],markershape=player2_shape,color=player2_color,label="", alpha=alpha_list)
scatter!([x_st_fourth[end][1]], [x_st_fourth[end][2]],markershape=player1_shape, label="", color = player1_color)
scatter!([x_st_fourth[end][5]], [x_st_fourth[end][6]],markershape=player2_shape, label="", color = player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L" ")
savefig(folder_name * "/" *"highway_st_fourth.pdf")



# plot the fifth trajectory:
x_st_fifth = current_op_list[1][5].x;
scatter([x_st_fifth[t][1] for t in 1:horizon], [x_st_fifth[t][2] for t in 1:horizon],markershape=player1_shape, color=player1_color,label="", alpha=alpha_list)
scatter!([x_st_fifth[t][5] for t in 1:horizon], [x_st_fifth[t][6] for t in 1:horizon],markershape=player2_shape,color=player2_color,label="", alpha=alpha_list)
scatter!([x_st_fifth[end][1]], [x_st_fifth[end][2]],markershape=player1_shape, label="", color = player1_color)
scatter!([x_st_fifth[end][5]], [x_st_fifth[end][6]],markershape=player2_shape, label="", color = player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L" ")
savefig(folder_name * "/" *"highway_st_fifth.pdf")



# plot the sixth trajectory:
x_st_sixth = current_op_list[1][6].x;
scatter([x_st_sixth[t][1] for t in 1:horizon], [x_st_sixth[t][2] for t in 1:horizon],markershape=player1_shape, color=player1_color,label="", alpha=alpha_list)
scatter!([x_st_sixth[t][5] for t in 1:horizon], [x_st_sixth[t][6] for t in 1:horizon],markershape=player2_shape,color=player2_color,label="", alpha=alpha_list)
scatter!([x_st_sixth[end][1]], [x_st_sixth[end][2]],markershape=player1_shape, label="", color = player1_color)
scatter!([x_st_sixth[end][5]], [x_st_sixth[end][6]],markershape=player2_shape, label="", color = player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L" ")
savefig(folder_name * "/" *"highway_st_sixth.pdf")






# in what follows, we plot loss values.
# create a 2 by 2 subplot
grid_option = true
# plot loss vs. iterations for the first iteration
plot(1:length(loss_list[1]), log10.(loss_list[1]), 
    label = "", color = :black, linewidth = 2, 
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm,
    size = (300, 200),
    linestyle = :solid,
    grid=grid_option
)
xlabel!("iterations "*L"k")
ylabel!(L"\lg(\|\|K_\rho(\mathbf{z}^{(k)})\|\|_2)")
savefig(folder_name * "/" *"highway_st_loss_first.pdf")


# # plot loss vs. iterations for the second iteration
# plot(1:length(loss_list[2]), log10.(loss_list[2]), 
#     label = "", color = :black, linewidth = 2, 
#     left_margin=10Plots.mm,
#     bottom_margin=5Plots.mm,
#     size = (300, 200),
#     linestyle = :solid,
#     grid=grid_option
# )
# xlabel!("iterations "*L"k")
# # ylabel!(L"\lg(K(\mathbf{z}^{(k)}))")
# savefig(folder_name * "/" *"highway_st_loss_second.pdf")



# # plot loss vs. iterations for the third iteration
# plot(1:length(loss_list[3]), log10.(loss_list[3]), 
#     label = "", color = :black, linewidth = 2, 
#     left_margin=10Plots.mm,
#     bottom_margin=5Plots.mm,
#     size = (300, 200),
#     linestyle = :solid,
#     grid=grid_option
# )
# xlabel!("iterations "*L"k")
# # ylabel!(L"\lg(K(\mathbf{z}^{(k)}))")
# savefig(folder_name * "/" *"highway_st_loss_third.pdf")


# # plot loss vs. iterations for the fifth iteration
# plot(1:length(loss_list[5]), log10.(loss_list[5]), 
#     label = "", color = :black, linewidth = 2, 
#     left_margin=10Plots.mm,
#     bottom_margin=5Plots.mm,
#     size = (300, 200),
#     linestyle = :solid,
#     grid=grid_option
# )
# xlabel!("iterations "*L"k")
# # ylabel!(L"\lg(K(\mathbf{z}^{(k)}))")
# savefig(folder_name * "/" *"highway_st_loss_fifth.pdf")

# # plot loss vs. iterations for the sixth iteration
# plot(1:length(loss_list[6]), log10.(loss_list[6]), 
#     label = "loss", color = :black, linewidth = 2, 
#     left_margin=10Plots.mm,
#     bottom_margin=5Plots.mm,
#     size = (300, 200),
#     linestyle = :solid,
#     grid=grid_option
# )
# xlabel!("iterations "*L"k")
# # ylabel!(L"\lg(K(\mathbf{z}^{(k)}))")
# savefig(folder_name * "/" *"highway_st_loss_sixth.pdf")



# # plot loss vs. iterations for the tenth iteration
# plot(1:length(loss_list[10]), log10.(loss_list[10]), 
#     label = "merit function value", color = :black, linewidth = 2, 
#     left_margin=10Plots.mm,
#     bottom_margin=5Plots.mm,
#     size = (300, 200),
#     linestyle = :solid,
#     grid=grid_option
# )
# xlabel!("iterations "*L"k")
# # ylabel!(L"\lg(K(\mathbf{z}^{(k)}))")
# savefig(folder_name * "/" *"highway_st_loss_tenth.pdf")






