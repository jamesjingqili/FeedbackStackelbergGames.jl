using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time
using Infiltrator
# this file is certified to be correct.

x0 = [
    1.0;
    2.0;
    2.0;
    3.0
];

horizon = 10;
n_players = 2;
nx = 4;
nu = 4; # sum of dim of each player's control
m = Int(nu/2); # dim of each player's control
players_u_index_list = [1:2, 3:4];


# A = Matrix(1.0*I(4));
# B = Matrix(0.1*I(4));
A = rand(4,4);
B = rand(4,4);
Q1 = Matrix(4.0*I(nx));
Q2 = Matrix(4.0*I(nx));
S1 = zeros(nu,nx);
S2 = zeros(nu,nx);
q1 = zeros(nx);
q2 = zeros(nx);
R1 = 2*[I(m) zeros(m,m); zeros(m,m) zeros(m,m)];
R2 = 2*[zeros(m,m) zeros(m,m); zeros(m,m) I(m)];
r1 = zeros(nu);
r2 = zeros(nu);


A_list = [A for t in 1:horizon];
B_list = [B for t in 1:horizon];
c_list = [zeros(nx) for t in 1:horizon];
Q_list = [[Q1, Q2] for t in 1:horizon+1];
S_list = [[S1, S2] for t in 1:horizon];
R_list = [[R1, R2] for t in 1:horizon];
q_list = [[q1, q2] for t in 1:horizon+1];
r_list = [[r1, r2] for t in 1:horizon];


Hx1 = [0 0 0 0]
Hx2 = [0 0 0 0]
Hu1 = [0 0 0 0]
Hu2 = [0 0 0 0]
h1 = [0]
h2 = [0]
Hx1_terminal = [0 0 1.0 0]*0.0
Hx2_terminal = [0 0 1.0 0]*0.0
h1_terminal = [-1.]*0.0
h2_terminal = [-1.]*0.0


Hx_list = [[Hx1, Hx2] for t in 1:horizon];
Hu_list = [[Hu1, Hu2] for t in 1:horizon];
h_list = [[h1, h2] for t in 1:horizon];
HxT = [Hx1_terminal, Hx2_terminal];
hxT = [h1_terminal, h2_terminal];
equality_constraints_size = size(Hx1)[1];

Gx1 = [0 0 0 0]
Gx2 = [0 0 0 0]
Gu1 = [0 0 0 0]
Gu2 = [0 0 0 0]
g1 = [0]
g2 = [0]
Gx1_terminal = [0 0 0 0]
Gx2_terminal = [0 0 0 0]
g1_terminal = [0]
g2_terminal = [0]
Gx_list = [[Gx1, Gx2] for t in 1:horizon];
Gu_list = [[Gu1, Gu2] for t in 1:horizon];
g_list = [[g1, g2] for t in 1:horizon];
GxT = [Gx1_terminal, Gx2_terminal];
gxT = [g1_terminal, g2_terminal];
inequality_constraints_size = size(Gx1)[1];

# initializing the strategy:
π = strategy([zeros(nu,nx) for t in 1:horizon], [zeros(nu) for t in 1:horizon])

g = Constrained_LQGame(
    horizon,
    n_players,
    nx,
    nu,
    players_u_index_list,
    A_list,
    B_list,
    c_list,
    Q_list,
    S_list,
    R_list,
    q_list,
    r_list,
    equality_constraints_size,
    Hx_list,
    Hu_list,
    h_list,
    HxT,
    hxT,
    inequality_constraints_size,
    Gx_list,
    Gu_list,
    g_list,
    GxT,
    gxT,
    x0
);

f_list = [(x, u)-> A_list[t]*x + B_list[t]*u for t in 1:horizon];
costs_list = [
    [(x, u)-> 0.5*(x'*Q_list[t][ii]*x + u'*R_list[t][ii]*u + 
    2*x'*S_list[t][ii]*u + 2*q_list[t][ii]'*x + 2*r_list[t][ii]'*u) for ii in 1:n_players] for t in 1:horizon
];
terminal_costs_list = [
    (x)-> 0.5*(x'*Q_list[end][ii]*x + 2*q_list[end][ii]'*x) for ii in 1:n_players
];
equality_constraints_list = [
    [(z)-> Hx_list[t][ii]*z[1:nx] + Hu_list[t][ii]*z[nx+1:nx+nu] + h_list[t][ii] for ii in 1:n_players] for t in 1:horizon
];
terminal_equality_constraints_list = [
    (x)-> HxT[ii]*x + hxT[ii] for ii in 1:n_players
];
inequality_constraints_list = [
    [(z)-> Gx_list[t][ii]*z[1:nx] + Gu_list[t][ii]*z[nx+1:nx+nu] + g_list[t][ii] for ii in 1:n_players] for t in 1:horizon
];
terminal_inequality_constraints_list = [
    (x)-> GxT[ii]*x + gxT[ii] for ii in 1:n_players
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
)

# initializing the trajectory:
current_op = trajectory(
    x = [x0 for t in 1:horizon+1],
    u = [zeros(nu) for t in 1:horizon],
    λ = [zeros(nx*n_players) for t in 1:horizon],
    η = [zeros(nu) for t in 1:horizon-1],
    ψ = [zeros(m) for t in 1:horizon],
    μ = [zeros(equality_constraints_size*n_players) for t in 1:horizon+1],
    γ = [zeros(inequality_constraints_size*n_players) for t in 1:horizon+1],
)

# we simulate the initial trajectory!
forward_simulation!(current_op, nonlinear_g)

# initializing the LQ approximation:
lq_approx = deepcopy(g)

# we linearize the dynamics of the nonlinear game around the current operating point!
lq_approximation!(lq_approx, nonlinear_g, current_op)

x_true, u_true, μ_true, λ_true, η_true, ψ_true, _, _, _ = constrained_fbst_lq_solver!(
    π, 
    g, 
    false,
    true
)

lq_approx = deepcopy(g);

current_op = trajectory(
    x = [x0 for t in 1:horizon+1],
    u = [zeros(nu) for t in 1:horizon],
    λ = [zeros(nx*n_players) for t in 1:horizon],
    η = [zeros(nu) for t in 1:horizon-1],
    ψ = [zeros(m) for t in 1:horizon],
    μ = [zeros(equality_constraints_size*n_players) for t in 1:horizon+1],
    γ = [zeros(inequality_constraints_size*n_players) for t in 1:horizon+1],
);
next_op = deepcopy(current_op);
forward_simulation!(current_op, nonlinear_g);

lq_approximation!(lq_approx, nonlinear_g, current_op);
x, u, μ, λ, η, ψ, Mₜ, Nₜ, nₜ = constrained_fbst_lq_solver!(π, lq_approx);

num_iter = 40;
loss_list = zeros(num_iter);
α_list = zeros(num_iter);

for iter in 1:num_iter
    global x, u, μ, λ, η, ψ, Mₜ, Nₜ, nₜ, Δ, α, new_loss, next_op, current_op
    lq_approximation!(lq_approx, nonlinear_g, current_op)
    x, u, μ, λ, η, ψ, Mₜ, Nₜ, nₜ = constrained_fbst_lq_solver!(π, lq_approx)
    Δ = trajectory(
        x = x,
        u = u,
        λ = λ,
        η = η,
        ψ = ψ,
        μ = μ,
        γ = [zeros(inequality_constraints_size*n_players) for t in 1:horizon+1]
    );
    α, new_loss, next_op = line_search!(
        π,
        Δ,
        current_op, 
        lq_approx,
        nonlinear_g,
        1.0, # initial step size
        0.5 # step size reduction factor
    );
    # @infiltrate

    current_op = next_op;
    loss_list[iter] = new_loss;
    α_list[iter] = α;
    # forward_simulation!(current_op, nonlinear_g);
end

loss_list

