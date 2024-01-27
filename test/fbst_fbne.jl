
using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time
using Infiltrator
# this file is certified to be correct.

# target: player 1 reaches lane 1.0, player 2 reaches lane -1.0
x0 = [
    -1.0; # left lane x1
    0.05; # y1
    0.0; # vx1 
    1.1; # vy1 
    1.0; # right lane x2
    0.0; # y2
    0.0; # vx2
    1.1; # vy2
] + 0.05*randn(8);

horizon = 30;
n_players = 2;
nx = 8;
nu = 4; # sum of dim of each player's control
m = Int(nu/2); # dim of each player's control
players_u_index_list = [1:2, 3:4];
dt = 0.1;
A = [
    I(2) dt*I(2) zeros(2,4);
    zeros(2,2) I(2) zeros(2,4);
    zeros(2,4) I(2) dt*I(2);
    zeros(2,4) zeros(2,2) I(2)
];
B = [
    zeros(2,4);
    dt*I(2) zeros(2,2);
    zeros(2,4);
    zeros(2,2) dt*I(2)
];
# Q1 = 4*[zeros(2,2) zeros(2,2); zeros(2,2) I(2)];
# Q2 = 4*[I(2) -I(2); -I(2) I(2)];
collision_avoidance_mat = zeros(nx, nx);
collision_avoidance_mat[1,1] = 1.0;
collision_avoidance_mat[2,2] = 1.0;
collision_avoidance_mat[5,5] = 1.0;
collision_avoidance_mat[6,6] = 1.0;
collision_avoidance_mat[1,5] = -1.0;
collision_avoidance_mat[5,1] = -1.0;
collision_avoidance_mat[2,6] = -1.0;
collision_avoidance_mat[6,2] = -1.0;
tmp1 = zeros(nx,nx)
tmp2 = zeros(nx,nx)
tmp1[1,1] = 1.1;
# tmp1[2,2] = 1.0;
tmp1[3,3] = 1.0;
tmp1[4,4] = 1.0;


tmp2[5,5] = 1.1;
# tmp2[6,6] = 1.0;
tmp2[7,7] = 1.0;
tmp2[8,8] = 1.0;

Q1 = tmp1 - 0.45*collision_avoidance_mat; # set to be 0.5, TODO!!!
Q2 = tmp2 - 0.45*collision_avoidance_mat;

S1 = zeros(nu,nx);
S2 = zeros(nu,nx);
q1 = [
    -1.0; 
    0; 
    0; 
    -1.0; 
    0; 
    0; 
    0; 
    0
];
q2 = [
    0; 
    0; 
    0; 
    0; 
    1.0; 
    0; 
    0; 
    -1.0
];
# R1 = 4*[I(m) zeros(m,m); zeros(m,m) zeros(m,m)];
# R2 = 4*[zeros(m,m) zeros(m,m); zeros(m,m) I(m)];
R1 = 4.0*I(nu);
R2 = 4.0*I(nu);
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


Hx1 = zeros(1,nx);
Hx2 = zeros(1,nx);
Hu1 = zeros(1,nu);
Hu2 = zeros(1,nu);
h1 = [0];
h2 = [0];
Hx1_terminal = zeros(1,nx);
Hx2_terminal = zeros(1,nx);
h1_terminal = [0]
h2_terminal = [0]

Hx_list = [[Hx1, Hx2] for t in 1:horizon];
Hu_list = [[Hu1, Hu2] for t in 1:horizon];
h_list = [[h1, h2] for t in 1:horizon];
HxT = [Hx1_terminal, Hx2_terminal];
hxT = [h1_terminal, h2_terminal];
equality_constraints_size = size(Hx1)[1];

Gx1 = zeros(1,nx);
Gx2 = zeros(1,nx);
Gu1 = zeros(1,nu);
Gu2 = zeros(1,nu);
g1 = [0]
g2 = [0]
Gx1_terminal = zeros(1,nx);
Gx2_terminal = zeros(1,nx);
g1_terminal = [0]
g2_terminal = [0]
Gx_list = [[Gx1, Gx2] for t in 1:horizon];
Gu_list = [[Gu1, Gu2] for t in 1:horizon];
g_list = [[g1, g2] for t in 1:horizon];
GxT = [Gx1_terminal, Gx2_terminal];
gxT = [g1_terminal, g2_terminal];
inequality_constraints_size = size(Gx1)[1];


# we first consider fbst case:
# initializing the strategy:
π_st = strategy([zeros(nu,nx) for t in 1:horizon], [zeros(nu) for t in 1:horizon])

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

x_st, u_st, μ, λ, η, ψ, Mₜ, Nₜ, nₜ = constrained_fbst_lq_solver!(π_st, g, false, true)

J1_st = [0. for t in 1:horizon]
J2_st = [0. for t in 1:horizon]
for t = 1:horizon
    J1_st[t] = 1/2*x_st[t]'*Q1*x_st[t] + 1/2*u_st[t]'*R1*u_st[t]
    J2_st[t] = 1/2*x_st[t]'*Q2*x_st[t] + 1/2*u_st[t]'*R2*u_st[t]
end
J1_st[end] = 1/2*x_st[end]'*Q1*x_st[end]
J2_st[end] = 1/2*x_st[end]'*Q2*x_st[end]

# -------------------------------------
# we then consider the fbne case:

π_ne = strategy([zeros(nu,nx) for t in 1:horizon], [zeros(nu) for t in 1:horizon])
fbne_lq_solver!(π_ne, g)

x_ne = [zeros(nx) for t in 1:horizon+1]
u_ne = [zeros(nu) for t in 1:horizon]
J1_ne = [0. for t in 1:horizon]
J2_ne = [0. for t in 1:horizon]
x_ne[1] = x0
for t = 1:horizon
    u_ne[t] = -π_ne.P[t]*x_ne[t]
    x_ne[t+1] = A*x_ne[t] + B*u_ne[t]
end

for t = 1:horizon

    J1_ne[t] = 1/2*x_ne[t]'*Q1*x_ne[t] + 1/2*u_ne[t]'*R1*u_ne[t]
    J2_ne[t] = 1/2*x_ne[t]'*Q2*x_ne[t] + 1/2*u_ne[t]'*R2*u_ne[t]
end
J1_ne[end] = 1/2*x_ne[end]'*Q1*x_ne[end]
J2_ne[end] = 1/2*x_ne[end]'*Q2*x_ne[end]







# ---------------------- below is for plotting ----------------------
x_ne

# TODO: design LQ intersection experiment
# TODO: compute the two costs!

x_st


using Plots
total_cost_J1_st = round(sum(J1_st), digits=1)
total_cost_J2_st = round(sum(J2_st), digits=1)
total_cost_J1_ne = round(sum(J1_ne), digits=1)
total_cost_J2_ne = round(sum(J2_ne), digits=1)


marker_size_list = 6*[0.98^(horizon+1-t) for t in 1:horizon+1]
scatter([x_st[t][1] for t in 1:horizon+1], [x_st[t][2] for t in 1:horizon+1],markershape=:square, label="player 1 fbst", markersize=marker_size_list)
scatter!([x_st[t][5] for t in 1:horizon+1], [x_st[t][6] for t in 1:horizon+1],markershape=:square, label="player 2 fbst", markersize=marker_size_list)
scatter!([x_ne[t][1] for t in 1:horizon+1], [x_ne[t][2] for t in 1:horizon+1],markershape=:circle, label="player 1 fbne", markersize=marker_size_list)
scatter!([x_ne[t][5] for t in 1:horizon+1], [x_ne[t][6] for t in 1:horizon+1],markershape=:circle, label="player 2 fbne", markersize=marker_size_list)
title!("J1_st = $total_cost_J1_st, J2_st = $total_cost_J2_st, J1_ne = $total_cost_J1_ne, J2_ne = $total_cost_J2_ne")
savefig("fbst_fbne.png")


# animation for fbst:

anim1 = @animate for t = 1:horizon+1
    scatter([x_st[t][1] for t in 1:t], [x_st[t][2] for t in 1:t],markershape=:square, label="player 1 fbst", markersize=marker_size_list[1:t])
    scatter!([x_st[t][5] for t in 1:t], [x_st[t][6] for t in 1:t],markershape=:square, label="player 2 fbst", markersize=marker_size_list[1:t])
    xlims!(-1.1,1.1)
    ylims!(0,4.1)
    title!("t = $t")
end
gif(anim1, "fbst.gif", fps = 10)


# animation for fbne:

anim2 = @animate for t = 1:horizon+1
    scatter([x_ne[t][1] for t in 1:t], [x_ne[t][2] for t in 1:t],markershape=:circle, label="player 1 fbne", markersize=marker_size_list[1:t])
    scatter!([x_ne[t][5] for t in 1:t], [x_ne[t][6] for t in 1:t],markershape=:circle, label="player 2 fbne", markersize=marker_size_list[1:t])
    xlims!(-1.1,1.1)
    ylims!(0,4.1)
    title!("t = $t")
end
gif(anim2, "fbne.gif", fps = 10)



