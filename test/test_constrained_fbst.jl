using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time
using Infiltrator
# this file is certified to be correct.

x0 = [
    1.0;
    2.0;
    3.0;
    4.0
];

horizon = 3;
n_players = 2;
nx = 4;
nu = 4; # sum of dim of each player's control
m = Int(nu/2); # dim of each player's control
players_u_index_list = [1:2, 3:4];

# A = [1 2 3 4; 
#     0 1 2 3; 
#     0 0 1 2; 
#     0 0 0 1];
# B = [0.1 0 0 0; 
#     0 0.1 0 0.1; 
#     0 0 0.1 0;
#     0.1 0 0 0.1];
A = Matrix(I(4));
B = Matrix(0.1*I(4));
# Q1 = 4*[zeros(2,2) zeros(2,2); zeros(2,2) I(2)];
# Q2 = 4*[I(2) -I(2); -I(2) I(2)];
# Q2 = 4*[I(2) zeros(2,2); zeros(2,2) zeros(2,2)];
Q1 = Matrix(4*I(nx));
Q2 = Matrix(4*I(nx));
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
Hx1_terminal = [0 0 0 0]
Hx2_terminal = [0 0 0 0]
h1_terminal = [0]
h2_terminal = [0]

# Hx1 = [0 0 0 0; 0 0 0 0];
# Hx2 = [0 0 0 0; 0 0 0 0];
# Hu1 = [0 0 0 0; 0 0 0 0];
# Hu2 = [0 0 0 0; 0 0 0 0];
# h1 = [0; 0];
# h2 = [0; 0];
# Hx1_terminal = [1 0 0 0; 0 1 0 0];
# Hx2_terminal = [1 0 0 0; 0 1 0 0];
# h1_terminal = [2; 3];
# h2_terminal = [2; 3];

# Hx1 = [1 0]
# Hx2 = [0 0]
# Hu1 = [1 0]
# Hu2 = [0 0]
# h1 = [1]
# h2 = [0]
# Hx1_terminal = [0 0]
# Hx2_terminal = [0 0]
# h1_terminal = [0]
# h2_terminal = [0]

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
# f = [(x, u)-> A_list[t]*x + B_list[t]*u for t in 1:horizon];



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

x, u, μ, λ, η, ψ, Mₜ, Nₜ, nₜ = constrained_fbst_lq_solver!(π, g)



Δx, Δu, λ̂, η̂, ψ̂ =fbst_lq_solver!(π, g)



