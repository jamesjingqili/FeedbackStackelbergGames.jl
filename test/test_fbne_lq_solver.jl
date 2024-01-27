using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time

import iLQGames_discrete_time: LQGame, strategy, fbne_lq_solver!

x0 = [1;1;2;1];

horizon = 10;
n_players = 2;
nx = 4;
nu = 4;
m=2; # dim of each player's control
players_u_index_list = [1:2, 3:4];

A = I(nx);
B = 0.1*I(nx);
Q1 = 4*[zeros(2,2) zeros(2,2); zeros(2,2) I(2)];
Q2 = 4*[I(2) -I(2); -I(2) I(2)];
S1 = zeros(nu,nx);
S2 = zeros(nu,nx);
q1 = zeros(nx);
q2 = zeros(nx);
R1 = 2*[I(2) zeros(2,2); zeros(2,2) zeros(2,2)];
R2 = 2*[zeros(2,2) zeros(2,2); zeros(2,2) I(2)];
r1 = zeros(4);
r2 = zeros(4);

A_list = [A for t in 1:horizon];
B_list = [B for t in 1:horizon];
c_list = [zeros(nx) for t in 1:horizon];
Q_list = [[Q1, Q2] for t in 1:horizon];
S_list = [[S1, S2] for t in 1:horizon];
R_list = [[R1, R2] for t in 1:horizon];
q_list = [[q1, q2] for t in 1:horizon];
r_list = [[r1, r2] for t in 1:horizon];

# initializing the strategy:
π = strategy([zeros(4,4) for t in 1:horizon], [zeros(4) for t in 1:horizon])

g = LQGame(horizon, n_players, nx, nu, players_u_index_list, A_list, B_list, c_list, Q_list, S_list, R_list, q_list, r_list, x0)


fbne_lq_solver!(π, g)

x_traj = [x0 for t in 1:horizon+1]
u_traj = [π.P[t]*x_traj[t] for t in 1:horizon]
for t = 1:horizon
    u_traj[t] = π.P[t]*x_traj[t] + π.α[t]
    x_traj[t+1] = A*x_traj[t] + B*u_traj[t]
end

