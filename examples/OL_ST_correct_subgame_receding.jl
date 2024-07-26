using LinearAlgebra
using Infiltrator
using ForwardDiff
using BlockArrays
using FileIO



# ------------------- The code below is for computing OLSE -------------------

x0 = [
    1.0;
    2.0;
    2.0;
    1.0;
];
T = 30;
nx = 4;
nu = 4;
m = 2;
A = Matrix(1.0*I(4));
B = Matrix(0.1*I(4));
B1 = B[:,1:2];
B2 = B[:,3:4];
Q1 = 4.0 * [
    0 0 0 0;
    0 0 0 0;
    0 0 1 0;
    0 0 0 1;
];
Q2 = 4.0 * [
    1 0 -1 0;
    0 1 0 -1;
    -1 0 1 0;
    0 -1 0 1;
];
# Q1 = 4.0 * I(nx);
# Q2 = 4.0 * I(nx);
# R1 = 2*[I(m) zeros(m,m); zeros(m,m) zeros(m,m)];
# R2 = 2*[zeros(m,m) zeros(m,m); zeros(m,m) I(m)];
R1 = 2*I(m);
R2 = 2*I(m);


M2 = BlockArray(zeros((nx+m)*T+nx*T, (nx+m)*T+nx*T), 
    vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T)), 
    vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T))
);
N2 = BlockArray(zeros((nx+m)*T+nx*T, nx+m*T), 
    vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T)),
    vcat([nx], m*ones(Int, T))
);
n2 = BlockArray(zeros((nx+m)*T+nx*T), 
    vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T))
);
λ1 = zeros(nx*T);
λ2 = zeros(nx*T);
η = zeros(m*T); # dual variable for the follower control constraints

# we first solve the follower's strategy: M2*[u2;λ2;x_1] + N2*[x_0;u1] + n2 = 0
# first T rows: for taking gradient over u2:
for t in 1:T
    M2[Block(t,t)] = R2
    M2[Block(t,t+T)] = -B2'
end
# second T rows: for taking gradient over x:
for t in 1:T
    M2[Block(t+T,t+2*T)] = Q2
    M2[Block(t+T, t+T)] = I(nx)
    if t>1
        M2[Block(t+T-1, t+T)] = -A'
    end
end
# third T rows: for the equality constraint of the dynamics equation:
for t in 1:T
    M2[Block(t+2*T, t+2*T)] = I(nx)
    M2[Block(t+2*T, t)] = -B2
    if t >1
        M2[Block(t+2*T, t+2*T-1)] = -A
    end
    N2[Block(t+2*T, t+1)] = -B1
end
N2[Block(2*T+1, 1)] = -A;

sol2 = -inv(M2)*N2;

# we assume uₜ² = K0*x₀ + K1*uₜ¹
# we then solve the leader's strategy: M*u1 + N*x0 + n = 0
M = BlockArray(zeros(m*T + 2*nx*T, m*T + 2*nx*T), 
    vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T)), 
    vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T))
);
N = BlockArray(zeros(m*T + 2*nx*T, nx), 
    vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T)),
    [nx]
);
# first T rows: for taking gradient over u1:
for t in 1:T
    M[Block(t,t)] = R1
    M[Block(t,t+T)] = -B1'
end
# second T rows: for taking gradient over x:
for t in 1:T
    M[Block(t+T,t+2*T)] = Q1
    M[Block(t+T, t+T)] = I(nx)
    if t>1
        M[Block(t+T-1, t+T)] = -A'
    end
end
# third T rows: for the equality constraint of the dynamics equation:
for t in 1:T
    M[Block(t+2*T, t+2*T)] = I(nx) # dx' / dx'
    M[Block(t+2*T, t)] = -(B1 + B2*sol2[Block(t, t+1)]) # dx' / du1
    for τ = 1:T-t
        # M[Block(t+2*T, t+τ)] = -B2*sol2[Block(t, t+1+τ)] 
    end
    if t >1
        N[Block(t+2*T,1)] = -B2*sol2[Block(t,1)]
        M[Block(t+2*T, t+2*T-1)] = -A
    end
end
N[Block(2*T+1, 1)] = -A-B2*sol2[Block(1,1)];

sol = -inv(M)*N*x0;



u1 = sol[1:m*T]
u2 = sol2[1:m*T, 1:nx]*x0 + sol2[1:m*T, nx+1:nx+m*T]*u1


x_OL = [x0 for t in 1:T+1]
u1_OL = [u1[1+m*(t-1):m*t] for t in 1:T]
u2_OL = [u2[1+m*(t-1):m*t] for t in 1:T]
for t in 1:T
    x_OL[t+1] = A*x_OL[t] + B1*u1_OL[t] + B2*u2_OL[t]
end

x_OL











function compute_OLSE(A,B1,B2,Q1,Q2,R1,R2,T,x0)
    nx = size(A,1)
    nu = size(B1,2) + size(B2,2)
    m = size(B1,2)
    M2 = BlockArray(zeros((nx+m)*T+nx*T, (nx+m)*T+nx*T), 
        vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T)), 
        vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T))
    );
    N2 = BlockArray(zeros((nx+m)*T+nx*T, nx+m*T), 
        vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T)),
        vcat([nx], m*ones(Int, T))
    );
    n2 = BlockArray(zeros((nx+m)*T+nx*T), 
        vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T))
    );
    λ1 = zeros(nx*T);
    λ2 = zeros(nx*T);
    η = zeros(m*T); # dual variable for the follower control constraints

    # we first solve the follower's strategy: M2*[u2;λ2;x_1] + N2*[x_0;u1] + n2 = 0
    # first T rows: for taking gradient over u2:
    for t in 1:T
        M2[Block(t,t)] = R2
        M2[Block(t,t+T)] = -B2'
    end
    # second T rows: for taking gradient over x:
    for t in 1:T
        M2[Block(t+T,t+2*T)] = Q2
        M2[Block(t+T, t+T)] = I(nx)
        if t>1
            M2[Block(t+T-1, t+T)] = -A'
        end
    end
    # third T rows: for the equality constraint of the dynamics equation:
    for t in 1:T
        M2[Block(t+2*T, t+2*T)] = I(nx)
        M2[Block(t+2*T, t)] = -B2
        if t >1
            M2[Block(t+2*T, t+2*T-1)] = -A
        end
        N2[Block(t+2*T, t+1)] = -B1
    end
    N2[Block(2*T+1, 1)] = -A;

    sol2 = -inv(M2)*N2;

    # we assume uₜ² = K0*x₀ + K1*uₜ¹
    # we then solve the leader's strategy: M*u1 + N*x0 + n = 0
    M = BlockArray(zeros(m*T + 2*nx*T, m*T + 2*nx*T), 
        vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T)), 
        vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T))
    );
    N = BlockArray(zeros(m*T + 2*nx*T, nx), 
        vcat(m*ones(Int, T), nx*ones(Int, T), nx*ones(Int, T)),
        [nx]
    );
    # first T rows: for taking gradient over u1:
    for t in 1:T
        M[Block(t,t)] = R1
        M[Block(t,t+T)] = -B1'
    end
    # second T rows: for taking gradient over x:
    for t in 1:T
        M[Block(t+T,t+2*T)] = Q1
        M[Block(t+T, t+T)] = I(nx)
        if t>1
            M[Block(t+T-1, t+T)] = -A'
        end
    end
    # third T rows: for the equality constraint of the dynamics equation:
    for t in 1:T
        M[Block(t+2*T, t+2*T)] = I(nx) # dx' / dx'
        M[Block(t+2*T, t)] = -(B1 + B2*sol2[Block(t, t+1)]) # dx' / du1
        for τ = 1:T-t
            # M[Block(t+2*T, t+τ)] = -B2*sol2[Block(t, t+1+τ)] 
        end
        if t >1
            N[Block(t+2*T,1)] = -B2*sol2[Block(t,1)]
            M[Block(t+2*T, t+2*T-1)] = -A
        end
    end
    N[Block(2*T+1, 1)] = -A-B2*sol2[Block(1,1)];

    sol = -inv(M)*N*x0;



    u1 = sol[1:m*T]
    u2 = sol2[1:m*T, 1:nx]*x0 + sol2[1:m*T, nx+1:nx+m*T]*u1


    x_OL = [x0 for t in 1:T+1]
    u1_OL = [u1[1+m*(t-1):m*t] for t in 1:T]
    u2_OL = [u2[1+m*(t-1):m*t] for t in 1:T]
    for t in 1:T
        x_OL[t+1] = A*x_OL[t] + B1*u1_OL[t] + B2*u2_OL[t]
    end

    return x_OL, u1_OL, u2_OL
end



for t in 1:T
    _, u1, u2 = compute_OLSE(A, B1, B2, Q1, Q2, R1, R2, T+1-t, x_OL[t]);
    x_OL[t+1] = A*x_OL[t] + B1*u1[1] + B2*u2[1];
end







# ------------------- The code below is for computing FNE -------------------




using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time
using Infiltrator

x0 = [
    1.0;
    2.0;
    2.0;
    1.0
];

horizon = 30;
n_players = 2;
nx = 4;
nu = 4; # sum of dim of each player's control
m = Int(nu/2); # dim of each player's control
players_u_index_list = [1:2, 3:4];

A = Matrix(I(4));
B = Matrix(0.1*I(4));
Q1 = 4*[zeros(2,2) zeros(2,2); zeros(2,2) I(2)];
Q2 = 4*[I(2) -I(2); -I(2) I(2)];

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

x_FB, u_FB, μ, λ, η, ψ, Mₜ, Nₜ, nₜ = constrained_fbst_lq_solver!(π, g, false, true)


# ------------------- compare the two strategies -------------------

using Plots

ENV["GKSwstype"]="nul"

plot([x_OL[t][1] for t in 1:T+1], [x_OL[t][2] for t in 1:T+1], label="p1, RH-OLSE")
plot!([x_OL[t][3] for t in 1:T+1], [x_OL[t][4] for t in 1:T+1], label="p2, RH-OLSE")
plot!([x_FB[t][1] for t in 1:T+1], [x_FB[t][2] for t in 1:T+1], label="p1, FBST")
plot!([x_FB[t][3] for t in 1:T+1], [x_FB[t][4] for t in 1:T+1], label="p2, FBST")
savefig("FBSE_vs_RH_OLSE.png")


