using LinearAlgebra
using Infiltrator
using ForwardDiff
using BlockArrays
using FileIO

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
    M[Block(t+2*T, t+2*T)] = I(nx)
    M[Block(t+2*T, t)] = -(B1 + B2*sol2[Block(t, t+1)])
    for τ = 1:T-t
        M[Block(t+2*T, t+τ)] = -B2*sol2[Block(t, t+1+τ)]
    end
    if t >1
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

