using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time
using Infiltrator
# this file is certified to be correct.

x0 = [
    2.0;
    2.0;
];

horizon = 10;
n_players = 2;
nx = 2;
nu = 2; # sum of dim of each player's control
m = Int(nu/2); # dim of each player's control
players_u_index_list = [1, 2];


A = Matrix(1.0*I(nx));
B = Matrix(0.1*I(nu));
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


Hx1 = [0 0]
Hx2 = [0 0]
Hu1 = [0 0]
Hu2 = [0 0]
h1 = [0]
h2 = [0]
Hx1_terminal = [1.0 0]*0.0
Hx2_terminal = [1.0 0]*0.0
h1_terminal = [-1.]*0.0
h2_terminal = [-1.]*0.0


Hx_list = [[Hx1, Hx2] for t in 1:horizon];
Hu_list = [[Hu1, Hu2] for t in 1:horizon];
h_list = [[h1, h2] for t in 1:horizon];
HxT = [Hx1_terminal, Hx2_terminal];
hxT = [h1_terminal, h2_terminal];
equality_constraints_size = size(Hx1)[1];

turn_off_inequality_constraints = 1.0;

Gx1 = [1.0 0]*turn_off_inequality_constraints
Gx2 = [1.0 0]*turn_off_inequality_constraints
Gu1 = [0 0]*turn_off_inequality_constraints
Gu2 = [0 0]*turn_off_inequality_constraints
g1 = [-1.9]*turn_off_inequality_constraints
g2 = [-1.9]*turn_off_inequality_constraints
Gx1_terminal = [0 1.0]*turn_off_inequality_constraints
Gx2_terminal = [0 1.0]*turn_off_inequality_constraints
g1_terminal = [-1.9]*turn_off_inequality_constraints
g2_terminal = [-1.9]*turn_off_inequality_constraints
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



f_list = [(x, u)-> A_list[t]*[x[1]+0.0*x[1]^2; x[2]] + B_list[t]*u for t in 1:horizon];
costs_list = [
    [(x, u)-> 0.5*(x'*Q_list[t][ii]*x + u'*R_list[t][ii]*u + 
    2*x'*S_list[t][ii]*u + 2*q_list[t][ii]'*x + 2*r_list[t][ii]'*u) for ii in 1:n_players] 
    for t in 1:horizon
];
terminal_costs_list = [
    (x)-> 0.5*(x'*Q_list[end][ii]*x + 2*q_list[end][ii]'*x) for ii in 1:n_players
];
equality_constraints_list = [
    [(z)-> Hx_list[t][ii]*z[1:nx] + Hu_list[t][ii]*z[nx+1:nx+nu] + h_list[t][ii] for ii in 1:n_players] 
    for t in 1:horizon
];
terminal_equality_constraints_list = [
    (x)-> HxT[ii]*x + hxT[ii] for ii in 1:n_players
];
inequality_constraints_list = [
    [(z)-> Gx_list[t][ii]*z[1:nx] + Gu_list[t][ii]*z[nx+1:nx+nu] + g_list[t][ii] for ii in 1:n_players] 
    for t in 1:horizon
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




# define the current_op
ρ₀ = 1/1.0;
# TODO: we should think about initializing γ by considering γ .* inequality_constraints_list = ρ
initial_op = trajectory(
    x = [x0 for t in 1:horizon+1],
    u = [zeros(nu) for t in 1:horizon],
    λ = [zeros(nx*n_players) for t in 1:horizon],
    η = [zeros(nu) for t in 1:horizon-1],
    ψ = [zeros(m) for t in 1:horizon],
    μ = [zeros(equality_constraints_size*n_players) for t in 1:horizon+1],
    γ = [30*ones(inequality_constraints_size*n_players) for t in 1:horizon+1],
);
forward_simulation!(initial_op, nonlinear_g);
calibrated_γ = initial_op.γ;
for t = 1:horizon+1
    if t == horizon+1
        calibrated_γ[t] =  [
            diagm(
                nonlinear_g.terminal_inequality_constraints_list[1](initial_op.x[t])
            )^(-1)*(ρ₀.*ones(inequality_constraints_size));
            diagm(
                nonlinear_g.terminal_inequality_constraints_list[2](initial_op.x[t])
            )^(-1)*(ρ₀.*ones(inequality_constraints_size));
        ]
    else
        calibrated_γ[t] =  [
        diagm(
            nonlinear_g.inequality_constraints_list[t][1](vcat(initial_op.x[t], initial_op.u[t]))
        )^(-1)*(ρ₀.*ones(inequality_constraints_size));
        diagm(
            nonlinear_g.inequality_constraints_list[t][2](vcat(initial_op.x[t], initial_op.u[t]))
        )^(-1)*(ρ₀.*ones(inequality_constraints_size))
    ]
    end
end

current_op = trajectory(
    x = initial_op.x,
    u = initial_op.u,
    λ = initial_op.λ,
    η = initial_op.η,
    ψ = initial_op.ψ,
    μ = initial_op.μ,
    γ = calibrated_γ
);


# initializing the LQ approximation:
lq_approx = deepcopy(g)

# we linearize the dynamics of the nonlinear game around the current operating point!
lq_approximation!(lq_approx, nonlinear_g, current_op)



# initialize the lq approximation and all the solution variables
lq_approximation!(lq_approx, nonlinear_g, current_op);
x, u, γ, μ, λ, η, ψ, Mₜ, Nₜ, nₜ = pdip_fbst_lq_solver!(π, lq_approx, current_op, 1/1.0, false, true, true);




# we do an outer iteration, where we periodically update ρ
# in inner iterations, we fix ρ and update the linear policy till convergence

σ = 1.5;
num_outer_iter = 40;
num_iter = 20;
ρ = ρ₀;
ρ_list = [ρ for ii in 1:num_outer_iter];

loss_list = [zeros(num_iter) for ii in 1:num_outer_iter];
homotopy_loss = [zeros(num_iter) for ii in 1:num_outer_iter];
α_list = [zeros(num_iter) for ii in 1:num_outer_iter];
α_γ_list = [zeros(num_iter) for ii in 1:num_outer_iter];
min_ineq_list = [zeros(num_iter) for ii in 1:num_outer_iter];
min_γ_list = [zeros(num_iter) for ii in 1:num_outer_iter];
current_op_list = [[] for ii in 1:num_outer_iter];
Δ_list = [[] for ii in 1:num_outer_iter];
M_list = [[] for ii in 1:num_outer_iter];
N_list = [[] for ii in 1:num_outer_iter];
n_list = [[] for ii in 1:num_outer_iter];

early_stop_counter = 0;


lq_approximation!(lq_approx, nonlinear_g, current_op);
for outer_iter = 1:num_outer_iter
    global x, u, γ, μ, λ, η, ψ, Mₜ, Nₜ, nₜ, Δ, α, ρ, new_loss, next_op, current_op, lq_approx, early_stop_counter, calibrated_γ

    if outer_iter == 1
        ρ_list[outer_iter] = ρ;
    else
        ρ_list[outer_iter] = ρ_list[outer_iter-1]*1/σ;
    end

    for t = 1:horizon+1
        if t == horizon+1
            calibrated_γ[t] =  [
                diagm(
                    nonlinear_g.terminal_inequality_constraints_list[1](current_op.x[t])
                )^(-1)*(ρ_list[outer_iter].*ones(inequality_constraints_size));
                diagm(
                    nonlinear_g.terminal_inequality_constraints_list[2](current_op.x[t])
                )^(-1)*(ρ_list[outer_iter].*ones(inequality_constraints_size));
            ]
        else
            calibrated_γ[t] =  [
            diagm(
                nonlinear_g.inequality_constraints_list[t][1](vcat(current_op.x[t], current_op.u[t]))
            )^(-1)*(ρ_list[outer_iter].*ones(inequality_constraints_size));
            diagm(
                nonlinear_g.inequality_constraints_list[t][2](vcat(current_op.x[t], current_op.u[t]))
            )^(-1)*(ρ_list[outer_iter].*ones(inequality_constraints_size))
        ]
        end
    end
    
    current_op = trajectory(
        x = current_op.x,
        u = current_op.u,
        λ = current_op.λ,
        η = current_op.η,
        ψ = current_op.ψ,
        μ = current_op.μ,
        γ = calibrated_γ
    );
    
    for iter in 1:num_iter
        # lq_approximation!(lq_approx, nonlinear_g, current_op)

        x, u, γ, μ, λ, η, ψ, Mₜ, Nₜ, nₜ = pdip_fbst_lq_solver!(
            π, 
            lq_approx, 
            current_op, 
            ρ_list[outer_iter], 
            false, 
            false
        )
        Δ = trajectory(
            x = x,
            u = u,
            λ = λ,
            η = η,
            ψ = ψ,
            μ = μ,
            γ = γ
        );
        
        Δ_list[outer_iter] = [Δ_list[outer_iter]; Δ];
        M_list[outer_iter] = [M_list[outer_iter]; [Mₜ]];
        N_list[outer_iter] = [N_list[outer_iter]; [Nₜ]];
        n_list[outer_iter] = [n_list[outer_iter]; [nₜ]];
        current_op_list[outer_iter] = [current_op_list[outer_iter]; current_op];

        α, new_loss, next_op, lq_approx, min_ineq, min_γ, new_homotopy_loss, α_γ = pdip_line_search(
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
        if α == 0.0
            break
        end
        current_op = next_op;
        loss_list[outer_iter][iter] = new_loss;
        homotopy_loss[outer_iter][iter] = new_homotopy_loss;
        α_list[outer_iter][iter] = α;
        α_γ_list[outer_iter][iter] = α_γ;
        min_ineq_list[outer_iter][iter] = min_ineq;
        min_γ_list[outer_iter][iter] = min_γ;
        # if abs(loss_list[outer_iter][iter] - ) < 1e-6
        #     early_stop_counter += 1;
        # else
        #     early_stop_counter = 0;
        # end

        
        # forward_simulation!(current_op, nonlinear_g);
    end
end
loss_list




