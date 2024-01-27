using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time
using Infiltrator
using BlockArrays
using ProgressBars
# this file is certified to be correct.


Q = I(2);
q = [1.0;1.0]; # optimal solution is -q

Ce = [1 0]; # Ce*x-Be = 0
Ci = [0 -1]; # Ci*x-Bi <= 0
Be = [1.0];
Bi = [-0.5];

num_iterations = 10;
num_outer_iterations = 20;
M = [[zeros(4,4) for t in 1:num_iterations] for iter in 1:num_outer_iterations];
n = [[zeros(4,1) for t in 1:num_iterations] for iter in 1:num_outer_iterations];
x = [[[1.0; 1.0] for t in 1:num_iterations+1] for iter in 1:num_outer_iterations];
ν = [[1. for t in 1:num_iterations+1] for iter in 1:num_outer_iterations];
λ = [[1.0 for t in 1:num_iterations+1] for iter in 1:num_outer_iterations];

mu = 2;
δx_list = [[[0.0; 0.0] for t in 1:num_iterations] for iter in 1:num_outer_iterations];
δλ_list = [[0.0 for t in 1:num_iterations] for iter in 1:num_outer_iterations];
δν_list = [[0.0 for t in 1:num_iterations] for iter in 1:num_outer_iterations];

α_list = [[0.0 for t in 1:num_iterations] for iter in 1:num_outer_iterations];

function linesearch(sol, n, x, λ, ν, t)
    α = 1.0;
    current_residual = norm(n, Inf)
    for i in 1:100
        x_next = x + α*sol[1:2];
        λ_next = λ + α*sol[3];
        ν_next = ν + α*sol[4];
        
        n_next = -[Q*x_next+q+Ci'*λ_next+Ce'*ν_next; 
            -λ_next*(Ci*x_next-Bi)-[1/t];
            Ce*x_next-Be
        ];
        residual = norm(n_next, Inf)
        if residual < current_residual && λ_next > 0
            return α, residual;
        end
        α = 0.5*α;
    end
    return α, current_residual;
end

t = 10.0;
for iter in 1:num_outer_iterations
    global t;
    if iter > 1
        x[iter][1] = x[iter-1][end];
        ν[iter][1] = ν[iter-1][end];
        λ[iter][1] = λ[iter-1][end];
    end
    
    for k in 1:num_iterations
        # The below is the KKT matrix for the constrained QP problem
        M[iter][k] = [
            Q Ci' Ce';
            -λ[iter][k]*Ci -(Ci*x[iter][k]-Bi) 0;
            Ce 0 0
        ];
        n[iter][k] = -[
            Q*x[iter][k]+q+Ci'*λ[iter][k]+Ce'*ν[iter][k]; 
            -λ[iter][k]*(Ci*x[iter][k]-Bi)-[1/t];
            Ce*x[iter][k]-Be
        ];
        sol = pinv(Array(M[iter][k]))*n[iter][k];
        δx, δλ, δν = sol[1:2], sol[3], sol[4];
        α, current_residual = linesearch(sol, n[iter][k], x[iter][k], λ[iter][k], ν[iter][k], t)
        # @infiltrate
        x[iter][k+1] = x[iter][k] + α*δx;
        λ[iter][k+1] = λ[iter][k] + α*δλ;
        ν[iter][k+1] = ν[iter][k] + α*δν;        
        δx_list[iter][k] = δx;
        δλ_list[iter][k] = δλ;
        δν_list[iter][k] = δν;
        
        α_list[iter][k] = α;
        
    end
    t = mu*t;
end





