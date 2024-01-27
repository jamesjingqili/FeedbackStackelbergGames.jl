using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time
using Infiltrator
using BlockArrays
using ProgressBars
# this file is certified to be correct.


Q = I(2);
q = [0.0;0.0]; # solution is -q

Ce = [1 0]; # Ce*x = Be
Ci = [0 1]; # Ci*x >= Bi
Be = [1.0];
Bi = [0.5];

num_iterations = 10;
num_outer_iterations = 10;
M = [[zeros(5,5) for t in 1:num_iterations] for iter in 1:num_outer_iterations];
n = [[zeros(5,1) for t in 1:num_iterations] for iter in 1:num_outer_iterations];
x = [[[1.0; 1.0] for t in 1:num_iterations+1] for iter in 1:num_outer_iterations];
y = [[1.0 for t in 1:num_iterations+1] for iter in 1:num_outer_iterations];
s = [[0.2 for t in 1:num_iterations+1] for iter in 1:num_outer_iterations];
z = [[1.0 for t in 1:num_iterations+1] for iter in 1:num_outer_iterations];
mu = [0.2 for iter in 1:num_outer_iterations];
δs_list = [[0.0 for t in 1:num_iterations] for iter in 1:num_outer_iterations];
δz_list = [[0.0 for t in 1:num_iterations] for iter in 1:num_outer_iterations];
δx_list = [[[0.0; 0.0] for t in 1:num_iterations] for iter in 1:num_outer_iterations];
δy_list = [[0.0 for t in 1:num_iterations] for iter in 1:num_outer_iterations];
αs_list = [[0.0 for t in 1:num_iterations] for iter in 1:num_outer_iterations];
αz_list = [[0.0 for t in 1:num_iterations] for iter in 1:num_outer_iterations];
α_list = [[0.0 for t in 1:num_iterations] for iter in 1:num_outer_iterations];

function linesearch(sol, n, x, y, s, z, mu)
    α = 1.0;
    current_residual = norm(n, 2)
    for i in 1:100
        x_next = x + α*sol[1:2];
        y_next = y + α*sol[3];
        s_next = s + α*sol[4];
        z_next = z + α*sol[5];

        n_next = [Q*x_next+q-Ce'*y_next-Ci'*z_next; 
            s_next*z_next-mu; 
            Ce*x_next-Be; 
            Ci*x_next-Bi-[s_next]
        ];
        residual = norm(n_next, 2)
        if residual < current_residual && s_next > 0 && z_next > 0
            return α, residual;
        end
        α = 0.5*α;
    end
    return α, current_residual;
end


for iter in 1:num_outer_iterations
    if iter > 1
        x[iter][1] = x[iter-1][end];
        y[iter][1] = y[iter-1][end];
        s[iter][1] = s[iter-1][end];
        z[iter][1] = z[iter-1][end];
    end
    
    for k in 1:num_iterations
        M[iter][k] = [
            Q zeros(2,1) -Ce' -Ci';
            zeros(1,2) z[iter][k] 0 s[iter][k];
            Ce 0 0 0;
            Ci -1.0 0 0
        ];
        n[iter][k] = -[
            Q*x[iter][k]+q-Ce'*y[iter][k]-Ci'*z[iter][k];
            z[iter][k]*s[iter][k]-mu[iter];
            Ce*x[iter][k]-Be;
            Ci*x[iter][k]-Bi-[s[iter][k]]
        ];
        sol = pinv(Array(M[iter][k]))*n[iter][k];
        δx = sol[1:2];
        δy = sol[3];
        δs = sol[4];
        δz = sol[5];
        α, current_residual = linesearch(sol, n[iter][k], x[iter][k], y[iter][k], s[iter][k],z[iter][k], mu[iter]);
        # α = 0.001;
        
        x[iter][k+1] = x[iter][k] + α*δx;
        s[iter][k+1] = s[iter][k] + α*δs;
        y[iter][k+1] = y[iter][k] + α*δy;
        z[iter][k+1] = z[iter][k] + α*δz;
        
        δx_list[iter][k] = δx;
        δy_list[iter][k] = δy;
        δs_list[iter][k] = δs;
        δz_list[iter][k] = δz;
        
        α_list[iter][k] = α;
        
    end
    if iter < num_outer_iterations
        mu[iter+1] = 0.2*mu[iter];
    end
end





