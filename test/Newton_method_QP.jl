using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time
using Infiltrator
using BlockArrays
# this file is certified to be correct.


Q = I(2);
q = -[2.0;3.0];

Ce = [1 0]; # Ce*x = Be
Be = [1.0];

num_iterations = 100000;
M = [zeros(3,3) for t in 1:num_iterations];
n = [zeros(3,1) for t in 1:num_iterations];
x = [[-1.5; -1.0] for t in 1:num_iterations+1];
y = [1.0 for t in 1:num_iterations+1];
# s = [1.0 for t in 1:num_iterations+1];
# z = [1.0 for t in 1:num_iterations+1];
# mu=1;

for k in 1:num_iterations
    global δx, δy, δs, δz, αs, αz;
    # The below is the KKT matrix for the constrained QP problem
    M[k] = [
        Q -Ce';
        Ce 0;
    ];
    # @infiltrate
    n[k] = -[
        Q*x[k]+q - Ce'*y[k];
        Ce*x[k] - Be;
    ];

    sol = pinv(Array(M[k]))*n[k];
    δx = sol[1:2];
    δy = sol[3];
    # @infiltrate
    # αs = 3/4*min(1.0, 0.995*s[k]/δs);
    # αz = 3/4*min(1.0, 0.995*z[k]/δz);
    αs = 0.001;
    αz = 0.001;
    x[k+1] = x[k] + αs*δx;
    # s[k+1] = s[k] + αs*δs;
    
    y[k+1] = y[k] + αz*δy;
    # z[k+1] = z[k] + αz*δz;
end



