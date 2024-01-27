# Primal-Dual Interior Point Method For Constrained Nonlinear Feedback Stackelberg Games

Dependency: Julia 1.9.0 - 1.9.2

Recommended actions:
1. make sure julia version is Julia 1.9.0 - 1.9.2
2. git clone <this_repo>
3. cd to FeedbackStackelbergGames.jl
4. install the dependency package by running in the terminal:
   1) "julia --project"
   2) once the Julia REPL is openned, enter the package management mode by typing "]"
   3) type "instantiate", and then press "enter".
   4) wait for julia to finish installing the necessary packages 


For highway experiments:
1. cd to the folder /example
2. run the following codes in the terminal: 
    "julia --project highway.jl"
3. wait, it may take a while to precompile the program, but after the precompilation, it will execute the algorithm super fast, returning a solution within 10 seconds.
4. data file and plots will be stored in a folder named after the date and time


Note that parameters of the PDIP algorithm, dynamics, costs, and constraints could be customized in the file /example/highway.jl

Limitations of the current implementation:
1. The KKT conditions construction process (, e.g., in /src/lq_solvers/constrained_fbst_lq_solver.jl) has not been optimized. We can speed it up by using StaticArray.jl and automatically constructing the first-order KKT conditions approximation using automatic differentiation. However, in the current implementation, we manually programmed the KKT conditions. 
2. In the current implementation, we assume each player has the same state, control, and constraint dimension. However, it should be relaxed in the future.



