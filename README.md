# An Efficient Julia Solver for Feedback Stackelberg Games

Dependency: Julia 1.9.0 - 1.9.2

For highway experiments:
1. cd to the folder /example
2. run the following codes in terminal: 
    "julia --project highway.jl"
3. data file and plots will be stored in a folder named after the date and time


Note that parameters of the PDIP algorithm, dynamics, costs, and constraints could be customized in the file /example/highway.jl

Limitations of the current implementation:
1. The KKT conditions construction process (, e.g., in /src/lq_solvers/constrained_fbst_lq_solver.jl) has not been optimized. We can speed it up by using StaticArray.jl and also automatically constructing the first-order KKT conditions approximation by using automatic differentiation. However, in the current implementation, we manually programmed the KKT conditions. 
2. In the current implementation, we assume that each player has the same dimension of the state, control, and constraints. However, it should be relaxed in future.



