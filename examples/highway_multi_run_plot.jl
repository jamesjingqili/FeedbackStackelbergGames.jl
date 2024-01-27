using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time
using Infiltrator
# this file is certified to be correct.

using JLD2
using Dates
using LaTeXStrings
using FilePathsBase
using Statistics


# check if there dones have an 0.0 in the i-th row of list_loss_list
list_converged_x0 = [];
list_of_converged_index = [];
for ii in 1:num_samples
    if 0.0 ∉ list_loss_list[ii, :, :]
        push!(list_converged_x0, x0_list[:,ii])
        push!(list_of_converged_index, ii)
    end
end

list_of_converged_index = list_of_converged_index[1:10];
converged_loss_list = list_loss_list[list_of_converged_index,:,:];
converged_x0_list = x0_list[:,list_of_converged_index];

time_list_converged = list_time_list[list_of_converged_index,:,:];


number_of_converged = length(list_of_converged_index);
num_iter = size(converged_loss_list)[2];
# compute the mean and the variance of computation time for those converged case
# for each rho, across different initializations:
mean_time_list = zeros(number_of_converged, num_iter);
var_time_list = zeros(number_of_converged, num_iter);
total_time_list = zeros(number_of_converged, num_iter);
for outer_iter in 1:length(list_of_converged_index)
    for iter in 1:length(list_time_list[outer_iter, :, 1])
        total_time_list[outer_iter, iter] = sum(list_time_list[outer_iter, iter,:]);
        mean_time_list[outer_iter, iter] = mean(list_time_list[outer_iter, iter,:]);
        var_time_list[outer_iter, iter] = var(list_time_list[outer_iter, iter,:]);
    end
end


# compute the mean of the time of across different initializations and rho and iterations:
mean_time_list_across_rho = zeros(num_iter);
var_time_list_across_rho = zeros(num_iter);
total_time_list_across_rho = zeros(num_iter);
for iter in 1:num_iter
    total_time_list_across_rho[iter] = sum(total_time_list[:, iter]);
    mean_time_list_across_rho[iter] = mean(total_time_list[:, iter]);
    var_time_list_across_rho[iter] = var(total_time_list[:, iter]);
end





# save x_list and list_loss_list, under the name of the current time and highway_multi_run:
now = Dates.now()
now_str = Dates.format(now, "yyyy-mm-dd_HH:MM:SS")

save("highway_parallel_"*now_str*".jld2", 
"x0_list", converged_x0_list, 
"list_loss_list", converged_loss_list)



folder_name = now_str*"_multi_run"
mkdir(folder_name)
plot_size = (200, 400)

player1_color = :red
player2_color = :blue
player1_shape = :square
player2_shape = :square
fillalpha = 0.2


mean_loss_list = zeros(length(list_of_converged_index),10)
std_loss_list = zeros(length(list_of_converged_index),10)
for i in 1:length(list_of_converged_index)
    mean_loss_list[i,:] = mean(log10.(converged_loss_list[:,i,:]), dims=1)
    std_loss_list[i,:] = std(log10.(converged_loss_list[:,i,:]), dims=1)
end


# in what follows, we plot loss values.
# create a 2 by 2 subplot
grid_option = true
# plot loss vs. iterations for the first iteration
plot(1:length(mean_loss_list[1,:]), mean_loss_list[1,:],
    ribbon = std_loss_list[1,:], fillalpha=fillalpha,
    label = "", color = :black, linewidth = 2, 
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm,
    size = (300, 200),
    linestyle = :solid,
    grid=grid_option
)
xlabel!("iterations "*L"k")
ylabel!(L"\lg(\|\|K_\rho(\mathbf{z}^{(k)})\|\|_2)")
savefig(folder_name * "/" *"highway_st_loss_first.pdf")


# plot loss vs. iterations for the second iteration
plot(1:length(mean_loss_list[1,:]), mean_loss_list[2,:],
    ribbon = std_loss_list[2,:], fillalpha=fillalpha,
    label = "", color = :black, linewidth = 2, 
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm,
    size = (300, 200),
    linestyle = :solid,
    grid=grid_option
)
xlabel!("iterations "*L"k")
# ylabel!(L"\lg(K(\mathbf{z}^{(k)}))")
savefig(folder_name * "/" *"highway_st_loss_second.pdf")



# plot loss vs. iterations for the third iteration
plot(1:length(mean_loss_list[1,:]), mean_loss_list[3,:],
    ribbon = std_loss_list[3,:], fillalpha=fillalpha,
    label = "", color = :black, linewidth = 2, 
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm,
    size = (300, 200),
    linestyle = :solid,
    grid=grid_option
)
xlabel!("iterations "*L"k")
# ylabel!(L"\lg(K(\mathbf{z}^{(k)}))")
savefig(folder_name * "/" *"highway_st_loss_third.pdf")


# plot loss vs. iterations for the fifth iteration
plot(1:length(mean_loss_list[1,:]), mean_loss_list[5,:],
    ribbon = std_loss_list[5,:], fillalpha=fillalpha,
    label = "", color = :black, linewidth = 2, 
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm,
    size = (300, 200),
    linestyle = :solid,
    grid=grid_option
)
xlabel!("iterations "*L"k")
# ylabel!(L"\lg(K(\mathbf{z}^{(k)}))")
savefig(folder_name * "/" *"highway_st_loss_fifth.pdf")

# plot loss vs. iterations for the sixth iteration
plot(1:length(mean_loss_list[1,:]), mean_loss_list[6,:],
    ribbon = std_loss_list[6,:], fillalpha=fillalpha,
    label = "loss", color = :black, linewidth = 2, 
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm,
    size = (300, 200),
    linestyle = :solid,
    grid=grid_option
)
xlabel!("iterations "*L"k")
# ylabel!(L"\lg(K(\mathbf{z}^{(k)}))")
savefig(folder_name * "/" *"highway_st_loss_sixth.pdf")



# plot loss vs. iterations for the tenth iteration
plot(1:length(mean_loss_list[1,:]), mean_loss_list[10,:],
    ribbon = std_loss_list[10,:], fillalpha=fillalpha,
    label = "merit function value", color = :black, linewidth = 2, 
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm,
    size = (300, 200),
    linestyle = :solid,
    grid=grid_option
)
xlabel!("iterations "*L"k")
# ylabel!(L"\lg(K(\mathbf{z}^{(k)}))")
savefig(folder_name * "/" *"highway_st_loss_tenth.pdf")


