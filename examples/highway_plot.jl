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

# # for data processing:
# file_name = "highway_2023-11-07-18:41:01.jld2"



# # file_name = "highway_2023-09-19-05:54:25.jld2";

# # file_name = "highway_2023-09-19-20:09:25.jld2";
# # file_name = "highway_2023-09-19-20:27:41.jld2";
# file_name = "highway_2023-09-23-08:59:11.jld2";





# # load the data from the JLD2 file
# data = load(file_name, "data");

# # access the variables in the data dictionary
# loss_list = data["loss_list"];
# homotopy_loss = data["homotopy_loss"];
# α_list = data["α_list"];
# α_γ_list = data["α_γ_list"];
# min_s_list = data["min_s_list"];
# min_γ_list = data["min_γ_list"];
# min_ineq_list = data["min_ineq_list"];
# current_op_list = data["current_op_list"];
# nonlinear_g = data["nonlinear_g"];
# lq_approx = data["lq_approx"];
# x0 = data["x0"];
# road_length = data["road_length"];
# base_x = data["base_x"];
# ineq_constraint_eps = data["ineq_constraint_eps"];
# ρ_list = data["ρ_list"];














# --------------------------------------------------------------------
# this code follows from highway.jl


now_str = Dates.format(now(), "yyyy-mm-dd-HH:MM:SS")
marker_size_list = 6*[0.98^(horizon+1-t) for t in 1:horizon+1]
alpha_list = [0.95^(horizon+1-t) for t in 1:horizon];
x_st = current_op.x;

folder_name = now_str
# for storing data:
data = Dict(
    "loss_list" => loss_list,
    "homotopy_loss" => homotopy_loss,
    "α_list" => α_list,
    "α_γ_list" => α_γ_list,
    "min_s_list" => min_s_list,
    "min_γ_list" => min_γ_list,
    "min_ineq_list" => min_ineq_list,
    "current_op_list" => current_op_list,
    "nonlinear_g" => nonlinear_g,
    "lq_approx" => lq_approx,
    "x0" => x0,
    "road_length" => road_length,
    "base_x" => base_x,
    "ineq_constraint_eps" => ineq_constraint_eps,
    "ρ_list" => ρ_list
)
# save the data to a JLD2 file
file_path = folder_name * "/" * "highway.jld2"
save(file_path, "data", data)





# simplified storage:
data = Dict(
    "loss_list" => loss_list,
    "homotopy_loss" => homotopy_loss,
    "α_list" => α_list,
    "α_γ_list" => α_γ_list,
    "min_s_list" => min_s_list,
    "min_γ_list" => min_γ_list,
    "min_ineq_list" => min_ineq_list,
    # "current_op_list" => current_op_list,
    # "nonlinear_g" => nonlinear_g,
    # "lq_approx" => lq_approx,
    "x0" => x0,
    "road_length" => road_length,
    "base_x" => base_x,
    "ineq_constraint_eps" => ineq_constraint_eps,
    "ρ_list" => ρ_list
)
# save the data to a JLD2 file
save(folder_name * "/" *"simple_data_highway.jld2", "data", data)







plot_size = (200, 400)

player1_color = :red
player2_color = :blue
player1_shape = :square
player2_shape = :square

# , color=player1_color
# , color=player2_color

# x_inclined_lines = [base_x, x_right_corner]
# y_inclined_lines = [road_length, y_right_corner]
upper_angle_list = LinRange(0, 2*segment_angle, 20)
lower_angle_list = LinRange(2*segment_angle, 0, 20)

upper_curve_x = base_x .+ radius .* (1 .- cos.(upper_angle_list))
upper_curve_y = road_length .- radius .* sin.(upper_angle_list)
lower_curve_x = x_right_corner .- radius .* (1 .- cos.(lower_angle_list))
lower_curve_y = y_right_corner .+ radius .* sin.(lower_angle_list)

x_inclined_lines = [upper_curve_x; lower_curve_x]
y_inclined_lines = [upper_curve_y; lower_curve_y]

x_upper_right_lines = [base_x, base_x]
y_upper_right_lines = [road_length, 5.0]
x_lower_right_lines = [x_right_corner, x_right_corner]
y_lower_right_lines = [y_right_corner, 0.0]
x_left_lines = [0.25, 0.25]
y_left_lines = [0.0, 5.0]
scatter([x_st[t][1] for t in 1:horizon], [x_st[t][2] for t in 1:horizon],markershape=player1_shape, markercolor=player1_color, label="", alpha=alpha_list)
scatter!([x_st[t][5] for t in 1:horizon], [x_st[t][6] for t in 1:horizon],markershape=player2_shape, markercolor=player2_color,label="", alpha=alpha_list)
scatter!([x_st[end][1]], [x_st[end][2]],markershape=player1_shape, label="player 1", markercolor=player1_color)
scatter!([x_st[end][5]], [x_st[end][6]],markershape=player2_shape, label="player 2", markercolor=player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "road edge", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L" ")
# scatter!([x_ne[t][1] for t in 1:horizon+1], [x_ne[t][2] for t in 1:horizon+1],markershape=:circle, label="player 1 fbne", markersize=marker_size_list)
# scatter!([x_ne[t][5] for t in 1:horizon+1], [x_ne[t][6] for t in 1:horizon+1],markershape=:circle, label="player 2 fbne", markersize=marker_size_list)
# title!("J1_st = $total_cost_J1_st, J2_st = $total_cost_J2_st, J1_ne = $total_cost_J1_ne, J2_ne = $total_cost_J2_ne")
savefig(folder_name * "/" *"highway_st.pdf")




# animation for fbst:

# anim1 = @animate for t = 1:horizon+1
#     scatter([x_st[t][1] for t in 1:t], [x_st[t][2] for t in 1:t],markershape=:circle, label="player 1 fbst", markersize=marker_size_list[1:t])
#     scatter!([x_st[t][5] for t in 1:t], [x_st[t][6] for t in 1:t],markershape=:circle, label="player 2 fbst", markersize=marker_size_list[1:t])
#     xlims!(0.0, 1.5)
#     ylims!(0.0, 5.0)
#     plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
#     plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
#     plot!(x_left_lines, y_left_lines, label = "road edge", color = :black, linewidth = 2, linestyle = :solid)
#     plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
#     title!("t = $t")
# end
# gif(anim1, folder_name * "/" *highway_st_$now_str.gif", fps = 10)




# plot initial trajectory:
x_st_init = current_op_list[1][1].x;
scatter([x_st_init[t][1] for t in 1:horizon], [x_st_init[t][2] for t in 1:horizon],markershape=player1_shape, color=player1_color,label="", alpha=alpha_list)
scatter!([x_st_init[t][5] for t in 1:horizon], [x_st_init[t][6] for t in 1:horizon],markershape=player2_shape,color=player2_color,label="", alpha=alpha_list)
scatter!([x_st_init[end][1]], [x_st_init[end][2]],markershape=player1_shape, label="", color = player1_color)
scatter!([x_st_init[end][5]], [x_st_init[end][6]],markershape=player2_shape, label="", color = player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L"p_y")
savefig(folder_name * "/" *"highway_st_init.pdf")


# animation for fbst:

# anim1 = @animate for t = 1:horizon+1
#     scatter([x_st_init[t][1] for t in 1:t], [x_st_init[t][2] for t in 1:t],markershape=:circle, label="player 1 fbst", markersize=marker_size_list[1:t])
#     scatter!([x_st_init[t][5] for t in 1:t], [x_st_init[t][6] for t in 1:t],markershape=:circle, label="player 2 fbst", markersize=marker_size_list[1:t])
#     xlims!(0.0, 1.5)
#     ylims!(0.0, 5.0)
#     plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
#     plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
#     plot!(x_left_lines, y_left_lines, label = "road edge", color = :black, linewidth = 2, linestyle = :solid)
#     plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
#     title!("t = $t")
# end
# gif(anim1, folder_name * "/" *"highway_st_init.gif", fps = 10)

# plot second trajectory:
x_st_second = current_op_list[1][2].x;
scatter([x_st_second[t][1] for t in 1:horizon], [x_st_second[t][2] for t in 1:horizon],markershape=player1_shape, color=player1_color,label="", alpha=alpha_list)
scatter!([x_st_second[t][5] for t in 1:horizon], [x_st_second[t][6] for t in 1:horizon],markershape=player2_shape,color=player2_color,label="", alpha=alpha_list)
scatter!([x_st_second[end][1]], [x_st_second[end][2]],markershape=player1_shape, label="", color = player1_color)
scatter!([x_st_second[end][5]], [x_st_second[end][6]],markershape=player2_shape, label="", color = player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L" ")
savefig(folder_name * "/" *"highway_st_second.pdf")


# animation for fbst:

# anim1 = @animate for t = 1:horizon+1
#     scatter([x_st_second[t][1] for t in 1:t], [x_st_second[t][2] for t in 1:t],markershape=:circle, label="player 1 fbst", markersize=marker_size_list[1:t])
#     scatter!([x_st_second[t][5] for t in 1:t], [x_st_second[t][6] for t in 1:t],markershape=:circle, label="player 2 fbst", markersize=marker_size_list[1:t])
#     xlims!(0.0, 1.5)
#     ylims!(0.0, 5.0)
#     plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
#     plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
#     plot!(x_left_lines, y_left_lines, label = "road edge", color = :black, linewidth = 2, linestyle = :solid)
#     plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
#     title!("t = $t")
# end
# gif(anim1, folder_name * "/" *"highway_st_second.gif", fps = 10)


# plot third trajectory:
x_st_third = current_op_list[1][3].x;
scatter([x_st_third[t][1] for t in 1:horizon], [x_st_third[t][2] for t in 1:horizon],markershape=player1_shape, color=player1_color,label="", alpha=alpha_list)
scatter!([x_st_third[t][5] for t in 1:horizon], [x_st_third[t][6] for t in 1:horizon],markershape=player2_shape,color=player2_color,label="", alpha=alpha_list)
scatter!([x_st_third[end][1]], [x_st_third[end][2]],markershape=player1_shape, label="", color = player1_color)
scatter!([x_st_third[end][5]], [x_st_third[end][6]],markershape=player2_shape, label="", color = player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L" ")
savefig(folder_name * "/" *"highway_st_third.pdf")

# plot the fourth trajectory:
x_st_fourth = current_op_list[1][4].x;
scatter([x_st_fourth[t][1] for t in 1:horizon], [x_st_fourth[t][2] for t in 1:horizon],markershape=player1_shape, color=player1_color,label="", alpha=alpha_list)
scatter!([x_st_fourth[t][5] for t in 1:horizon], [x_st_fourth[t][6] for t in 1:horizon],markershape=player2_shape,color=player2_color,label="", alpha=alpha_list)
scatter!([x_st_fourth[end][1]], [x_st_fourth[end][2]],markershape=player1_shape, label="", color = player1_color)
scatter!([x_st_fourth[end][5]], [x_st_fourth[end][6]],markershape=player2_shape, label="", color = player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L" ")
savefig(folder_name * "/" *"highway_st_fourth.pdf")



# plot the fifth trajectory:
x_st_fifth = current_op_list[1][5].x;
scatter([x_st_fifth[t][1] for t in 1:horizon], [x_st_fifth[t][2] for t in 1:horizon],markershape=player1_shape, color=player1_color,label="", alpha=alpha_list)
scatter!([x_st_fifth[t][5] for t in 1:horizon], [x_st_fifth[t][6] for t in 1:horizon],markershape=player2_shape,color=player2_color,label="", alpha=alpha_list)
scatter!([x_st_fifth[end][1]], [x_st_fifth[end][2]],markershape=player1_shape, label="", color = player1_color)
scatter!([x_st_fifth[end][5]], [x_st_fifth[end][6]],markershape=player2_shape, label="", color = player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L" ")
savefig(folder_name * "/" *"highway_st_fifth.pdf")



# plot the sixth trajectory:
x_st_sixth = current_op_list[1][6].x;
scatter([x_st_sixth[t][1] for t in 1:horizon], [x_st_sixth[t][2] for t in 1:horizon],markershape=player1_shape, color=player1_color,label="", alpha=alpha_list)
scatter!([x_st_sixth[t][5] for t in 1:horizon], [x_st_sixth[t][6] for t in 1:horizon],markershape=player2_shape,color=player2_color,label="", alpha=alpha_list)
scatter!([x_st_sixth[end][1]], [x_st_sixth[end][2]],markershape=player1_shape, label="", color = player1_color)
scatter!([x_st_sixth[end][5]], [x_st_sixth[end][6]],markershape=player2_shape, label="", color = player2_color)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
plot!(size = plot_size,left_margin=5Plots.mm, bottom_margin=5Plots.mm,grid=false)
xlabel!(L"p_x")
ylabel!(L" ")
savefig(folder_name * "/" *"highway_st_sixth.pdf")














# in what follows, we plot loss values.
# create a 2 by 2 subplot
grid_option = true
# plot loss vs. iterations for the first iteration
plot(1:length(loss_list[1]), log10.(loss_list[1]), 
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
plot(1:length(loss_list[2]), log10.(loss_list[2]), 
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
plot(1:length(loss_list[3]), log10.(loss_list[3]), 
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
plot(1:length(loss_list[5]), log10.(loss_list[5]), 
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
plot(1:length(loss_list[6]), log10.(loss_list[6]), 
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
plot(1:length(loss_list[10]), log10.(loss_list[10]), 
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




# # plot loss vs. iterations for the tenth iteration
# plot(1:length(loss_list[5]), log10.(loss_list[5]), 
#     label = "loss", color = :black, linewidth = 2, 
#     # left_margin=10Plots.mm,
#     # bottom_margin=5Plots.mm,
#     size = (300, 200),
#     linestyle = :solid
# )
# xlabel!("iterations")
# ylabel!("lg(K(z))")
# savefig("highway_st_loss_fifth_$now_str.png")

