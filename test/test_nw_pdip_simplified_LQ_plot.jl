using JLD2
using Dates
# this file is certified to be correct.






file_name = "";
# load the data from the JLD2 file
data = load(file_name, "data");

# access the variables in the data dictionary
loss_list = data["loss_list"];
homotopy_loss = data["homotopy_loss"];
α_list = data["α_list"];
α_γ_list = data["α_γ_list"];
min_s_list = data["min_s_list"];
min_γ_list = data["min_γ_list"];
min_ineq_list = data["min_ineq_list"];
current_op_list = data["current_op_list"];
nonlinear_g = data["nonlinear_g"];
lq_approx = data["lq_approx"];
x0 = data["x0"];
road_length = data["road_length"];
base_x = data["base_x"];
ineq_constraint_eps = data["ineq_constraint_eps"];
ρ_list = data["ρ_list"];















# this code follows from test_nw_pdip_simplified_LQ.jl


now_str = Dates.format(now(), "yyyy-mm-dd-HH:MM:SS")
marker_size_list = 6*[0.98^(horizon+1-t) for t in 1:horizon+1]
alpha_list = [0.98^(horizon+1-t) for t in 1:horizon];
x_st = current_op.x;


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
save("highway_$now_str.jld2", "data", data)





scatter([x_st[t][1] for t in 1:horizon], [x_st[t][2] for t in 1:horizon],markershape=:circle, color=:red,label="", alpha=alpha_list)
scatter!([x_st[t][5] for t in 1:horizon], [x_st[t][6] for t in 1:horizon],markershape=:circle,color=:blue,label="", alpha=alpha_list)
scatter!([x_st[end][1]], [x_st[end][2]],markershape=:circle, label="player 1", color = :red)
scatter!([x_st[end][5]], [x_st[end][6]],markershape=:circle, label="player 2", color = :blue)
plot!(size = (300, 600))
savefig("highway_st_$now_str.png")


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
# gif(anim1, "highway_st_$now_str.gif", fps = 10)




# 










# plot initial trajectory:
x_st_init = current_op_list[1][1].x;
scatter([x_st_init[t][1] for t in 1:horizon], [x_st_init[t][2] for t in 1:horizon],markershape=:circle, color=:red,label="", alpha=alpha_list)
scatter!([x_st_init[t][5] for t in 1:horizon], [x_st_init[t][6] for t in 1:horizon],markershape=:circle,color=:blue,label="", alpha=alpha_list)
scatter!([x_st_init[end][1]], [x_st_init[end][2]],markershape=:circle, label="", color = :red)
scatter!([x_st_init[end][5]], [x_st_init[end][6]],markershape=:circle, label="", color = :blue)
plot!(size = (300, 200))
savefig("highway_st_init_$now_str.png")


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
# gif(anim1, "highway_st_init_$now_str.gif", fps = 10)

# plot second trajectory:
x_st_second = current_op_list[1][2].x;
scatter([x_st_second[t][1] for t in 1:horizon], [x_st_second[t][2] for t in 1:horizon],markershape=:circle, color=:red,label="", alpha=alpha_list)
scatter!([x_st_second[t][5] for t in 1:horizon], [x_st_second[t][6] for t in 1:horizon],markershape=:circle,color=:blue,label="", alpha=alpha_list)
scatter!([x_st_second[end][1]], [x_st_second[end][2]],markershape=:circle, label="", color = :red)
scatter!([x_st_second[end][5]], [x_st_second[end][6]],markershape=:circle, label="", color = :blue)
plot!(size = (300, 200))
savefig("highway_st_second_$now_str.png")


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
# gif(anim1, "highway_st_second_$now_str.gif", fps = 10)


# plot third trajectory:
x_st_third = current_op_list[1][3].x;
scatter([x_st_third[t][1] for t in 1:horizon], [x_st_third[t][2] for t in 1:horizon],markershape=:circle, color=:red,label="", alpha=alpha_list)
scatter!([x_st_third[t][5] for t in 1:horizon], [x_st_third[t][6] for t in 1:horizon],markershape=:circle,color=:blue,label="", alpha=alpha_list)
scatter!([x_st_third[end][1]], [x_st_third[end][2]],markershape=:circle, label="", color = :red)
scatter!([x_st_third[end][5]], [x_st_third[end][6]],markershape=:circle, label="", color = :blue)
plot!(size = (300, 200))
savefig("highway_st_third_$now_str.png")
















# create a 2 by 2 subplot

# plot loss vs. iterations for the first iteration
plot(1:length(loss_list[1]), log.(loss_list[1]), 
    label = "", color = :black, linewidth = 2, 
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm,
    size = (300, 200),
    linestyle = :solid
)
xlabel!("iterations")
ylabel!("log(loss)")
savefig("highway_st_loss_first_$now_str.png")


# plot loss vs. iterations for the second iteration
plot(1:length(loss_list[2]), log.(loss_list[2]), 
    label = "", color = :black, linewidth = 2, 
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm,
    size = (300, 200),
    linestyle = :solid
)
xlabel!("iterations")
ylabel!("log(loss)")
savefig("highway_st_loss_second_$now_str.png")



# plot loss vs. iterations for the third iteration
plot(1:length(loss_list[3]), log.(loss_list[3]), 
    label = "", color = :black, linewidth = 2, 
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm,
    size = (300, 200),
    linestyle = :solid
)
xlabel!("iterations")
ylabel!("log(loss)")
savefig("highway_st_loss_third_$now_str.png")


# plot loss vs. iterations for the fifth iteration
plot(1:length(loss_list[5]), log.(loss_list[5]), 
    label = "", color = :black, linewidth = 2, 
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm,
    size = (300, 200),
    linestyle = :solid
)
xlabel!("iterations")
ylabel!("log(loss)")
savefig("highway_st_loss_fifth_$now_str.png")


# plot loss vs. iterations for the tenth iteration
plot(1:length(loss_list[10]), log.(loss_list[10]), 
    label = "loss", color = :black, linewidth = 2, 
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm,
    size = (300, 200),
    linestyle = :solid
)
xlabel!("iterations")
ylabel!("log(loss)")
savefig("highway_st_loss_tenth_$now_str.png")





