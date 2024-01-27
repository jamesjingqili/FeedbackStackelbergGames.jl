using LinearAlgebra
using Infiltrator
using StaticArrays
using iLQGames_discrete_time
using Infiltrator
# this file is certified to be correct.

using JLD2
using Dates



# for data processing:


# file_name = "highway_2023-09-19-05:54:25.jld2";

file_name = "highway_2023-09-19-20:09:25.jld2";
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





x_right_corner = 1.1; 
y_right_corner = 2.0;
base_x = 0.7; 
road_length = 4.0;

x_inclined_lines = [base_x, x_right_corner]
y_inclined_lines = [road_length, y_right_corner]

x_upper_right_lines = [base_x, base_x]
y_upper_right_lines = [road_length, 5.0]

x_lower_right_lines = [x_right_corner, x_right_corner]
y_lower_right_lines = [y_right_corner, 0.0]

x_left_lines = [0.25, 0.25]
y_left_lines = [0.0, 5.0]


plot(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "road edge", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
xlabel!("x")
ylabel!("y")
plot!(size = (300, 600))
savefig("see.png")






x_st = current_op_list[end][end].x;
alpha_list = [0.98^(horizon+1-t) for t in 1:horizon];
scatter([x_st[t][1] for t in 1:horizon], [x_st[t][2] for t in 1:horizon],markershape=:circle, label="", color=:red, alpha = alpha_list)
scatter!([x_st[t][5] for t in 1:horizon], [x_st[t][6] for t in 1:horizon],markershape=:circle, label="", color=:blue, alpha = alpha_list)
scatter!([x_st[end][1]], [x_st[end][2]],markershape=:circle, label="player 1", color = :red)
scatter!([x_st[end][5]], [x_st[end][6]],markershape=:circle, label="player 2", color = :blue)
plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_left_lines, y_left_lines, label = "road edge", color = :black, linewidth = 2, linestyle = :solid)
plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
xlims!(0.0, 1.5)
ylims!(0.0, 5.0)
xlabel!("x")
ylabel!("y")
plot!(size = (300, 600))
savefig("highway_st_finalized.png")



anim1 = @animate for t = 1:horizon+1
    # if t == horizon+1
    #     scatter([x_st[end][1]], [x_st[end][2]],markershape=:circle, label="player 1", color = :red)
    #     scatter!([x_st[end][5]], [x_st[end][6]],markershape=:circle, label="player 2", color = :blue)
    # else
    #     scatter([x_st[t][1] for t in 1:t], [x_st[t][2] for t in 1:t],color=:red, markershape=:circle, label="player 1 fbst", alpha=alpha_list[1:t])
    #     scatter!([x_st[t][5] for t in 1:t], [x_st[t][6] for t in 1:t],color=:blue, markershape=:circle, label="player 2 fbst", alpha=alpha_list[1:t])    
    # end
    
    title!("t = $t")
    plot!(x_upper_right_lines, y_upper_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
    plot!(x_lower_right_lines, y_lower_right_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
    plot!(x_left_lines, y_left_lines, label = "road edge", color = :black, linewidth = 2, linestyle = :solid)
    plot!(x_inclined_lines, y_inclined_lines, label = "", color = :black, linewidth = 2, linestyle = :solid)
    xlims!(0.0, 1.5)
    ylims!(0.0, 5.0)
    xlabel!("x")
    ylabel!("y")
    plot!(size = (300, 600))
end
gif(anim1, "highway_st_finalized.gif", fps = 10)


