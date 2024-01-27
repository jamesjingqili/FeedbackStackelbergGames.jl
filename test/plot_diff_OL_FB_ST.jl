# run OL_ST.jl
# run FB_ST.jl


# --------------------



# below is the plots!


using Plots
using Plots.PlotMeasures
marker_sizes = 4*[0.95^(T+1-t) for t in 1:T+1]
marker_alpha_list = [0.95^(T+1-t) for t in 1:T+1]
scatter([x_FB[t][1] for t in 1:T+1], [x_FB[t][2] for t in 1:T+1], label="player 1 FB", markersize=marker_sizes)
scatter!([x_FB[t][3] for t in 1:T+1], [x_FB[t][4] for t in 1:T+1], label="player 2 FB", markersize=marker_sizes)


scatter!([x_OL[t][1] for t in 1:T+1], [x_OL[t][2] for t in 1:T+1], label="player 1 OL", markersize=marker_sizes)
scatter!([x_OL[t][3] for t in 1:T+1], [x_OL[t][4] for t in 1:T+1], label="player 2 OL", markersize=marker_sizes)
savefig("test.png")



plot(0:T, [x_FB[t][1] for t in 1:T+1], label="x, player 1 FB", color=:red, linestyle=:solid)
plot!(0:T,[x_FB[t][2] for t in 1:T+1], label="y, player 1 FB", color=:red, linestyle=:dash)
plot!(0:T, [x_FB[t][3] for t in 1:T+1], label="x, player 2 FB", color=:blue, linestyle=:solid)
plot!(0:T,[x_FB[t][4] for t in 1:T+1], label="y, player 2 FB", color=:blue, linestyle=:dash)
savefig("OLFB_ST_FB.png")


plot(0:T, [x_OL[t][1] for t in 1:T+1], label="x, player 1 OL", color=:red, linestyle=:solid)
plot!(0:T,[x_OL[t][2] for t in 1:T+1], label="y, player 1 OL", color=:red, linestyle=:dash)
plot!(0:T,[x_OL[t][3] for t in 1:T+1], label="x, player 2 OL", color=:blue, linestyle=:solid)
plot!(0:T,[x_OL[t][4] for t in 1:T+1], label="y, player 2 OL", color=:blue, linestyle=:dash)
savefig("OLFB_ST_OL.png")




# --------------------------------
line_width = 2.0
plot_size = (400,300)
player1_color = :red
player2_color = :blue
player1_x_linestyle = :solid
player1_y_linestyle = :dash
player2_x_linestyle = :solid
player2_y_linestyle = :dash
# plot(0:T, [x_FB[t][1] for t in 1:T+1], label="x, player 1 FB", color=:black, linestyle=:solid, linewidth=line_width)
# plot!(0:T,[x_FB[t][2] for t in 1:T+1], label="y, player 1 FB", color=:black, linestyle=:dash, linewidth=line_width)
# plot!(0:T, [x_OL[t][1] for t in 1:T+1], label="x, player 1 OL", color=:black, linestyle=:dot, linewidth=line_width)
# plot!(0:T,[x_OL[t][2] for t in 1:T+1], label="y, player 1 OL", color=:black, linestyle=:dashdot, linewidth=line_width)
# plot!(size = plot_size,xlabel="t", ylabel="value", legend=(0.7,0.8),
#     bottom_margin=10mm, left_margin=10mm, right_margin=10mm,grid=false
# )
# savefig("OLFB_ST_p1.pdf")



# plot(0:T, [x_FB[t][3] for t in 1:T+1], label="x, player 2 FB", color=:black, linestyle=:solid,linewidth=line_width)
# plot!(0:T,[x_FB[t][4] for t in 1:T+1], label="y, player 2 FB", color=:black, linestyle=:dash,linewidth=line_width)
# plot!(0:T,[x_OL[t][3] for t in 1:T+1], label="x, player 2 OL", color=:black, linestyle=:dot,linewidth=line_width)
# plot!(0:T,[x_OL[t][4] for t in 1:T+1], label="y, player 2 OL", color=:black, linestyle=:dashdot,linewidth=line_width)
# plot!(size = plot_size,xlabel="t", ylabel="value", legend=(0.7,0.8),
#     bottom_margin=10mm, left_margin=10mm, right_margin=10mm,grid=false
# )
# savefig("OLFB_ST_p2.pdf")


scatter([x_FB[t][1] for t in 1:T+1], [x_FB[t][2] for t in 1:T+1], label="", color=:red, markershape=:square,
    seriesalpha=marker_alpha_list)
scatter!([x_FB[t][3] for t in 1:T+1], [x_FB[t][4] for t in 1:T+1], label="", color=:blue, markershape=:square,
    seriesalpha=marker_alpha_list)
scatter!([x_OL[t][1] for t in 1:T+1], [x_OL[t][2] for t in 1:T+1], label="", color=:red, markershape=:circle,
    seriesalpha=marker_alpha_list)
scatter!([x_OL[t][3] for t in 1:T+1], [x_OL[t][4] for t in 1:T+1], label="", color=:blue, markershape=:circle,
    seriesalpha=marker_alpha_list)

scatter!([x_FB[T+1][1]], [x_FB[T+1][2]], label="player 1 FSE", color=:red, markershape=:square)
scatter!([x_FB[T+1][3]], [x_FB[T+1][4]], label="player 2 FSE", color=:blue, markershape=:square)
scatter!([x_OL[T+1][1]], [x_OL[T+1][2]], label="player 1 RH-OLSE", color=:red, markershape=:circle)
scatter!([x_OL[T+1][3]], [x_OL[T+1][4]], label="player 2 RH-OLSE", color=:blue, markershape=:circle)
plot!(size = plot_size,xlabel="x position", ylabel="y position", legend=:bottomright,
    bottom_margin=10mm, left_margin=10mm, right_margin=10mm,grid=false
)
savefig("OLFB_ST.pdf")


plot(0:T, [x_FB[t][1] for t in 1:T+1], label="x, player 1", color=player1_color, linestyle=player1_x_linestyle, linewidth=line_width)
plot!(0:T,[x_FB[t][2] for t in 1:T+1], label="y, player 1", color=player1_color, linestyle=player1_y_linestyle, linewidth=line_width)
plot!(0:T,[x_FB[t][3] for t in 1:T+1], label="x, player 2", color=player2_color, linestyle=player2_x_linestyle, linewidth=line_width)
plot!(0:T,[x_FB[t][4] for t in 1:T+1], label="y, player 2", color=player2_color, linestyle=player2_y_linestyle, linewidth=line_width)
plot!(size = plot_size,xlabel="t", ylabel="position", legend=(0.7,0.8),
    bottom_margin=10mm, left_margin=10mm, right_margin=10mm,grid=true, ylims=(-0.0,2.1)
)
savefig("FB_ST.pdf")

plot(0:T, [x_OL[t][1] for t in 1:T+1], label="x, player 1", color=player1_color, linestyle=player1_x_linestyle,linewidth=line_width)
plot!(0:T,[x_OL[t][2] for t in 1:T+1], label="y, player 1", color=player1_color, linestyle=player1_y_linestyle,linewidth=line_width)
plot!(0:T,[x_OL[t][3] for t in 1:T+1], label="x, player 2", color=player2_color, linestyle=player2_x_linestyle,linewidth=line_width)
plot!(0:T,[x_OL[t][4] for t in 1:T+1], label="y, player 2", color=player2_color, linestyle=player2_y_linestyle,linewidth=line_width)
plot!(size = plot_size,xlabel="t", ylabel="position", legend=(0.7,0.8),
    bottom_margin=10mm, left_margin=10mm, right_margin=10mm,grid=true,ylims=(-0.0,2.1)
)
savefig("OL_ST.pdf")



# above is the plots!
# --------------------

marker_sizes = 4*[0.99^(T+1-t) for t in 1:T+1]

anim1 = @animate for t = 1:horizon+1
    scatter([x_FB[t][1] for t in 1:t], [x_FB[t][2] for t in 1:t],markershape=:circle, label="player 1 FB", markersize=marker_sizes[1:t])
    scatter!([x_FB[t][3] for t in 1:t], [x_FB[t][4] for t in 1:t],markershape=:circle, label="player 2 FB", markersize=marker_sizes[1:t])
    xlims!(-0.2,2.2)
    ylims!(-0.2,2.2)
    title!("t = $t")
end
gif(anim1, "FB_LQ_diff.gif", fps = 10)


# animation for fbne:

anim2 = @animate for t = 1:horizon+1
    scatter([x_OL[t][1] for t in 1:t], [x_OL[t][2] for t in 1:t],markershape=:circle, label="player 1 OL", markersize=marker_sizes[1:t])
    scatter!([x_OL[t][3] for t in 1:t], [x_OL[t][4] for t in 1:t],markershape=:circle, label="player 2 OL", markersize=marker_sizes[1:t])
    xlims!(-0.2,2.2)
    ylims!(-0.2,2.2)
    title!("t = $t")
end
gif(anim2, "OL_LQ_diff.gif", fps = 10)




