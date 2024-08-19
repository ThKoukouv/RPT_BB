using LinearAlgebra, Statistics, StatsBase, Distributions, JuMP, DelimitedFiles
using Gurobi, Random, Ipopt, Mosek, MosekTools, Convex, SCS, SCIP, CSV, DataFrames

function generate_hyperplane_eigen(x_opt,X_opt)
    n_x = size(X_opt,1)
    Y = X_opt .- x_opt*x_opt'
    (λ,U) = eigen(Y)
    f = U[:,n_x]
    l = f'*x_opt
    return f, l
end

function calculate_mc_input(x,y,U)
    X = [x]
    for j in 1:size(y)[1]
        if y[j] == 0
            push!(X,x)
        else
            push!(X,U[:,j]/y[j])
        end
    end
    return X
end

function calculate_different_vectors(X,x)
    Y = x
    for i in 1:length(X)
        if norm(X[i] .- x) > 1e-3
            Y = hcat(Y,X[i])
        end
    end
    return Y
end

function solve_rpt_relaxation_lse(C,d,use_lmi)
    L,n = size(C)
    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, y[1:n]>=0)
    @variable(m1, w[1:n])
    @variable(m1, X[1:n,1:n]>=0, Symmetric)
    @variable(m1, Y[1:n,1:n]>=0, Symmetric)
    @variable(m1, W[1:n,1:n], Symmetric)
    @variable(m1, U[1:n,1:n]>=0)
    @variable(m1, Q[1:n,1:n])
    @variable(m1, P[1:n,1:n])
    @variable(m1, x_aux_1[1:n,1:n])
    @variable(m1, y_aux_1[1:n,1:n])
    @variable(m1, Y_aux_1[1:n,1:n])
    @variable(m1, U_aux_1[1:n,1:n])
    @variable(m1, Q_aux_1[1:n,1:n])
    @variable(m1, P_aux_1[1:n,1:n])
    @constraint(m1, [i in 1:n], x_aux_1[:,i] .== x)
    @constraint(m1, [i in 1:n], y_aux_1[:,i] .== y)
    @constraint(m1, Y_aux_1 .== Y)
    @constraint(m1, U_aux_1 .== U)
    @constraint(m1, Q_aux_1 .== Q)
    @constraint(m1, P_aux_1 .== P)

    @constraint(m1, [i in 1:n], W[i,i] >= 0)

    @constraint(m1, C*x .<= d)
    @constraint(m1, [j in 1:n], C*X[:,j] .<= x[j]*d)
    @constraint(m1, [j in 1:n], C*U[:,j] .<= y[j]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')
    @constraint(m1, [i in 1:n], [w[i], y[i], 1] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n, j in 1:n], [Q_aux_1[j,i], U_aux_1[j,i], x_aux_1[j,i]] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n, j in 1:n], [P_aux_1[j,i], Y_aux_1[i,j], y_aux_1[j,i]] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n, j in 1:L], [d[j]*w[i] - C[j,:]'*Q[:,i],
                                           d[j]*y[i] - C[j,:]'*U[:,i],
                                           d[j] - C[j,:]'*x] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n, j in i:n], [P[i,j]+P[j,i],Y[i,j],1] in MOI.ExponentialCone())

    @constraint(m1, sum(y[i] for i in 1:n) == 1)
    @constraint(m1, sum(U[:,i] for i in 1:n) .== x)
    @constraint(m1, sum(Y[:,i] for i in 1:n) .== y)
    @constraint(m1, sum(P[i,:] for i in 1:n) .== w)

    if use_lmi
        @constraint(m1, [X U Q x; U' Y P y; Q' P' W w; x' y' w' 1] in PSDCone())
    end

    @objective(m1, Min, -tr(U) - sum(w[i] for i in 1:n))
    optimize!(m1)

    if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.SLOW_PROGRESS
        return true, JuMP.value.(x), JuMP.value.(y), JuMP.value.(X), JuMP.value.(Y), JuMP.value.(U), objective_value(m1)
    else
        return false, zeros(n), zeros(n), zeros(n,n), zeros(n,n), zeros(n,n), 1e6
    end
end

function solve_inner_x_lse(y,C,d)
    n = size(C,2)
    m1 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m1, "OutputFlag", 0)
    @variable(m1, x[1:n]>=0)
    @constraint(m1, C*x .<= d)
    @objective(m1, Min, -x'*y)
    optimize!(m1)
    return JuMP.value.(x)
end

function solve_inner_y_lse(x)
    n = size(x,1)
    m1 = Model(Mosek.Optimizer)
    @variable(m1, y[1:n]>=0)
    @variable(m1, w[1:n])
    @constraint(m1, sum(y[i] for i in 1:n) == 1)
    @constraint(m1, [i in 1:n], [w[i], y[i], 1] in MOI.ExponentialCone())
    @objective(m1, Min, -x'*y - sum(w[i] for i in 1:n))
    optimize!(m1)
    return JuMP.value.(y)
end

function mountain_climbing_lse(X,C,d)
    L = []
    for x in X
        y = solve_inner_y_lse(x)
        eps = 1
        while abs(eps) > 0.001
            Ub = -x'*y + sum(y[i]*log(y[i]) for i in 1:n)
            x = solve_inner_x_lse(y,C,d)
            y = solve_inner_y_lse(x)
            Ubx = -x'*y + sum(y[i]*log(y[i]) for i in 1:n)
            eps = Ubx - Ub
        end
        push!(L,[x,y])
    end
    ind_min = ind_min = argmin([-L[i][1]'*L[i][2] + sum(L[i][2][j]*log(L[i][2][j]) for j in 1:n) for i in 1:length(L)])
    x_mc, y_mc = L[ind_min][1], L[ind_min][2]
    return x_mc, y_mc, -x_mc'*y_mc + sum(y_mc[i]*log(y_mc[i]) for i in 1:n)
end

function rpt_bb_lse(C_init,d_init,δ,use_lmi)
    C, d = C_init, d_init
    gen_hyper = 0

    # Root Node
    res_root, x_lb, y_lb, X_lb, Y_lb, U_lb, lb = solve_rpt_relaxation_lse(C,d,use_lmi)
    X = calculate_mc_input(x_lb,y_lb,U_lb)
    x_ub, y_ub, ub = mountain_climbing_lse(X,C,d)
    Y_split = calculate_different_vectors(X,x_lb)

    x_cur, X_cur = x_lb, X_lb

    UB, LB, opt_sol, opt_val = ub, lb, x_lb, ub

    nodes_list = []

    t0_1 = time_ns()
    total_time = 0.0

    while UB - LB > δ && total_time < 3600
        f_opt, l_opt = generate_hyperplane_eigen(x_cur,X_cur)

        f_r, l_r, f_l, l_l  = f_opt, l_opt, -f_opt, -l_opt
        C_r, d_r, C_l, d_l = vcat(C,f_r'), vcat(d,l_r), vcat(C,f_l'), vcat(d,l_l)
        gen_hyper += 1

        # Right child
        res_r, x_lb_r, y_lb_r, X_lb_r, Y_lb_r, U_lb_r, lb_r = solve_rpt_relaxation_lse(C_r,d_r,use_lmi)
        if res_r
            X_r = calculate_mc_input(x_lb_r,y_lb_r,U_lb_r)
            Y_r = calculate_different_vectors(X_r,x_lb_r)
            x_ub_r, y_ub_r, ub_r = mountain_climbing_lse(X_r,C_r,d_r)
            if lb_r < UB
                push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r,Y_r])

            end
        end
        # Left child
        res_l, x_lb_l, y_lb_l, X_lb_l, Y_lb_l, U_lb_l, lb_l = solve_rpt_relaxation_lse(C_l,d_l,use_lmi)
        if res_l
            X_l = calculate_mc_input(x_lb_l,y_lb_l,U_lb_l)
            Y_l = calculate_different_vectors(X_l,x_lb_l)
            x_ub_l, y_ub_l, ub_l = mountain_climbing_lse(X_l,C_l,d_l)
            if lb_l < UB
                push!(nodes_list,[ub_l,lb_l,x_lb_l,X_lb_l,C_l,d_l,Y_l])

            end
        end

        if isempty(nodes_list)
            break
        else
            ind = argmin([nodes_list[i][2] for i in 1:length(nodes_list)])
            cur_node = nodes_list[ind]
            deleteat!(nodes_list, ind)
            ub, lb = cur_node[1], cur_node[2]
            x_cur, X_cur = cur_node[3], cur_node[4]
            C, d, Y_split = cur_node[5], cur_node[6], cur_node[7]
            LB = lb
            if ub < UB
                UB = ub
                opt_sol, opt_val = cur_node[3], cur_node[1]
            end
        end
        t0_2 = time_ns()
        total_time = (t0_2-t0_1)*10^(-9)
    end
    return opt_sol, opt_val, gen_hyper
end

#######################     Experiments     ##########################

δ = 1e-4
use_lmi = false


obj_vals_total = []
times_total = []
tree_depths_total = []

# Instance 1
n = 10
C_mat = CSV.read("Data/LSE/LogSumExp_C_10_20.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LSE/LogSumExp_d_10_20.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times, tree_depths = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_lse(C,d,δ,use_lmi)
    t2 = time_ns()
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    push!(tree_depths,depth)
    i += n
    cnt += 1
end
push!(obj_vals_total, -mean(obj_vals))
push!(times_total, mean(times))
push!(tree_depths_total, mean(tree_depths))

# Instance 2
n = 40
C_mat = CSV.read("Data/LSE/LogSumExp_C_40_80.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LSE/LogSumExp_d_40_80.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times, tree_depths = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_lse(C,d,δ,use_lmi)
    t2 = time_ns()
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    push!(tree_depths,depth)
    i += n
    cnt += 1
end
push!(obj_vals_total, -mean(obj_vals))
push!(times_total, mean(times))
push!(tree_depths_total, mean(tree_depths))

# Instance 3
n = 10
C_mat = CSV.read("Data/LSE/LogSumExp_C_10_100.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LSE/LogSumExp_d_10_100.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times, tree_depths = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_lse(C,d,δ,use_lmi)
    t2 = time_ns()
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    push!(tree_depths,depth)
    i += n
    cnt += 1
end
push!(obj_vals_total, -mean(obj_vals))
push!(times_total, mean(times))
push!(tree_depths_total, mean(tree_depths))

# Instance 4
n = 20
C_mat = CSV.read("Data/LSE/LogSumExp_C_20_20.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LSE/LogSumExp_d_20_20.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times, tree_depths = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_lse(C,d,δ,use_lmi)
    t2 = time_ns()
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    push!(tree_depths,depth)
    i += n
    cnt += 1
end
push!(obj_vals_total, -mean(obj_vals))
push!(times_total, mean(times))
push!(tree_depths_total, mean(tree_depths))

# Instance 5
n = 50
C_mat = CSV.read("Data/LSE/LogSumExp_C_50_50.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LSE/LogSumExp_d_50_50.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times, tree_depths = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_lse(C,d,δ,use_lmi)
    t2 = time_ns()
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    push!(tree_depths,depth)
    i += n
    cnt += 1
end
push!(obj_vals_total, -mean(obj_vals))
push!(times_total, mean(times))
push!(tree_depths_total, mean(tree_depths))


results = DataFrame("Instance"=>1:5, "Obj_Val"=>obj_vals_total, "Time"=>times_total, "Depth"=>tree_depths_total)

println(results)
