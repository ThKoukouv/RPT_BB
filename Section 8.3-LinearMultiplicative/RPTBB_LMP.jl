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

function calculate_candidate_vectors(x,X)
    Y = [x]
    for j in 1:size(x)[1]
        if x[j] == 0
            push!(Y,x)
        else
            push!(Y,X[:,j]/x[j])
        end
    end
    return Y
end

function solve_rpt_relaxation_lmp(C,d,A,b,use_lmi)
    L,n = size(C)
    p = size(A,1)
    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, y[1:p]>=0)
    @variable(m1, w[1:p])
    @variable(m1, X[1:n,1:n]>=0, Symmetric)
    @variable(m1, Y[1:p,1:p]>=0, Symmetric)
    @variable(m1, W[1:p,1:p], Symmetric)
    @variable(m1, U[1:n,1:p]>=0)
    @variable(m1, Q[1:n,1:p])
    @variable(m1, P[1:p,1:p])

    @constraint(m1, C*x .<= d)
    @constraint(m1, [j in 1:n], C*X[:,j] .<= x[j]*d)
    @constraint(m1, [j in 1:p], C*U[:,j] .<= y[j]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')

    @constraint(m1, [i in 1:p], [-w[i]-1, 1, y[i]] in MOI.ExponentialCone())

    @constraint(m1, [i in 1:p, j in 1:n], [-Q[j,i]-x[j], x[j], U[j,i]] in MOI.ExponentialCone())

    @constraint(m1, [i in 1:p, j in 1:p], [-P[j,i]-y[j], y[j], Y[i,j]] in MOI.ExponentialCone())

    @constraint(m1, [i in 1:p, j in 1:L], [C[j,:]'*x-d[j]-w[i]*d[j]+C[j,:]'*Q[:,i],
                                           d[j] - C[j,:]'*x,
                                           y[i]*d[j] - C[j,:]'*U[:,i]] in MOI.ExponentialCone())

    @constraint(m1, [i in 1:p, j in i:p], [-w[i]-w[j]-2, 1, Y[i,j]] in MOI.ExponentialCone())
    if use_lmi
        @constraint(m1, [X U Q x; U' Y P y; Q' P' W w; x' y' w' 1] in PSDCone())
    end
    @objective(m1, Min, tr(U*A) + b'*y + sum(w[i] for i in 1:p))
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.SLOW_PROGRESS
        return true, JuMP.value.(x), JuMP.value.(y), JuMP.value.(X), JuMP.value.(Y), JuMP.value.(U), objective_value(m1)
    else
        return false, zeros(n), zeros(p), zeros(n,n), zeros(p,p), zeros(n,p), 1e6
    end
end

function solve_inner_x_lmp(y,C,d,A,b)
    L, n = size(C)
    m1 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m1, "OutputFlag", 0)
    @variable(m1, x[1:n]>=0)
    @constraint(m1, C*x .<= d)
    @objective(m1, Min, x'*(A'*y))
    optimize!(m1)
    return JuMP.value.(x)
end

function solve_inner_y_lmp(x,A,b)
    p = size(A,1)
    m1 = Model(Mosek.Optimizer)
    @variable(m1, y[1:p]>=0)
    @variable(m1, w[1:p])
    @constraint(m1, [i in 1:p], [-w[i]-1, 1, y[i]] in MOI.ExponentialCone())
    @objective(m1, Min, (A*x.+b)'*y + sum(w[i] for i in 1:p))
    optimize!(m1)
    return JuMP.value.(y), JuMP.value.(w)
end

function mountain_climbing_lmp(X,C,d,A,b)
    L = []
    for x in X
        y, w = solve_inner_y_lmp(x,A,b)
        eps = 1
        while abs(eps) > 0.001
            Ub = (A*x.+b)'*y + sum(w[i] for i in 1:p)
            x = solve_inner_x_lmp(y,C,d,A,b)
            y, w = solve_inner_y_lmp(x,A,b)
            Ubx = (A*x.+b)'*y + sum(w[i] for i in 1:p)
            eps = Ubx - Ub
        end
        push!(L,[x,y])
    end
    ind_min = argmin([(A*L[i][1].+b)'*L[i][2] + sum(-log(L[i][2][j])-1 for j in 1:p) for i in 1:length(L)])
    x_mc, y_mc = L[ind_min][1], L[ind_min][2]
    return x_mc, y_mc, (A*x_mc.+b)'*y_mc + sum(-log(y_mc[i])-1 for i in 1:p)
end


function rpt_bb_lmp(A,b,C_init,d_init,δ,use_lmi)
    C, d = C_init, d_init
    gen_hyper = 0

    # Root Node
    res_root, x_lb, y_lb, X_lb, Y_lb, U_lb, lb = solve_rpt_relaxation_lmp(C,d,A,b,use_lmi)
    X_root = calculate_mc_input(x_lb,y_lb,U_lb)
    x_ub, y_ub, ub = mountain_climbing_lmp(X_root,C,d,A,b)
    Y_split = calculate_different_vectors(X_root,x_lb)

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
        res_r, x_lb_r, y_lb_r, X_lb_r, Y_lb_r, U_lb_r, lb_r = solve_rpt_relaxation_lmp(C_r,d_r,A,b,use_lmi)
        if res_r
            X_r = calculate_mc_input(x_lb_r,y_lb_r,U_lb_r)
            Y_r = calculate_different_vectors(X_r,x_lb_r)
            x_ub_r, y_ub_r, ub_r = mountain_climbing_lmp(X_r,C_r,d_r,A,b)
            if lb_r < UB
                push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r,Y_r])
            end
        end
        # Left child
        res_l, x_lb_l, y_lb_l, X_lb_l, Y_lb_l, U_lb_l, lb_l = solve_rpt_relaxation_lmp(C_l,d_l,A,b,use_lmi)
        if res_l
            X_l = calculate_mc_input(x_lb_l,y_lb_l,U_lb_l)
            Y_l = calculate_different_vectors(X_l,x_lb_l)
            x_ub_l, y_ub_l, ub_l = mountain_climbing_lmp(X_l,C_l,d_l,A,b)
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


##########################   Experiments    ###############################

δ = 1e-4
use_lmi = true

obj_vals_total = []
times_total = []
tree_depths_total = []


# Instance 1
n, L, p = 5, 5, 5
C_mat = CSV.read("Data/LMP/LMP_C_5_5_5.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LMP/LMP_d_5_5_5.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times, tree_depths = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    A = -C[L+1:L+p,:]
    b = d[L+1:L+p]
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_lmp(A,b,C,d,δ,use_lmi)
    t2 = time_ns()
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    push!(tree_depths,depth)
    i += n
    cnt += 1
end
push!(obj_vals_total, mean(obj_vals))
push!(times_total, mean(times))
push!(tree_depths_total, mean(tree_depths))



# Instance 2
n, L, p = 7, 7, 10
C_mat = CSV.read("Data/LMP/LMP_C_7_7_10.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LMP/LMP_d_7_7_10.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times, tree_depths = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    A = -C[L+1:L+p,:]
    b = d[L+1:L+p]
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_lmp(A,b,C,d,δ,use_lmi)
    t2 = time_ns()
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    push!(tree_depths,depth)
    i += n
    cnt += 1
end
push!(obj_vals_total, mean(obj_vals))
push!(times_total, mean(times))
push!(tree_depths_total, mean(tree_depths))

# Instance 3
n, L, p = 10, 10, 9
C_mat = CSV.read("Data/LMP/LMP_C_10_10_9.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LMP/LMP_d_10_10_9.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times, tree_depths = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    A = -C[L+1:L+p,:]
    b = d[L+1:L+p]
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_lmp(A,b,C,d,δ,use_lmi)
    t2 = time_ns()
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    push!(tree_depths,depth)
    i += n
    cnt += 1
end
push!(obj_vals_total, mean(obj_vals))
push!(times_total, mean(times))
push!(tree_depths_total, mean(tree_depths))

# Instance 4
n, L, p = 20, 20, 8
C_mat = CSV.read("Data/LMP/LMP_C_20_20_8.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LMP/LMP_d_20_20_8.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times, tree_depths = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    A = -C[L+1:L+p,:]
    b = d[L+1:L+p]
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_lmp(A,b,C,d,δ,use_lmi)
    t2 = time_ns()
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    push!(tree_depths,depth)
    i += n
    cnt += 1
end
push!(obj_vals_total, mean(obj_vals))
push!(times_total, mean(times))
push!(tree_depths_total, mean(tree_depths))


# Instance 5
n, L, p = 40, 40, 4
C_mat = CSV.read("Data/LMP/LMP_C_40_40_4.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LMP/LMP_d_40_40_4.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times, tree_depths = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    A = -C[L+1:L+p,:]
    b = d[L+1:L+p]
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_lmp(A,b,C,d,δ,use_lmi)
    t2 = time_ns()
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    push!(tree_depths,depth)
    i += n
    cnt += 1
end
push!(obj_vals_total, mean(obj_vals))
push!(times_total, mean(times))
push!(tree_depths_total, mean(tree_depths))


results = DataFrame("Instance"=>1:5, "Obj_Val"=>obj_vals_total, "Time"=>times_total, "Depth"=>tree_depths_total)

println(results)
