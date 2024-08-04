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

##########################    X = X1   ################################

function solve_rpt_relaxation_somol_X1(A,b,C,d,m,K,use_lmi)
    L,n = size(C)
    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n])
    @variable(m1, y[1:m*K])
    @variable(m1, X[1:n,1:n]>=0, Symmetric)
    @variable(m1, Y[1:m*K,1:m*K]>=0, Symmetric)
    @variable(m1, U[1:n,1:m*K]>=0)

    @constraint(m1, [j in 1:n], C*X[:,j] .<= x[j]*d)
    @constraint(m1, [j in 1:m*K], C*U[:,j] .<= y[j]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')
    for j in collect(1:m:m*K)
        @constraint(m1, sum(y[i] for i in j:j+m-1) == 1)
        @constraint(m1, sum(U[:,i] for i in j:j+m-1) .== x)
        @constraint(m1, sum(Y[:,i] for i in j:j+m-1) .== y)
    end

    if use_lmi
        @constraint(m1, [X  U x; U' Y y; x' y' 1] in PSDCone())
    end

    @objective(m1, Min, -tr(U*A') - b'*y)
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.SLOW_PROGRESS
        return true, JuMP.value.(x), JuMP.value.(y), JuMP.value.(X), JuMP.value.(Y), JuMP.value.(U), objective_value(m1)
    else
        return false, zeros(n), zeros(m*K), zeros(n,n), zeros(m*K,m*K), zeros(n,m*K), 1e6
    end
end

function solve_inner_x_somol_X1(y,A,b,C,d)
    n = size(C,2)
    m1 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m1, "OutputFlag", 0)
    @variable(m1, x[1:n]>=0)
    @constraint(m1, C*x .<= d)
    @objective(m1, Min, -x'*(A*y))
    optimize!(m1)
    return JuMP.value.(x)
end

function solve_inner_y_somol(x,A,b,m,K)
    m1 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m1, "OutputFlag", 0)
    @variable(m1, y[1:m*K]>=0)
    for j in collect(1:m:m*K)
        @constraint(m1, sum(y[i] for i in j:j+m-1) == 1)
    end
    @objective(m1, Min, -x'*(A*y) - b'*y)
    optimize!(m1)
    return JuMP.value.(y)
end

function mountain_climbing_somol_X1(X,A,b,C,d,m,K)
    L = []
    for x in X
        y = solve_inner_y_somol(x,A,b,m,K)
        eps = 1
        while abs(eps) > 0.001
            Ub = -x'*A*y - b'*y
            x = solve_inner_x_somol_X1(y,A,b,C,d)
            y = solve_inner_y_somol(x,A,b,m,K)
            Ubx = -x'*(A*y) - b'*y
            eps = Ubx - Ub
        end
        push!(L,[x,y])
    end
    ind_max = argmin([-L[i][1]'*(A*L[i][2]) - b'*L[i][2] for i in 1:length(L)])
    x_mc, y_mc = L[ind_max][1], L[ind_max][2]
    return x_mc, y_mc, -x_mc'*(A*y_mc) - b'*y_mc
end

function rpt_bb_somol_X1(A,b,C_init,d_init,m,K,δ,use_lmi)
    C, d = C_init, d_init
    gen_hyper = 0

    # Root Node
    res_root, x_lb, y_lb, X_lb, Y_lb, U_lb, lb = solve_rpt_relaxation_somol_X1(A,b,C,d,m,K,use_lmi)
    X_root = calculate_mc_input(x_lb,y_lb,U_lb)
    x_ub, y_ub, ub = mountain_climbing_somol_X1(X_root,A,b,C,d,m,K)
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
        res_r, x_lb_r, y_lb_r, X_lb_r, Y_lb_r, U_lb_r, lb_r = solve_rpt_relaxation_somol_X1(A,b,C_r,d_r,m,K,use_lmi)
        if res_r
            X_r = calculate_mc_input(x_lb_r,y_lb_r,U_lb_r)
            Y_r = calculate_different_vectors(X_r,x_lb_r)
            x_ub_r, y_ub_r, ub_r = mountain_climbing_somol_X1(X_r,A,b,C_r,d_r,m,K)
            if lb_r < UB
                push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r,Y_r])
            end
        end
        # Left child
        res_l, x_lb_l, y_lb_l, X_lb_l, Y_lb_l, U_lb_l, lb_l = solve_rpt_relaxation_somol_X1(A,b,C_l,d_l,m,K,use_lmi)
        if res_l
            X_l = calculate_mc_input(x_lb_l,y_lb_l,U_lb_l)
            Y_l = calculate_different_vectors(X_l,x_lb_l)
            x_ub_l, y_ub_l, ub_l = mountain_climbing_somol_X1(X_l,A,b,C_l,d_l,m,K)
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

##########################    X = X2   ################################

function solve_rpt_relaxation_somol_X2(A,b,C,d,m,K,α,use_lmi)
    L,n = size(C)
    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n])
    @variable(m1, y[1:m*K]>=0)
    @variable(m1, z[1:n])
    @variable(m1, X[1:n,1:n]>=0, Symmetric)
    @variable(m1, Y[1:m*K,1:m*K]>=0, Symmetric)
    @variable(m1, Z[1:n,1:n], Symmetric)
    @variable(m1, U[1:n,1:m*K]>=0)
    @variable(m1, V[1:n,1:n])
    @variable(m1, R[1:m*K,1:n])

    @constraint(m1, [i in 1:n], Z[i,i] >= 0)

    @constraint(m1, [j in 1:n], C*X[:,j] .<= x[j]*d)
    @constraint(m1, [j in 1:m*K], C*U[:,j] .<= y[j]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')

    @constraint(m1, sum(z[i] for i in 1:n) <= 1)
    @constraint(m1, sum(V[:,i] for i in 1:n) .<= x)
    @constraint(m1, sum(R[:,i] for i in 1:n) .<= y)

    @constraint(m1, C*x .- C*sum(V[:,i] for i in 1:n) .<= (1-sum(z[i] for i in 1:n))*d)
    @constraint(m1, sum(sum(Z[i,j] for j in 1:n) for i in 1:n) - 2*sum(z[i] for i in 1:n) + 1 >= 0)
    @constraint(m1, [i in 1:n, j in 1:n], [X[i,j] - α*x[j], x[j], V[j,i]] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n, j in 1:L], [d[j]*x[i]-α*d[j]-C[j,:]'*X[:,i]+α*C[j,:]'*x,
                                           d[j]-C[j,:]'*x,
                                           d[j]*z[i] - C[j,:]'*V[:,i]] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n], [x[i] - α - sum(V[i,j] for j in 1:n) + α*sum(z[j] for j in 1:n),
                                 1 - sum(z[j] for j in 1:n),
                                 z[i]-sum(Z[j,i] for j in 1:n)] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n, j in i:n], [x[i]+x[j]-2*α,1,Z[i,j]] in MOI.ExponentialCone())

    for j in collect(1:m:m*K)
        @constraint(m1, sum(y[i] for i in j:j+m-1) == 1)
        @constraint(m1, sum(U[:,i] for i in j:j+m-1) .== x)
        @constraint(m1, sum(Y[:,i] for i in j:j+m-1) .== y)
        @constraint(m1, sum(R[i,:] for i in j:j+m-1) .== z)
    end

    @constraint(m1, [i in 1:n], [x[i] - α, 1, z[i]] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n, j in 1:m*K], [U[i,j] - α*y[j], y[j], R[j,i]] in MOI.ExponentialCone())

    if use_lmi
        @constraint(m1, [X U V x; U' Y R y; V' R' Z z; x' y' z' 1] in PSDCone())
    end

    @objective(m1, Min, -tr(U*A') - b'*y)
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.SLOW_PROGRESS
        return true, JuMP.value.(x), JuMP.value.(y), JuMP.value.(X), JuMP.value.(Y), JuMP.value.(U), objective_value(m1)
    else
        return false, zeros(n), zeros(m*K), zeros(n,n), zeros(m*K,m*K), zeros(n,m*K), 1e6
    end
end

function solve_inner_x_somol_X2(y,A,b,C,d,α)
    n = size(C,2)
    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, z[1:n])
    @constraint(m1, C*x .<= d)
    @constraint(m1, [i in 1:n], [x[i] - α, 1, z[i]] in MOI.ExponentialCone())
    @constraint(m1, sum(z[j] for j in 1:n) <= 1)
    @objective(m1, Min, -x'*(A*y))
    optimize!(m1)
    return JuMP.value.(x)
end

function mountain_climbing_somol_X2(X,A,b,C,d,m,K,α)
    L = []
    for x in X
        y = solve_inner_y_somol(x,A,b,m,K)
        eps = 1
        while abs(eps) > 0.001
            Ub = -x'*A*y - b'*y
            x = solve_inner_x_somol_X2(y,A,b,C,d,α)
            y = solve_inner_y_somol(x,A,b,m,K)
            Ubx = -x'*(A*y) - b'*y
            eps = Ubx - Ub
        end
        push!(L,[x,y])
    end
    ind_min = argmin([-L[i][1]'*(A*L[i][2]) - b'*L[i][2] for i in 1:length(L)])
    x_mc, y_mc = L[ind_min][1], L[ind_min][2]
    return x_mc, y_mc, -x_mc'*(A*y_mc) - b'*y_mc
end

function rpt_bb_somol_X2(A,b,C_init,d_init,α,m,K,δ,use_lmi)
    C, d = C_init, d_init
    gen_hyper = 0

    # Root Node
    res_root, x_lb, y_lb, X_lb, Y_lb, U_lb, lb = solve_rpt_relaxation_somol_X2(A,b,C,d,m,K,α,use_lmi)
    X_root = calculate_mc_input(x_lb,y_lb,U_lb)
    x_ub, y_ub, ub = mountain_climbing_somol_X2(X_root,A,b,C,d,m,K,α)
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
        res_r, x_lb_r, y_lb_r, X_lb_r, Y_lb_r, U_lb_r, lb_r = solve_rpt_relaxation_somol_X2(A,b,C_r,d_r,m,K,α,use_lmi)
        if res_r
            X_r = calculate_mc_input(x_lb_r,y_lb_r,U_lb_r)
            Y_r = calculate_different_vectors(X_r,x_lb_r)
            x_ub_r, y_ub_r, ub_r = mountain_climbing_somol_X2(X_r,A,b,C_r,d_r,m,K,α)
            if lb_r < UB
                push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r,Y_r])
            end
        end
        # Left child
        res_l, x_lb_l, y_lb_l, X_lb_l, Y_lb_l, U_lb_l, lb_l = solve_rpt_relaxation_somol_X2(A,b,C_l,d_l,m,K,α,use_lmi)
        if res_l
            X_l = calculate_mc_input(x_lb_l,y_lb_l,U_lb_l)
            Y_l = calculate_different_vectors(X_l,x_lb_l)
            x_ub_l, y_ub_l, ub_l = mountain_climbing_somol_X2(X_l,A,b,C_l,d_l,m,K,α)
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


######################     Experiments      ########################

δ = 1e-4
use_lmi = false
run_rptbb_x1 = true
run_rptbb_x2 = false

obj_vals_total = []
times_total = []
tree_depths_total = []


# instance 1
n, L, m, K, α, c, M = 5, 5, 5, 1, 3, 6, 100
A = CSV.read("/Users/thodoris/Downloads/SumMaxLinData/RPTproblem1.csv", DataFrame, header=false)
A = convert(Matrix, A[:,:])
b = zeros(size(A)[1])
C = zeros(n,n) + I
d = [n/i for i in 1:n]
if run_rptbb_x1
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_somol_X1(A',b,C,d,m,K,δ,use_lmi)
    t2 = time_ns()
end
if run_rptbb_x2
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_somol_X2(A',b,C,d,α,m,K,δ,use_lmi)
    t2 = time_ns()
end
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
if run_rptbb_x1 || run_rptbb_x2
    push!(tree_depths_total, depth)
end

# instance 1a
A_mat = CSV.read("/Users/thodoris/Downloads/SumMaxLinData_a/RPTproblem1aAA.csv", DataFrame, header=false)
A_mat = convert(Matrix, A_mat[:,:])
times, obj_vals, tree_depths = [], [], []
i = 1
while i < 10*n
    A = A_mat[:,i:i+n-1]
    if run_rptbb_x1
        t1 = time_ns()
        x_opt, obj_opt, depth = rpt_bb_somol_X1(A',b,C,d,m,K,δ,use_lmi)
        t2 = time_ns()
    end
    if run_rptbb_x2
        t1 = time_ns()
        x_opt, obj_opt, depth = rpt_bb_somol_X2(A',b,C,d,α,m,K,δ,use_lmi)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(times,total_time)
    push!(obj_vals,obj_opt)
    push!(tree_depths,depth)
    i += n
end
push!(obj_vals_total, -mean(obj_vals))
push!(times_total, mean(times))
if run_rptbb_x1 || run_rptbb_x2
    push!(tree_depths_total, mean(tree_depths))
end



# instance 2
n, L, m, K, α, c, M = 5, 5, 5, 10, 3, 6, 100
A = CSV.read("/Users/thodoris/Downloads/SumMaxLinData/RPTproblem2.csv", DataFrame, header=false)
A = convert(Matrix, A[:,:])
b = zeros(size(A)[1])
C = zeros(n,n) + I
d = [n/i for i in 1:n]
if run_rptbb_x1
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_somol_X1(A',b,C,d,m,K,δ,use_lmi)
    t2 = time_ns()
end
if run_rptbb_x2
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_somol_X2(A',b,C,d,α,m,K,δ,use_lmi)
    t2 = time_ns()
end
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
if run_rptbb_x1 || run_rptbb_x2
    push!(tree_depths_total, depth)
end

# instance 2a
A_mat = CSV.read("/Users/thodoris/Downloads/SumMaxLinData_a/RPTproblem2aAA.csv", DataFrame, header=false)
A_mat = convert(Matrix, A_mat[:,:])
b = zeros(size(A_mat)[1])
C = zeros(n,n) + I
d = [n/i for i in 1:n]
times, obj_vals, tree_depths = [], [], []
i = 1
while i < 10*n
    A = A_mat[:,i:i+n-1]
    if run_rptbb_x1
        t1 = time_ns()
        x_opt, obj_opt, depth = rpt_bb_somol_X1(A',b,C,d,m,K,δ,use_lmi)
        t2 = time_ns()
    end
    if run_rptbb_x2
        t1 = time_ns()
        x_opt, obj_opt, depth = rpt_bb_somol_X2(A',b,C,d,α,m,K,δ,use_lmi)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(times,total_time)
    push!(obj_vals,obj_opt)
    push!(tree_depths,depth)
    i += n
end
push!(obj_vals_total, -mean(obj_vals))
push!(times_total, mean(times))
if run_rptbb_x1 || run_rptbb_x2
    push!(tree_depths_total, mean(tree_depths))
end




# instance 3
n, L, m, K, α, c, M = 20, 5, 10, 10, 11, 25, 1000
A = CSV.read("/Users/thodoris/Downloads/SumMaxLinData/RPTproblem3.csv", DataFrame, header=false)
A = convert(Matrix, A[:,:])
b = zeros(size(A)[1])
C = zeros(n,n) + I
d = [n/i for i in 1:n]
if run_rptbb_x1
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_somol_X1(A',b,C,d,m,K,δ,use_lmi)
    t2 = time_ns()
end
if run_rptbb_x2
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_somol_X2(A',b,C,d,α,m,K,δ,use_lmi)
    t2 = time_ns()
end
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
if run_rptbb_x1 || run_rptbb_x2
    push!(tree_depths_total, depth)
end


A_mat = CSV.read("/Users/thodoris/Downloads/SumMaxLinData_a/RPTproblem3aAA.csv", DataFrame, header=false)
A_mat = convert(Matrix, A_mat[:,:])
times, obj_vals, tree_depths = [], [], []
i = 1
while i < 10*n
    A = A_mat[:,i:i+n-1]
    if run_rptbb_x1
        t1 = time_ns()
        x_opt, obj_opt, depth = rpt_bb_somol_X1(A',b,C,d,m,K,δ,use_lmi)
        t2 = time_ns()
    end
    if run_rptbb_x2
        t1 = time_ns()
        x_opt, obj_opt, depth = rpt_bb_somol_X2(A',b,C,d,α,m,K,δ,use_lmi)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(times,total_time)
    push!(obj_vals,obj_opt)
    push!(tree_depths,depth)
    i += n
end
push!(obj_vals_total, -mean(obj_vals))
push!(times_total, mean(times))
if run_rptbb_x1 || run_rptbb_x2
    push!(tree_depths_total, mean(tree_depths))
end

# instance 4
n, L, m, K, α, c, M = 10, 10, 5, 2, 3, 7, 1000
A = CSV.read("/Users/thodoris/Downloads/SumMaxLinData/RPTproblem7A.csv", DataFrame, header=false)
b = CSV.read("/Users/thodoris/Downloads/SumMaxLinData/RPTproblem7b.csv", DataFrame, header=false)
C = CSV.read("/Users/thodoris/Downloads/SumMaxLinData/RPTproblem7matrixD.csv", DataFrame, header=false)
d = CSV.read("/Users/thodoris/Downloads/SumMaxLinData/RPTproblem7D.csv", DataFrame, header=false)
A = convert(Matrix, A[:,:])
b = [b[i,1] for i in 1:10]
C = convert(Matrix, C[:,:])
d = [d[i,1] for i in 1:10]
if run_rptbb_x1
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_somol_X1(A',b,C,d,m,K,δ,use_lmi)
    t2 = time_ns()
end
if run_rptbb_x2
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_somol_X2(A',b,C,d,α,m,K,δ,use_lmi)
    t2 = time_ns()
end
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
if run_rptbb_x1 || run_rptbb_x2
    push!(tree_depths_total, depth)
end


# instance 4a
A_mat = CSV.read("/Users/thodoris/Downloads/SumMaxLinData_a/RPTproblem7aAA.csv", DataFrame, header=false)
b_mat = CSV.read("/Users/thodoris/Downloads/SumMaxLinData_a/RPTproblem7abb.csv", DataFrame, header=false)
C_mat = CSV.read("/Users/thodoris/Downloads/SumMaxLinData_a/RPTproblem7aDD.csv", DataFrame, header=false)
d_mat = CSV.read("/Users/thodoris/Downloads/SumMaxLinData_a/RPTproblem7addvector.csv", DataFrame, header=false)
A_mat = convert(Matrix, A_mat[:,:])
b_mat = convert(Matrix, b_mat[:,:])
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
times, obj_vals, tree_depths = [], [], []
i, cnt = 1, 1
while i < 10*n
    A = A_mat[:,i:i+n-1]
    b = b_mat[:,cnt]
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    if run_rptbb_x1
        t1 = time_ns()
        x_opt, obj_opt, depth = rpt_bb_somol_X1(A',b,C,d,m,K,δ,use_lmi)
        t2 = time_ns()
    end
    if run_rptbb_x2
        t1 = time_ns()
        x_opt, obj_opt, depth = rpt_bb_somol_X2(A',b,C,d,α,m,K,δ,use_lmi)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(times,total_time)
    push!(obj_vals,obj_opt)
    push!(tree_depths,depth)
    i += n
    cnt += 1
end
push!(obj_vals_total, -mean(obj_vals))
push!(times_total, mean(times))
if run_rptbb_x1 || run_rptbb_x2
    push!(tree_depths_total, mean(tree_depths))
end


# instance 5
n, L, m, K, α, c, M = 20, 10, 10, 10, 5, 30, 1000
C1 = [-3,7,0,-5,1,1,0,2,-1,1]
C2 = [7,0,-5,1,1,0,2,-1,-1,1]
C3 = [0,-5,1,1,0,2,-1,-1,-9,1]
C4 = [-5,1,1,0,2,-1,-1,-9,3,1]
C5 = [1,1,0,2,-1,-1,-9,3,5,1]
C6 = [1,0,2,-1,-1,-9,3,5,0,1]
C7 = [0,2,-1,-1,-9,3,5,0,0,1]
C8 = [2,-1,-1,-9,3,5,0,0,1,1]
C9 = [-1,-1,-9,3,5,0,0,1,7,1]
C10 = [-1,-9,3,5,0,0,1,7,-7,1]
C11 = [-9,3,5,0,0,1,7,-7,4,1]
C12 = [3,5,0,0,1,7,-7,-4,-6,1]
C13 = [5,0,0,1,7,-7,-4,-6,-3,1]
C14 = [0,0,1,7,-7,-4,-6,-3,7,1]
C15 = [0,1,7,-7,-4,-6,-3,7,0,1]
C16 = [1,7,-7,-4,-6,-3,7,0,-5,1]
C17 = [7,-7,-4,-6,-3,7,0,-5,1,1]
C18 = [-7,-4,-6,-3,7,0,-5,1,1,1]
C19 = [-4,-6,-3,7,0,-5,1,1,0,1]
C20 = [-6,-3,7,0,-5,1,1,0,2,1]
C = hcat(C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20)
d = [-5,2,-1,-3,5,4,-1,0,9,40]
A = CSV.read("/Users/thodoris/Downloads/SumMaxLinData/RPTproblem11A.csv", DataFrame, header=false)
b = CSV.read("/Users/thodoris/Downloads/SumMaxLinData/RPTproblem11b.csv", DataFrame, header=false)
A = convert(Matrix, A[:,:])
b = [b[i,1] for i in 1:size(b)[1]]
if run_rptbb_x1
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_somol_X1(A',b,C,d,m,K,δ,use_lmi)
    t2 = time_ns()
end
if run_rptbb_x2
    t1 = time_ns()
    x_opt, obj_opt, depth = rpt_bb_somol_X2(A',b,C,d,α,m,K,δ,use_lmi)
    t2 = time_ns()
end
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
if run_rptbb_x1 || run_rptbb_x2
    push!(tree_depths_total, depth)
end

# instance 5a
A_mat = CSV.read("/Users/thodoris/Downloads/SumMaxLinData_a/RPTproblem11aAA.csv", DataFrame, header=false)
b_mat = CSV.read("/Users/thodoris/Downloads/SumMaxLinData_a/RPTproblem11abb.csv", DataFrame, header=false)
A_mat = convert(Matrix, A_mat[:,:])
b_mat = convert(Matrix, b_mat[:,:])
times, obj_vals, tree_depths = [], [], []
i, cnt = 1, 1
while i < 10*n
    A = A_mat[:,i:i+n-1]
    b = b_mat[:,cnt]
    if run_rptbb_x1
        t1 = time_ns()
        x_opt, obj_opt, depth = rpt_bb_somol_X1(A',b,C,d,m,K,δ,use_lmi)
        t2 = time_ns()
    end
    if run_rptbb_x2
        t1 = time_ns()
        x_opt, obj_opt, depth = rpt_bb_somol_X2(A',b,C,d,α,m,K,δ,use_lmi)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(times,total_time)
    push!(obj_vals,obj_opt)
    push!(tree_depths,depth)
    i += n
    cnt += 1
end
push!(obj_vals_total, -mean(obj_vals))
push!(times_total, mean(times))
if run_rptbb_x1 || run_rptbb_x2
    push!(tree_depths_total, mean(tree_depths))
end

instances_sml = ["1", "1a", "2", "2a", "3", "3a", "4", "4a", "5", "5a"]

if run_rptbb_x1 || run_rptbb_x2
    results = DataFrame("Instance"=>instances_sml, "Obj_Val"=>obj_vals_total, "Time"=>times_total, "Depth"=>tree_depths_total)
else
    results = DataFrame("Instance"=>instances_sml, "Obj_Val"=>obj_vals_total, "Time"=>times_total)
end
println(results)
