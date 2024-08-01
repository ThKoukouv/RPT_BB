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

function solve_rpt_relaxation_qcqp(C,d,P,q,r,quad_cons_data,use_lmi)
    L, n = size(C)
    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, X[1:n,1:n]>=0, Symmetric)

    @constraint(m1, C*x .<= d)
    @constraint(m1, [j in 1:n], C*X[:,j] .<= x[j]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')

    quad_cons_data_convex, quad_cons_data_nonconvex = [], []

    for k in quad_cons_data
        P1 = k[1]
        l1, u1 = eigen(P1)
        if sum(l1 .>= 0) == length(l1)
            push!(quad_cons_data_convex,k)
        else
            push!(quad_cons_data_nonconvex,k)
        end
    end

    if length(quad_cons_data_nonconvex) > 0
        for k in quad_cons_data_nonconvex
            P1, q1, r1 = k[1], k[2], k[3]
            @constraint(m1, tr(P1*X) + q1'*x + r1 <= 0)
        end
    end

    if length(quad_cons_data_convex) > 0
        for data_i in quad_cons_data_convex
            Pi, qi, ri = data_i[1], data_i[2], data_i[3]
            for data_j in quad_cons_data_convex
                Pj, qj, rj = data_j[1], data_j[2], data_j[3]
                Q = Pi^(0.5)*X*Pj^(0.5)
                soc_list = [ri*rj+ri*qj'*x+rj*qi'*x+qi'*X*qj,Q[1,1]]
                for i in 1:n
                    for j in 1:n
                        if i != 1 && j != 1
                            push!(soc_list,Q[i,j])
                        end
                    end
                end
                @constraint(m1, soc_list in SecondOrderCone())
            end
            @constraint(m1, [j in 1:L],
                            [-ri*d[j]+ri*C[j,:]'*x-d[j]*qi'*x+qi'*X*C[j,:];
                            d[j]*Pi^(0.5)*x.-Pi^(0.5)*X*C[j,:]] in SecondOrderCone())
            @constraint(m1, [j in 1:n], [-ri*x[j]-qi'*X[:,j];Pi^(0.5)*X[:,j]] in SecondOrderCone())
        end
    end
    if use_lmi
        @constraint(m1, [X  x;  x' 1] in PSDCone())
    end
    @objective(m1, Min, tr(P*X) + q'*x + r)
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.SLOW_PROGRESS
        return true, JuMP.value.(x), JuMP.value.(X), objective_value(m1)
    else
        return false, zeros(n), zeros(n,n), 1e6
    end
end

function qcqp_ub_ipopt(C,d,P,q,r,quad_cons_data,x0)
    L, n = size(C)
    m1 = Model(Ipopt.Optimizer)
    @variable(m1, x[1:n]>=0)
    for j in 1:n
        JuMP.set_start_value(x[j], x0[j])
    end
    @constraint(m1, C*x .<= d)
    for k in quad_cons_data
        P1, q1, r1 = k[1], k[2], k[3]
        @constraint(m1, x'*P1*x + q1'*x + r1 <= 0)
    end
    @objective(m1, Min, x'*P*x + q'*x +r)
    optimize!(m1)
    return objective_value(m1)
end

function is_qcqp_feasible(C,d,quad_cons_data,x)
    n1 = size(C,1)
    n2 = length(quad_cons_data)
    cnt_lin = sum(C*x .<= d)
    cnt_quad = 0
    for k in quad_cons_data
        P1, q1, r1 = k[1], k[2], k[3]
        if x'*P1*x + q1'*x + r1 <= 0.0
            cnt_quad += 1
        end
    end
    res = (cnt_lin + cnt_quad  == n1 + n2)
    return res
end

function get_qcqp_ub(X,C,d,P,q,r,quad_cons_data)
    best_ub = 1e6
    for i in 1:length(X)
        x = X[i]
        if is_qcqp_feasible(C,d,quad_cons_data,x)
            # cur_ub = x'*P*x + q'*x + r
            cur_ub = qcqp_ub_ipopt(C,d,P,q,r,quad_cons_data,x)
            if cur_ub < best_ub
                best_ub = cur_ub
            end
        end
    end
    return best_ub
end

function rpt_bb_qcqp(C_init,d_init,P,q,r,quad_cons_data,δ,use_lmi)
    C, d = C_init, d_init
    gen_hyper = 0

    # Root Node
    res_root, x_lb, X_lb, lb = solve_rpt_relaxation_qcqp(C,d,P,q,r,quad_cons_data,use_lmi)
    X = calculate_candidate_vectors(x_lb,X_lb)
    Y_split = calculate_different_vectors(X,x_lb)
    ub = get_qcqp_ub(X,C,d,P,q,r,quad_cons_data)

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
        res_r, x_lb_r, X_lb_r, lb_r = solve_rpt_relaxation_qcqp(C_r,d_r,P,q,r,quad_cons_data,use_lmi)
        if res_r
            X_r = calculate_candidate_vectors(x_lb_r,X_lb_r)
            Y_r = calculate_different_vectors(X_r,x_lb_r)
            ub_r = get_qcqp_ub(X_r,C_r,d_r,P,q,r,quad_cons_data)
            if lb_r < UB
                push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r,Y_r])
            end
        end
        # Left child
        res_l, x_lb_l, X_lb_l, lb_l = solve_rpt_relaxation_qcqp(C_l,d_l,P,q,r,quad_cons_data,use_lmi)
        if res_l
            X_l = calculate_candidate_vectors(x_lb_l,X_lb_l)
            Y_l = calculate_different_vectors(X_l,x_lb_l)
            ub_l = get_qcqp_ub(X_l,C_l,d_l,P,q,r,quad_cons_data)
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

#####################    Experiments       #######################

δ = 1e-4
use_lmi = true

obj_vals_total = []
times_total = []
tree_depths_total = []


# Problem 1
n, L = 20, 10
C1 = [-3,7,0,-5,1,1,0,2,-1,1]
C2 = [7,0,-5,1,1,0,2,-1,-1,1]
C3 = [0,-5,1,1,0,2,-1,-1,-9,1]
C4 = [-5,1,1,0,2,-1,-1,-9,3,1]
C5 = [1,1,0,2,-1,-1,-9,3,5,1]
C6 = [1,0,2,-1,-1,-9,3,5,0,1]
C7 = [0,2,-1,-1,-9,3,5,0,0,1]
C8 =[2,-1,-1,-9,3,5,0,0,1,1]
C9 = [-1,-1,-9,3,5,0,0,1,7,1]
C10 = [-1,-9,3,5,0,0,1,7,-7,1]
C11 = [-9,3,5,0,0,1,7,-7,-4,1]
C12 = [3,5,0,0,1,7,-7,-4,-6,1]
C13 = [5,0,0,1,7,-7,-4,-6,-3,1]
C14 = [0,0,1,7,-7,-4,-6,-3,7,1]
C15 = [0,1,7,-7,-4,-6,-3,7,0,1]
C16 = [1,7,-7,-4,-6,-3,7,0,-5,1]
C17 = [7,-7,-4,-6,-3,7,0,-5,1,1]
C18 = [-7,-4,-6,-3,7,0,-5,1,1,1]
C19 = [-4,-6,-3,7,0,-5,1,1,0,1]
C20 = [-6,-3,7,0,-5,1,1,0,2,1]
C_1 = hcat(C1,C2,C3,C4,C5,C6,C7,C8,C9,C10)
C_2 = hcat(C11,C12,C13,C14,C15,C16,C17,C18,C19,C20)
C = hcat(C_1,C_2)
d = [-5,2,-1,-3,5,4,-1,0,9,40]
C_init, d_init = C, d
id_mat = zeros(n,n) + I
P = -0.5*id_mat
q = 2*ones(n)
r = -2*n
quad_cons_data = []
t1 = time_ns()
x_opt, obj_opt, depth = rpt_bb_qcqp(C,d,P,q,r,quad_cons_data,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)


# Problem 2
n, L = 20, 10
C1 = [-3,7,0,-5,1,1,0,2,-1,1]
C2 = [7,0,-5,1,1,0,2,-1,-1,1]
C3 = [0,-5,1,1,0,2,-1,-1,-9,1]
C4 = [-5,1,1,0,2,-1,-1,-9,3,1]
C5 = [1,1,0,2,-1,-1,-9,3,5,1]
C6 = [1,0,2,-1,-1,-9,3,5,0,1]
C7 = [0,2,-1,-1,-9,3,5,0,0,1]
C8 =[2,-1,-1,-9,3,5,0,0,1,1]
C9 = [-1,-1,-9,3,5,0,0,1,7,1]
C10 = [-1,-9,3,5,0,0,1,7,-7,1]
C11 = [-9,3,5,0,0,1,7,-7,-4,1]
C12 = [3,5,0,0,1,7,-7,-4,-6,1]
C13 = [5,0,0,1,7,-7,-4,-6,-3,1]
C14 = [0,0,1,7,-7,-4,-6,-3,7,1]
C15 = [0,1,7,-7,-4,-6,-3,7,0,1]
C16 = [1,7,-7,-4,-6,-3,7,0,-5,1]
C17 = [7,-7,-4,-6,-3,7,0,-5,1,1]
C18 = [-7,-4,-6,-3,7,0,-5,1,1,1]
C19 = [-4,-6,-3,7,0,-5,1,1,0,1]
C20 = [-6,-3,7,0,-5,1,1,0,2,1]
C_1 = hcat(C1,C2,C3,C4,C5,C6,C7,C8,C9,C10)
C_2 = hcat(C11,C12,C13,C14,C15,C16,C17,C18,C19,C20)
C = hcat(C_1,C_2)
d = [-5,2,-1,-3,5,4,-1,0,9,40]
C_init, d_init = C, d
id_mat = zeros(n,n) + I
P = -0.5*id_mat
q = -5*ones(n)
r = -12.5*n
quad_cons_data = []
t1 = time_ns()
x_opt, obj_opt, depth = rpt_bb_qcqp(C,d,P,q,r,quad_cons_data,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)


# Problem 3
C = CSV.read("Data/QCQP/QCQP_P3_C.csv", DataFrame, header=false)
d = CSV.read("Data/QCQP/QCQP_P3_d.csv", DataFrame, header=false)
P = CSV.read("Data/QCQP/QCQP_P3_P.csv", DataFrame, header=false)
q = CSV.read("Data/QCQP/QCQP_P3_q.csv", DataFrame, header=false)
C = convert(Matrix, C[:,:])
d = convert(Vector, d[:,1])
P = convert(Matrix, P[:,:])
q = convert(Vector, q[:,1])
r = 0
quad_cons_data = []
t1 = time_ns()
x_opt, obj_opt, depth = rpt_bb_qcqp(C,d,P,q,r,quad_cons_data,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)


# Problem 4
C = CSV.read("Data/QCQP/QCQP_P4_C.csv", DataFrame, header=false)
d = CSV.read("Data/QCQP/QCQP_P4_d.csv", DataFrame, header=false)
P = CSV.read("Data/QCQP/QCQP_P4_P.csv", DataFrame, header=false)
q = CSV.read("Data/QCQP/QCQP_P4_q.csv", DataFrame, header=false)
C = convert(Matrix, C[:,:])
d = convert(Vector, d[:,1])
P = convert(Matrix, P[:,:])
q = convert(Vector, q[:,1])
r = 0
quad_cons_data = []
t1 = time_ns()
x_opt, obj_opt, depth = rpt_bb_qcqp(C,d,P,q,r,quad_cons_data,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)

# Problem 5
C = CSV.read("Data/QCQP/QCQP_P5_C.csv", DataFrame, header=false)
d = CSV.read("Data/QCQP/QCQP_P5_d.csv", DataFrame, header=false)
P = CSV.read("Data/QCQP/QCQP_P5_P.csv", DataFrame, header=false)
q = CSV.read("Data/QCQP/QCQP_P5_q.csv", DataFrame, header=false)
C = convert(Matrix, C[:,:])
d = convert(Vector, d[:,1])
P = convert(Matrix, P[:,:])
q = convert(Vector, q[:,1])
r = 0
quad_cons_data = []
t1 = time_ns()
x_opt, obj_opt, depth = rpt_bb_qcqp(C,d,P,q,r,quad_cons_data,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)

# Problem 6
n, m = 8, 4
C = CSV.read("Data/QCQP/QCQP_P6_C.csv", DataFrame, header=false)
d = CSV.read("Data/QCQP/QCQP_P6_d.csv", DataFrame, header=false)
P_mat = CSV.read("Data/QCQP/QCQP_P6_P.csv", DataFrame, header=false)
q_mat = CSV.read("Data/QCQP/QCQP_P6_q.csv", DataFrame, header=false)
r_mat = CSV.read("Data/QCQP/QCQP_P6_r.csv", DataFrame, header=false)
C = convert(Matrix, C[:,:])
d = convert(Vector, d[:,1])
P_mat = convert(Matrix, P_mat[:,:])
q_mat = convert(Matrix, q_mat[:,:])
r_mat = convert(Vector, r_mat[:,1])
P = P_mat[:,1:n]
q = q_mat[:,1]
r = r_mat[1]
quad_cons_data = []
cnt = 2
for i in 1:m
    P1 = P_mat[:,i*n+1:(i+1)*n]
    q1 = q_mat[:,cnt]
    r1 = r_mat[cnt]
    push!(quad_cons_data,[P1,q1,r1])
    cnt += 1
end
t1 = time_ns()
x_opt, obj_opt, depth = rpt_bb_qcqp(C,d,P,q,r,quad_cons_data,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)


# Problem 7
n, m = 12, 6
C = CSV.read("Data/QCQP/QCQP_P7_C.csv", DataFrame, header=false)
d = CSV.read("Data/QCQP/QCQP_P7_d.csv", DataFrame, header=false)
P_mat = CSV.read("Data/QCQP/QCQP_P7_P.csv", DataFrame, header=false)
q_mat = CSV.read("Data/QCQP/QCQP_P7_q.csv", DataFrame, header=false)
r_mat = CSV.read("Data/QCQP/QCQP_P7_r.csv", DataFrame, header=false)
C = convert(Matrix, C[:,:])
d = convert(Vector, d[:,1])
P_mat = convert(Matrix, P_mat[:,:])
q_mat = convert(Matrix, q_mat[:,:])
r_mat = convert(Vector, r_mat[:,1])
P = P_mat[:,1:n]
q = q_mat[:,1]
r = r_mat[1]
quad_cons_data = []
cnt = 2
for i in 1:m
    P1 = P_mat[:,(i*n+1):(i+1)*n]
    q1 = q_mat[:,cnt]
    r1 = r_mat[cnt]
    push!(quad_cons_data,[P1,q1,r1])
    cnt += 1
end
t1 = time_ns()
x_opt, obj_opt, depth = rpt_bb_qcqp(C,d,P,q,r,quad_cons_data,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)


# Problem 8
n, m = 16, 8
C = CSV.read("Data/QCQP/QCQP_P8_C.csv", DataFrame, header=false)
d = CSV.read("Data/QCQP/QCQP_P8_d.csv", DataFrame, header=false)
P_mat = CSV.read("Data/QCQP/QCQP_P8_P.csv", DataFrame, header=false)
q_mat = CSV.read("Data/QCQP/QCQP_P8_q.csv", DataFrame, header=false)
r_mat = CSV.read("Data/QCQP/QCQP_P8_r.csv", DataFrame, header=false)
C = convert(Matrix, C[:,:])
d = convert(Vector, d[:,1])
P_mat = convert(Matrix, P_mat[:,:])
q_mat = convert(Matrix, q_mat[:,:])
r_mat = convert(Vector, r_mat[:,1])
P = P_mat[:,1:n]
q = q_mat[:,1]
r = r_mat[1]
quad_cons_data = []
cnt = 2
for i in 1:m
    P1 = P_mat[:,(i*n+1):(i+1)*n]
    q1 = q_mat[:,cnt]
    r1 = r_mat[cnt]
    push!(quad_cons_data,[P1,q1,r1])
    cnt += 1
end
t1 = time_ns()
x_opt, obj_opt, depth = rpt_bb_qcqp(C,d,P,q,r,quad_cons_data,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)


# Problem 9
n, m = 30, 15
C = CSV.read("Data/QCQP/QCQP_P9_C.csv", DataFrame, header=false)
d = CSV.read("Data/QCQP/QCQP_P9_d.csv", DataFrame, header=false)
P_mat = CSV.read("Data/QCQP/QCQP_P9_P.csv", DataFrame, header=false)
q_mat = CSV.read("Data/QCQP/QCQP_P9_q.csv", DataFrame, header=false)
r_mat = CSV.read("Data/QCQP/QCQP_P9_r.csv", DataFrame, header=false)
C = convert(Matrix, C[:,:])
d = convert(Vector, d[:,1])
P_mat = convert(Matrix, P_mat[:,:])
q_mat = convert(Matrix, q_mat[:,:])
r_mat = convert(Vector, r_mat[:,1])
P = P_mat[:,1:n]
q = q_mat[:,1]
r = r_mat[1]
quad_cons_data = []
cnt = 2
for i in 1:m
    P1 = P_mat[:,(i*n+1):(i+1)*n]
    q1 = q_mat[:,cnt]
    r1 = r_mat[cnt]
    push!(quad_cons_data,[P1,q1,r1])
    cnt += 1
end
t1 = time_ns()
x_opt, obj_opt, depth = rpt_bb_qcqp(C,d,P,q,r,quad_cons_data,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)


push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)


# Problem 10
n, m = 40, 20
C = CSV.read("Data/QCQP/QCQP_P10_C.csv", DataFrame, header=false)
d = CSV.read("Data/QCQP/QCQP_P10_d.csv", DataFrame, header=false)
P_mat = CSV.read("Data/QCQP/QCQP_P10_P.csv", DataFrame, header=false)
q_mat = CSV.read("Data/QCQP/QCQP_P10_q.csv", DataFrame, header=false)
r_mat = CSV.read("Data/QCQP/QCQP_P10_r.csv", DataFrame, header=false)
C = convert(Matrix, C[:,:])
d = convert(Vector, d[:,1])
P_mat = convert(Matrix, P_mat[:,:])
q_mat = convert(Matrix, q_mat[:,:])
r_mat = convert(Vector, r_mat[:,1])
P = P_mat[:,1:n]
q = q_mat[:,1]
r = r_mat[1]
quad_cons_data = []
cnt = 2
for i in 1:m
    P1 = P_mat[:,(i*n+1):(i+1)*n]
    q1 = q_mat[:,cnt]
    r1 = r_mat[cnt]
    push!(quad_cons_data,[P1,q1,r1])
    cnt += 1
end
t1 = time_ns()
x_opt, obj_opt, depth = rpt_bb_qcqp(C,d,P,q,r,quad_cons_data,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, -obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)

results = DataFrame("Instance"=>1:10, "Obj_Val"=>obj_vals_total, "Times"=>times_total, "Depth"=>tree_depths_total)

println(results)
