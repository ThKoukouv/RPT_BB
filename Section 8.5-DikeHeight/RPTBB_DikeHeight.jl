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

function calculate_different_vectors(X,x)
    Y = x
    for i in 1:length(X)
        if norm(X[i] .- x) > 1e-3
            Y = hcat(Y,X[i])
        end
    end
    return Y
end

function calculate_candidate_vectors_dh(x,v,X,S)
    Y = [x]
    for j in 1:size(X)[2]
        if x[j] == 0
            push!(Y,x)
        else
            push!(Y,X[:,j]/x[j])
        end
    end
    for j in 1:size(S)[2]
        if v[j] == 0
            push!(Y,x)
        else
            push!(Y,S[:,j]/v[j])
        end
    end
    return Y
end

function solve_rpt_relaxation_dike_height(t,α,α1,δ,λ,θ,β,c,b,T,C_cons,d_cons,use_lmi)
    K = size(t)[1] - 2
    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:K+1]>=0)
    @variable(m1, u[1:K+1])
    @variable(m1, v[1:K+1])
    @variable(m1, w)
    @variable(m1, X[1:K+1,1:K+1]>=0, Symmetric)
    @variable(m1, V[1:K+1,1:K+1]>=0, Symmetric)
    @variable(m1, W>=0)
    @variable(m1, S[1:K+1,1:K+1])
    @variable(m1, z[1:K+1])
    @variable(m1, q[1:K+1])

    # epigraphical variables for objective terms
    @constraint(m1, [k in 1:K+1], [λ*c*sum(x[i] for i in 1:k)-δ*c*t[k]-δ*b*t[k]*x[k]+λ*b*sum(X[i,k] for i in 1:k),
                                        c + b*x[k],
                                        u[k]] in MOI.ExponentialCone())
    @constraint(m1, [k in 1:K+1], [-θ*sum(x[i] for i in 1:k),1,v[k]] in MOI.ExponentialCone())
    @constraint(m1, [β*T-θ*sum(x[i] for i in 1:K+1),1,w] in MOI.ExponentialCone())

    # multiply with each other and themselves
    @constraint(m1, [k in 1:K+1], [β*T-θ*sum(x[i] for i in 1:k)-θ*sum(x[i] for i in 1:K+1),1,q[k]] in MOI.ExponentialCone())
    @constraint(m1, [k in 1:K+1, l in 1:K+1], [-θ*sum(x[i] for i in 1:k) - θ*sum(x[i] for i in 1:l),
                                                1,
                                                V[k,l]] in MOI.ExponentialCone())
    @constraint(m1, [2*β*T-2*θ*sum(x[i] for i in 1:K+1),1,W] in MOI.ExponentialCone())

    # multiply with xj
    @constraint(m1, [k in 1:K+1, j in 1:K+1], [-θ*sum(X[i,j] for i in 1:k),x[j],S[j,k]] in MOI.ExponentialCone())
    @constraint(m1, [j in 1:K+1], [β*T*x[j]-θ*sum(X[i,j] for i in 1:K+1),x[j],z[j]] in MOI.ExponentialCone())

    if length(C_cons) > 0
        @constraint(m1, C_cons*x .<= d_cons)
        @constraint(m1, [j in 1:K+1], C_cons*X[:,j] .<= x[j]*d_cons)
        @constraint(m1, d_cons*x'*C_cons' .+ C_cons*x*d_cons' .<= C_cons*X*C_cons' .+ d_cons*d_cons')
        @constraint(m1, [j in 1:size(C_cons)[1], k in 1:K+1],
                                [-θ*d_cons[j]*sum(x[i] for i in 1:k) + θ*sum(C_cons[j,:]'*X[:,i] for i in 1:k),
                                d_cons[j]-C_cons[j,:]'*x,
                                d_cons[j]*v[k] - C_cons[j,:]'*S[:,k]] in MOI.ExponentialCone())
        @constraint(m1, [j in 1:size(C_cons)[1]],
                                [β*T*d_cons[j]-β*T*C_cons[j,:]'*x-
                                θ*d_cons[j]*sum(x[i] for i in 1:K+1)+
                                θ*sum(C_cons[j,:]'*X[:,i] for i in 1:K+1),
                                d_cons[j]-C_cons[j,:]'*x,
                                d_cons[j]*w-C_cons[j,:]'*z] in MOI.ExponentialCone())
    end

    if use_lmi
        @constraint(m1, [X S z x; S' V q v; z' q' W w; x' v' w 1] in PSDCone())
    end

    @objective(m1, Min, sum(u[k] + α[k]*v[k] for k in 1:K+1) + α1*w)
    optimize!(m1)

    if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.SLOW_PROGRESS
        return true, JuMP.value.(x), JuMP.value.(v), JuMP.value.(X), JuMP.value.(S), objective_value(m1)
    else
        return false, zeros(K+1), zeros(K+1), zeros(K+1,K+1), zeros(K+1,K+1), -1e6
    end
end

function dike_height_obj(x,t,α,α1,δ_dh,λ,θ,β,c,b,T)
    K = size(t,1) - 2
    obj1 = sum((c+b*x[k])*exp(λ*sum(x[i] for i in 1:k) - δ_dh*t[k]) for k in 1:K+1)
    obj2 = sum(α[k]*exp(-θ*sum(x[i] for i in 1:k)) for k in 1:K+1)
    obj3 = α1*exp(β*T - θ*sum(x[i] for i in 1:K+1))
    return obj1 + obj2 + obj3
end

function dike_height_ub_ipopt(t,α,α1,δ_dh,λ,θ,β,c,b,T,C,d,x0)
    K = size(t,1) - 2
    m1 = Model(Ipopt.Optimizer)
    @variable(m1, x[1:K+1]>=0)
    for j in 1:(K+1)
        JuMP.set_start_value(x[j], x0[j])
    end
    if length(C) > 0
        @constraint(m1, C*x .<= d)
    end
    @NLobjective(m1, Min, sum((c+b*x[k])*exp(λ*sum(x[i] for i in 1:k) - δ_dh*t[k]) for k in 1:K+1) +
                        sum(α[k]*exp(-θ*sum(x[i] for i in 1:k)) for k in 1:K+1) +
                        α1*exp(β*T - θ*sum(x[i] for i in 1:K+1)))
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.LOCALLY_SOLVED
        return objective_value(m1)
    else
        return 1e6
    end
end

function get_dike_height_ub(X,t,α,α1,δ,λ,θ,β,c,b,T)
    best_ub = 1e6
    for i in 1:length(X)
        x = X[i]
        # cur_ub = dike_height_obj(x,t,α,α1,δ,λ,θ,β,c,b,T)
        cur_ub = dike_height_ub_ipopt(t,α,α1,δ_dh,λ,θ,β,c,b,T,C,d,x)
        if cur_ub < best_ub
            best_ub = cur_ub
        end
    end
    return best_ub
end

function rpt_bb_dike_height(t,α,α1,δ_dh,λ,θ,β,c,b,T,C_init,d_init,δ,use_lmi)
    C, d = C_init, d_init
    gen_hyper = 0

    # Root Node
    res_root, x_lb, v_lb, X_lb, S_lb, lb = solve_rpt_relaxation_dike_height(t,α,α1,δ_dh,λ,θ,β,c,b,T,C,d,use_lmi)
    X = calculate_candidate_vectors_dh(x_lb,v_lb,X_lb,S_lb)
    Y_split = calculate_different_vectors(X,x_lb)
    ub = get_dike_height_ub(X,t,α,α1,δ_dh,λ,θ,β,c,b,T)

    x_cur, X_cur = x_lb, X_lb

    LB, UB, opt_sol, opt_val = lb, ub, x_lb, lb

    nodes_list = []

    t0_1 = time_ns()
    total_time = 0.0

    while UB - LB > δ && total_time < 3600
        f_opt, l_opt = generate_hyperplane_eigen(x_cur,X_cur)

        f_r, l_r, f_l, l_l  = f_opt, l_opt, -f_opt, -l_opt
        if length(C) == 0
            C_r, d_r, C_l, d_l = f_r', [l_r], f_l', [l_l]
        else
            C_r, d_r, C_l, d_l = vcat(C,f_r'), vcat(d,l_r), vcat(C,f_l'), vcat(d,l_l)
        end
        gen_hyper += 1

        # Right child
        res_r, x_lb_r, v_lb_r, X_lb_r, S_lb_r, lb_r = solve_rpt_relaxation_dike_height(t,α,α1,δ_dh,λ,θ,β,c,b,T,C_r,d_r,use_lmi)
        if res_r
            X_r = calculate_candidate_vectors_dh(x_lb_r,v_lb_r,X_lb_r,S_lb_r)
            Y_r = calculate_different_vectors(X_r,x_lb_r)
            ub_r = get_dike_height_ub(X_r,t,α,α1,δ_dh,λ,θ,β,c,b,T)
            if lb_r < UB
                push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r,Y_r])
            end
        end

        # Left child
        res_l, x_lb_l, v_lb_l, X_lb_l, S_lb_l, lb_l = solve_rpt_relaxation_dike_height(t,α,α1,δ_dh,λ,θ,β,c,b,T,C_l,d_l,use_lmi)
        if res_l
            X_l = calculate_candidate_vectors_dh(x_lb_l,v_lb_l,X_lb_l,S_lb_l)
            Y_l = calculate_different_vectors(X_l,x_lb_l)
            ub_l = get_dike_height_ub(X_l,t,α,α1,δ_dh,λ,θ,β,c,b,T)
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


######################   Experiments   #############################

δ = 1e-4
use_lmi = true

obj_vals_total = []
times_total = []
tree_depths_total = []

# parameters for 10 dike rings
α = 0.033027
c = 16.6939
b = 0.6258
λ = 0.0014
ζ = 0.003774
η = 0.32
S0 = 1564.9/2270
γ = 0.02
δ_dh = 0.04
T = 300
θ = α - ζ
β1 = α*η + γ
β = β1 - δ_dh

#25
t = [0,25,50,75,100,125,150,175,200,225,250,275,T]
K = size(t)[1] - 2
α = [(S0/β)*(exp(β*t[k+1]) - exp(β*t[k])) for k in 1:K+1]
α1 = S0/δ_dh
K = size(t)[1] - 2
α = [(S0/β)*(exp(β*t[k+1]) - exp(β*t[k])) for k in 1:K+1]
α1 = S0/δ_dh
C, d = zeros(K+1,K+1) + I, 300*ones(K+1)
t1 = time_ns()
x_opt, obj_opt, depth  = rpt_bb_dike_height(t,α,α1,δ_dh,λ,θ,β,c,b,T,C,d,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)

#50
t = [0,50,100,150,200,250,T]
K = size(t)[1] - 2
α = [(S0/β)*(exp(β*t[k+1]) - exp(β*t[k])) for k in 1:K+1]
α1 = S0/δ_dh
C, d = zeros(K+1,K+1) + I, 300*ones(K+1)
t1 = time_ns()
x_opt, obj_opt, depth  = rpt_bb_dike_height(t,α,α1,δ_dh,λ,θ,β,c,b,T,C,d,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)

# irregular
t = [0,20,50,90,130,155,180,210,255,270,T]
K = size(t)[1] - 2
α = [(S0/β)*(exp(β*t[k+1]) - exp(β*t[k])) for k in 1:K+1]
α1 = S0/δ_dh
C, d = zeros(K+1,K+1) + I, 300*ones(K+1)
t1 = time_ns()
x_opt, obj_opt, depth  = rpt_bb_dike_height(t,α,α1,δ_dh,λ,θ,β,c,b,T,C,d,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)



# parameters for 15 dike rings
α = 0.0502
c = 125.6422
b = 1.1268
λ = 0.0098
ζ = 0.003764
η = 0.76
S0 = 11810.4/729
γ = 0.02
δ_dh = 0.04
T = 300
θ = α - ζ
β1 = α*η + γ
β = β1 - δ_dh

#25
t = [0,25,50,75,100,125,150,175,200,225,250,275,T]
K = size(t)[1] - 2
α = [(S0/β)*(exp(β*t[k+1]) - exp(β*t[k])) for k in 1:K+1]
α1 = S0/δ_dh
K = size(t)[1] - 2
α = [(S0/β)*(exp(β*t[k+1]) - exp(β*t[k])) for k in 1:K+1]
α1 = S0/δ_dh
C, d = zeros(K+1,K+1) + I, 300*ones(K+1)
t1 = time_ns()
x_opt, obj_opt, depth  = rpt_bb_dike_height(t,α,α1,δ_dh,λ,θ,β,c,b,T,C,d,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)

#50
t = [0,50,100,150,200,250,T]
K = size(t)[1] - 2
α = [(S0/β)*(exp(β*t[k+1]) - exp(β*t[k])) for k in 1:K+1]
α1 = S0/δ_dh
C, d = zeros(K+1,K+1) + I, 300*ones(K+1)
t1 = time_ns()
x_opt, obj_opt, depth  = rpt_bb_dike_height(t,α,α1,δ_dh,λ,θ,β,c,b,T,C,d,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)

# irregular
t = [0,20,50,90,130,155,180,210,255,270,T]
K = size(t)[1] - 2
α = [(S0/β)*(exp(β*t[k+1]) - exp(β*t[k])) for k in 1:K+1]
α1 = S0/δ_dh
C, d = zeros(K+1,K+1) + I, 300*ones(K+1)
t1 = time_ns()
x_opt, obj_opt, depth  = rpt_bb_dike_height(t,α,α1,δ_dh,λ,θ,β,c,b,T,C,d,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)



# parameters dikering 16
α = 0.0574
c = 324.6287
b = 2.1304
λ = 0.01
ζ = 0.002032
η = 0.76
S0 = 22656.5/906
γ = 0.02
δ_dh = 0.04
T = 300
θ = α - ζ
H0 = 0;
delta2 = 0.04
delta1 = 0.04
β1 = α*η + γ
β = β1 - δ_dh


#25
t = [0,25,50,75,100,125,150,175,200,225,250,275,T]
K = size(t)[1] - 2
α = [(S0/β)*(exp(β*t[k+1]) - exp(β*t[k])) for k in 1:K+1]
α1 = S0/δ_dh
K = size(t)[1] - 2
α = [(S0/β)*(exp(β*t[k+1]) - exp(β*t[k])) for k in 1:K+1]
α1 = S0/δ_dh
C, d = zeros(K+1,K+1) + I, 300*ones(K+1)
t1 = time_ns()
x_opt, obj_opt, depth  = rpt_bb_dike_height(t,α,α1,δ_dh,λ,θ,β,c,b,T,C,d,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)

#50
t = [0,50,100,150,200,250,T]
K = size(t)[1] - 2
α = [(S0/β)*(exp(β*t[k+1]) - exp(β*t[k])) for k in 1:K+1]
α1 = S0/δ_dh
C, d = zeros(K+1,K+1) + I, 300*ones(K+1)
t1 = time_ns()
x_opt, obj_opt, depth  = rpt_bb_dike_height(t,α,α1,δ_dh,λ,θ,β,c,b,T,C,d,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)

# irregular
t = [0,20,50,90,130,155,180,210,255,270,T]
K = size(t)[1] - 2
α = [(S0/β)*(exp(β*t[k+1]) - exp(β*t[k])) for k in 1:K+1]
α1 = S0/δ_dh
C, d = zeros(K+1,K+1) + I, 300*ones(K+1)
t1 = time_ns()
x_opt, obj_opt, depth  = rpt_bb_dike_height(t,α,α1,δ_dh,λ,θ,β,c,b,T,C,d,δ,use_lmi)
t2 = time_ns()
total_time = (t2-t1)*10^(-9)
push!(obj_vals_total, obj_opt)
push!(times_total, total_time)
push!(tree_depths_total, depth)

instances_dh = ["10,25", "10,50", "10,Ir", "15,25", "15,50", "15,Ir", "16,25", "16,50", "16,Ir"]

results = DataFrame("Instance"=>instances_dh, "Obj_Val"=>obj_vals_total, "Time"=>times_total, "Depth"=>tree_depths_total)

println(results)
