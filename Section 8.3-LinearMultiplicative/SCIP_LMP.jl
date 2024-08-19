using LinearAlgebra, Statistics, StatsBase, Distributions, JuMP, DelimitedFiles
using Gurobi, Random, Ipopt, Mosek, MosekTools, Convex, SCS, SCIP, CSV, DataFrames

function scip_lmp_direct(A,b,C,d)
    L, n = size(C)
    p = size(A,1)
    m1 = Model(SCIP.Optimizer)
    set_optimizer_attribute(m1, "display/verblevel", 0)
    set_optimizer_attribute(m1, "limits/time", 3600)
    set_optimizer_attribute(m1, "limits/gap", 1e-4)
    @variable(m1, x[1:n]>=0)
    @variable(m1, t)
    @variable(m1, c[1:p])
    @constraint(m1, C*x .<= d)
    @constraint(m1, [i in 1:p], A[i,:]'*x + b[i] == c[i])
    @NLconstraint(m1, prod(c[i] for i in 1:p) == t)
    @objective(m1, Min, t)
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.SLOW_PROGRESS
        obj_val = objective_value(m1)
    elseif termination_status(m1) == MOI.TIME_LIMIT && has_values(m1)
        obj_val = objective_value(m1)
    else
        obj_val = 1e18
    end
    return obj_val
end

function scip_lmp_biconjugate(A,b,C,d)
    L, n = size(C)
    p = size(A,1)
    m1 = Model(SCIP.Optimizer)
    set_optimizer_attribute(m1, "display/verblevel", 0)
    set_optimizer_attribute(m1, "limits/time", 3600)
    set_optimizer_attribute(m1, "limits/gap", 1e-4)
    @variable(m1, x[1:n]>=0)
    @variable(m1, y[1:p]>=0)
    @variable(m1, w[1:p])
    @constraint(m1, C*x .<= d)
    @NLconstraint(m1, [i in 1:p], exp(-w[i]-1) <= y[i])
    @objective(m1, Max, (A*x.+b)'*y + sum(w[i] for i in 1:p))
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.SLOW_PROGRESS
        obj_val = objective_value(m1)
    elseif termination_status(m1) == MOI.TIME_LIMIT && has_values(m1)
        obj_val = objective_value(m1)
    else
        obj_val = 1e18
    end
    return obj_val
end


##########################   Experiments    ###############################

run_scip_dir = false

obj_vals_total = []
times_total = []
tree_depths_total = []


# Instance 1
n, L, p = 5, 5, 5
C_mat = CSV.read("Data/LMP/LMP_C_5_5_5.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LMP/LMP_d_5_5_5.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times = [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    A = -C[L+1:L+p,:]
    b = d[L+1:L+p]
    if run_scip_dir
        t1 = time_ns()
        obj_opt = scip_lmp_direct(A,b,C,d)
        t2 = time_ns()
    else
        t1 = time_ns()
        obj_opt = scip_lmp_biconjugate(A,b,C,d)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    i += n
    cnt += 1
end
if run_scip_dir
    push!(obj_vals_total, log.(obj_vals))
else
    push!(obj_vals_total, obj_vals)
end
push!(times_total, times)


# Instance 2
n, L, p = 7, 7, 10
C_mat = CSV.read("Data/LMP/LMP_C_7_7_10.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LMP/LMP_d_7_7_10.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times = [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    A = -C[L+1:L+p,:]
    b = d[L+1:L+p]
    if run_scip_dir
        t1 = time_ns()
        obj_opt = scip_lmp_direct(A,b,C,d)
        t2 = time_ns()
    else
        t1 = time_ns()
        obj_opt = scip_lmp_biconjugate(A,b,C,d)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    i += n
    cnt += 1
end
if run_scip_dir
    push!(obj_vals_total, log.(obj_vals))
else
    push!(obj_vals_total, obj_vals)
end
push!(times_total, times)

# Instance 3
n, L, p = 10, 10, 9
C_mat = CSV.read("Data/LMP/LMP_C_10_10_9.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LMP/LMP_d_10_10_9.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times = [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    A = -C[L+1:L+p,:]
    b = d[L+1:L+p]
    if run_scip_dir
        t1 = time_ns()
        obj_opt = scip_lmp_direct(A,b,C,d)
        t2 = time_ns()
    else
        t1 = time_ns()
        obj_opt = scip_lmp_biconjugate(A,b,C,d)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    i += n
    cnt += 1
end
if run_scip_dir
    push!(obj_vals_total, log.(obj_vals))
else
    push!(obj_vals_total, obj_vals)
end
push!(times_total, times)

# Instance 4
n, L, p = 20, 20, 8
C_mat = CSV.read("Data/LMP/LMP_C_20_20_8.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LMP/LMP_d_20_20_8.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times = [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    A = -C[L+1:L+p,:]
    b = d[L+1:L+p]
    if run_scip_dir
        t1 = time_ns()
        obj_opt = scip_lmp_direct(A,b,C,d)
        t2 = time_ns()
    else
        t1 = time_ns()
        obj_opt = scip_lmp_biconjugate(A,b,C,d)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    i += n
    cnt += 1
end
if run_scip_dir
    push!(obj_vals_total, log.(obj_vals))
else
    push!(obj_vals_total, obj_vals)
end
push!(times_total, times)


# Instance 5
n, L, p = 40, 40, 4
C_mat = CSV.read("Data/LMP/LMP_C_40_40_4.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LMP/LMP_d_40_40_4.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times = [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    A = -C[L+1:L+p,:]
    b = d[L+1:L+p]
    if run_scip_dir
        t1 = time_ns()
        obj_opt = scip_lmp_direct(A,b,C,d)
        t2 = time_ns()
    else
        t1 = time_ns()
        obj_opt = scip_lmp_biconjugate(A,b,C,d)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    i += n
    cnt += 1
end
if run_scip_dir
    push!(obj_vals_total, log.(obj_vals))
else
    push!(obj_vals_total, obj_vals)
end
push!(times_total, times)

obj_vals_total = DataFrame(hcat(obj_vals_total...))
times_total = DataFrame(hcat(times_total...))
obj_vals_avg = [mean(obj_vals_total[:,i]) for i in 1:5]
times_avg = [mean(times_total[:,i]) for i in 1:5]
results_avg = DataFrame("Instance"=>1:5, "Obj_Val"=>obj_vals_avg, "Time"=>times_avg)
println("-----------------------------------------")
println("Total objective values: ")
println(obj_vals_total)
println("Total times: ")
println(times_total)
println("-----------------------------------------")
println("Results average: ")
println(results_avg)
