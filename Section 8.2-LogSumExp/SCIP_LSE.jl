using LinearAlgebra, Statistics, StatsBase, Distributions, JuMP, DelimitedFiles
using Gurobi, Random, Ipopt, Mosek, MosekTools, Convex, SCS, SCIP, CSV, DataFrames

function scip_lse_direct(C,d)
    L, n = size(C)
    m1 = Model(SCIP.Optimizer)
    set_optimizer_attribute(m1, "display/verblevel", 0)
    set_optimizer_attribute(m1, "limits/time", 3600)
    set_optimizer_attribute(m1, "limits/gap", 1e-4)
    @variable(m1, x[1:n]>=0)
    @variable(m1, t)
    @constraint(m1, C*x .<= d)
    @NLconstraint(m1, log(sum(exp(x[i]) for i in 1:n)) == t)
    @objective(m1, Max, t)
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

function scip_lse_biconjugate(C,d)
    L, n = size(C)
    m1 = Model(SCIP.Optimizer)
    set_optimizer_attribute(m1, "display/verblevel", 0)
    set_optimizer_attribute(m1, "limits/time", 3600)
    set_optimizer_attribute(m1, "limits/gap", 1e-4)
    @variable(m1, x[1:n]>=0)
    @variable(m1, y[1:n]>=0)
    @variable(m1, w[1:n])
    @constraint(m1, C*x .<= d)
    @constraint(m1, sum(y[i] for i in 1:n) == 1)
    @NLconstraint(m1, [i in 1:n], y[i]*exp(w[i]/y[i]) <= 1)
    @objective(m1, Max, x'*y + sum(w[i] for i in 1:n))
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


#######################     Experiments     ##########################

run_scip_dir = false

obj_vals_total = []
times_total = []
tree_depths_total = []

# Instance 1
n = 10
C_mat = CSV.read("Data/LSE/LogSumExp_C_10_20.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LSE/LogSumExp_d_10_20.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    if run_scip_dir
        t1 = time_ns()
        obj_opt = scip_lse_direct(C,d)
        t2 = time_ns()
    else
        t1 = time_ns()
        obj_opt = scip_lse_biconjugate(C,d)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    i += n
    cnt += 1
end
push!(obj_vals_total, mean(obj_vals))
push!(times_total, mean(times))


# Instance 2
n = 40
C_mat = CSV.read("Data/LSE/LogSumExp_C_40_80.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LSE/LogSumExp_d_40_80.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    if run_scip_dir
        t1 = time_ns()
        obj_opt = scip_lse_direct(C,d)
        t2 = time_ns()
    else
        t1 = time_ns()
        obj_opt = scip_lse_biconjugate(C,d)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    i += n
    cnt += 1
end
push!(obj_vals_total, mean(obj_vals))
push!(times_total, mean(times))

# Instance 3
n = 10
C_mat = CSV.read("Data/LSE/LogSumExp_C_10_100.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LSE/LogSumExp_d_10_100.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    if run_scip_dir
        t1 = time_ns()
        obj_opt = scip_lse_direct(C,d)
        t2 = time_ns()
    else
        t1 = time_ns()
        obj_opt = scip_lse_biconjugate(C,d)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    i += n
    cnt += 1
end
push!(obj_vals_total, mean(obj_vals))
push!(times_total, mean(times))

# Instance 4
n = 20
C_mat = CSV.read("Data/LSE/LogSumExp_C_20_20.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LSE/LogSumExp_d_20_20.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    if run_scip_dir
        t1 = time_ns()
        obj_opt = scip_lse_direct(C,d)
        t2 = time_ns()
    else
        t1 = time_ns()
        obj_opt = scip_lse_biconjugate(C,d)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    i += n
    cnt += 1
end
push!(obj_vals_total, mean(obj_vals))
push!(times_total, mean(times))

# Instance 5
n = 50
C_mat = CSV.read("Data/LSE/LogSumExp_C_50_50.csv", DataFrame, header = false)
d_mat = CSV.read("Data/LSE/LogSumExp_d_50_50.csv", DataFrame, header = false)
C_mat = convert(Matrix, C_mat[:,:])
d_mat = convert(Matrix, d_mat[:,:])
obj_vals, times = [], [], []
i, cnt = 1, 1
while i < 10*n
    C = C_mat[:,i:i+n-1]
    d = d_mat[:,cnt]
    if run_scip_dir
        t1 = time_ns()
        obj_opt = scip_lse_direct(C,d)
        t2 = time_ns()
    else
        t1 = time_ns()
        obj_opt = scip_lse_biconjugate(C,d)
        t2 = time_ns()
    end
    total_time = (t2-t1)*10^(-9)
    push!(obj_vals,obj_opt)
    push!(times,total_time)
    i += n
    cnt += 1
end
push!(obj_vals_total, mean(obj_vals))
push!(times_total, mean(times))


results = DataFrame("Instance"=>1:5, "Obj_Val"=>obj_vals_total, "Time"=>times_total)

println(results)
