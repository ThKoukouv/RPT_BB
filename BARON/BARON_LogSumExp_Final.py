import numpy as np
from pyomo.environ import *
import pandas as pd
import time 

def solve_problem_direct(C,d):
    model = ConcreteModel()
    model.n = RangeSet(1, C.shape[1]) 
    model.x = Var(model.n, domain=NonNegativeReals)

    model.cons_x = ConstraintList()
    for i in range(C.shape[0]):
        model.cons_x.add( sum( C[i,q-1]*model.x[q] for q in model.n) <= d[i] )
    
    def obj_fun(model): 
        return log(sum(exp(model.x[i]) for i in model.n))
    
    model.obj = Objective(rule=obj_fun, sense=maximize)
    solver = SolverFactory('baron')
    solver.options['maxtime'] = 3600 
    solver.options['epsa'] = 1e-4
    solver.solve(model)
    return model.obj() 
    
def solve_problem_biconjugate(C,d):
    model = ConcreteModel()
    model.n = RangeSet(1, C.shape[1]) 
    model.x = Var(model.n, domain=NonNegativeReals)
    model.y = Var(model.n, domain=NonNegativeReals) 
    model.w = Var(model.n)

    model.cons_x = ConstraintList()
    for i in range(C.shape[0]):
        model.cons_x.add( sum( C[i,q-1]*model.x[q] for q in model.n) <= d[i] )
    
    model.cons_y = ConstraintList()
    for i in model.n: 
        model.cons_y.add( model.y[i]*exp(model.w[i]/model.y[i]) <= 1 ) 
    model.cons_y.add( sum(model.y[i] for i in model.n) == 1) 
    
    def obj_fun(model): 
        return sum(model.x[i]*model.y[i] for i in model.n) + sum(model.w[i] for i in model.n)
    
    model.obj = Objective(rule=obj_fun, sense=maximize)
    solver = SolverFactory('baron')
    solver.options['maxtime'] = 3600
    solver.options['epsa'] = 1e-4
    solver.solve(model)
    return model.obj()

use_dir = True  

obj_vals_total, times_total = [], []

# Instance 1
n = 10
C_mat = pd.read_csv("Data/LogSumExp_C_10_20.csv", header=None).to_numpy()
d_mat = np.squeeze(pd.read_csv("Data/LogSumExp_d_10_20.csv", header=None).to_numpy()) 
obj_vals, times = [], []
i = 0 
cnt = 0
while i < 10*n: 
    C = C_mat[:,i:i+n] 
    d = d_mat[:,cnt] 
    if use_dir: 
        start = time.time() 
        opt = solve_problem_direct(C,d) 
        end = time.time() 
    else:
        start = time.time()
        opt = solve_problem_biconjugate(C,d)
        end = time.time()
    times.append(end - start)
    obj_vals.append(opt)
    i += n
    cnt += 1

obj_vals_total.append(np.mean(obj_vals))
times_total.append(np.mean(times))

# Instance 2
n = 40
C_mat = pd.read_csv("Data/LogSumExp_C_40_80.csv", header=None).to_numpy()
d_mat = np.squeeze(pd.read_csv("Data/LogSumExp_d_40_80.csv", header=None).to_numpy()) 
obj_vals, times = [], []
i = 0 
cnt = 0
while i < 10*n: 
    C = C_mat[:,i:i+n] 
    d = d_mat[:,cnt] 
    if use_dir: 
        start = time.time() 
        opt = solve_problem_direct(C,d) 
        end = time.time() 
    else:
        start = time.time()
        opt = solve_problem_biconjugate(C,d)
        end = time.time()
    times.append(end - start)
    obj_vals.append(opt)
    i += n
    cnt += 1

obj_vals_total.append(np.mean(obj_vals))
times_total.append(np.mean(times))

# Instance 3
n = 10
C_mat = pd.read_csv("Data/LogSumExp_C_10_100.csv", header=None).to_numpy()
d_mat = np.squeeze(pd.read_csv("Data/LogSumExp_d_10_100.csv", header=None).to_numpy()) 
obj_vals, times = [], []
i = 0 
cnt = 0
while i < 10*n: 
    C = C_mat[:,i:i+n] 
    d = d_mat[:,cnt] 
    if use_dir: 
        start = time.time() 
        opt = solve_problem_direct(C,d) 
        end = time.time() 
    else:
        start = time.time()
        opt = solve_problem_biconjugate(C,d)
        end = time.time()
    times.append(end - start)
    obj_vals.append(opt)
    i += n
    cnt += 1

obj_vals_total.append(np.mean(obj_vals))
times_total.append(np.mean(times))

# Instance 4
n = 20
C_mat = pd.read_csv("Data/LogSumExp_C_20_20.csv", header=None).to_numpy()
d_mat = np.squeeze(pd.read_csv("Data/LogSumExp_d_20_20.csv", header=None).to_numpy()) 
obj_vals, times = [], []
i = 0 
cnt = 0
while i < 10*n: 
    C = C_mat[:,i:i+n] 
    d = d_mat[:,cnt] 
    if use_dir: 
        start = time.time() 
        opt = solve_problem_direct(C,d) 
        end = time.time() 
    else:
        start = time.time()
        opt = solve_problem_biconjugate(C,d)
        end = time.time()
    times.append(end - start)
    obj_vals.append(opt)
    i += n
    cnt += 1
    
obj_vals_total.append(np.mean(obj_vals))
times_total.append(np.mean(times))

# Instance 5
n = 50
C_mat = pd.read_csv("Data/LogSumExp_C_50_50.csv", header=None).to_numpy()
d_mat = np.squeeze(pd.read_csv("Data/LogSumExp_d_50_50.csv", header=None).to_numpy()) 
obj_vals, times = [], []
i = 0 
cnt = 0
while i < 10*n: 
    C = C_mat[:,i:i+n] 
    d = d_mat[:,cnt] 
    if use_dir: 
        start = time.time() 
        opt = solve_problem_direct(C,d) 
        end = time.time() 
    else:
        start = time.time()
        opt = solve_problem_biconjugate(C,d)
        end = time.time()
    times.append(end - start)
    obj_vals.append(opt)
    i += n
    cnt += 1

obj_vals_total.append(np.mean(obj_vals))
times_total.append(np.mean(times))

results = pd.DataFrame({'Instance': [1, 2, 3, 4, 5], 'Obj_Val': obj_vals_total, 'Time': times_total})
print(results)

