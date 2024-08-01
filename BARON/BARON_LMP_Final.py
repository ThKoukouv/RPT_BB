import numpy as np
from pyomo.environ import *
import pandas as pd
import time 

def solve_problem_direct(A,b,C,d):
    model = ConcreteModel()
    model.n = RangeSet(1, C.shape[1]) 
    model.x = Var(model.n, domain=NonNegativeReals)

    model.cons_x = ConstraintList()
    for i in range(C.shape[0]):
        model.cons_x.add( sum( C[i,q-1]*model.x[q] for q in model.n) <= d[i] )
    
    def obj_fun(model): 
        return prod( (sum(A[i,j-1]*model.x[j] for j in model.n) + b[i]) for i in range(A.shape[0]) ) 
                        
    model.obj = Objective(rule=obj_fun, sense=minimize)
    solver = SolverFactory('baron')
    solver.options['maxtime'] = 3600 
    solver.options['epsa'] = 1e-4
    solver.solve(model)
    return model.obj() 

def solve_problem_biconjugate(A,b,C,d):
    model = ConcreteModel()
    model.n = RangeSet(1, C.shape[1])  
    model.p = RangeSet(1, b.shape[0])
    model.x = Var(model.n, domain=NonNegativeReals) 
    model.y = Var(model.p, domain=NonNegativeReals)

    model.cons_x = ConstraintList()
    for i in range(C.shape[0]):
        model.cons_x.add( sum( C[i,q-1]*model.x[q] for q in model.n) <= d[i] )
    
    def obj_fun(model): 
        obj1 = -sum((sum(A[i-1,j-1]*model.x[j] for j in model.n) + b[i-1]) * model.y[i] for i in model.p) 
        obj2 = sum(log(model.y[i]) for i in model.p)
        return obj1 + obj2 + A.shape[0] 
                        
    model.obj = Objective(rule=obj_fun, sense=maximize)
    solver = SolverFactory('baron')
    solver.options['maxtime'] = 3600 
    solver.options['epsa'] = 1e-4
    solver.solve(model)
    return model.obj() 

use_dir = True

obj_vals_total, times_total = [], []

# Instance 1
n, L, p = 5, 5, 5 
C_mat = pd.read_csv("Data/LMP_C_5_5_5.csv", header=None).to_numpy()
d_mat = np.squeeze(pd.read_csv("Data/LMP_d_5_5_5.csv", header=None).to_numpy())  
obj_vals, times = [], []
i = 0 
cnt = 0
while i < 10*n: 
    C = C_mat[:,i:i+n] 
    d = d_mat[:,cnt] 
    A = -C[L:L+p,:]  
    b = d[L:L+p] 
    if use_dir: 
        start = time.time() 
        opt = opt = solve_problem_direct(A,b,C,d) 
        end = time.time() 
    else:
        start = time.time()
        opt = solve_problem_biconjugate(A,b,C,d)
        end = time.time()
    times.append(end - start)
    obj_vals.append(opt)
    i += n
    cnt += 1

if use_dir: 
    obj_vals_total.append(np.mean(np.log(obj_vals))) 
else: 
    obj_vals_total.append(np.mean(-np.array(obj_vals)))
times_total.append(np.mean(times))


# Instance 2
n, L, p = 7, 7, 10 
C_mat = pd.read_csv("Data/LMP_C_7_7_10.csv", header=None).to_numpy()
d_mat = np.squeeze(pd.read_csv("Data/LMP_d_7_7_10.csv", header=None).to_numpy())  
obj_vals, times = [], []
i = 0 
cnt = 0
while i < 10*n: 
    C = C_mat[:,i:i+n] 
    d = d_mat[:,cnt] 
    A = -C[L:L+p,:]  
    b = d[L:L+p] 
    if use_dir: 
        start = time.time() 
        opt = opt = solve_problem_direct(A,b,C,d) 
        end = time.time() 
    else:
        start = time.time()
        opt = solve_problem_biconjugate(A,b,C,d)
        end = time.time()
    times.append(end - start)
    obj_vals.append(opt)
    i += n
    cnt += 1

if use_dir: 
    obj_vals_total.append(np.mean(np.log(obj_vals))) 
else: 
    obj_vals_total.append(np.mean(-np.array(obj_vals)))
times_total.append(np.mean(times)) 

# Instance 3
n, L, p = 10, 10, 9 
C_mat = pd.read_csv("Data/LMP_C_10_10_9.csv", header=None).to_numpy()
d_mat = np.squeeze(pd.read_csv("Data/LMP_d_10_10_9.csv", header=None).to_numpy())  
obj_vals, times = [], []
i = 0 
cnt = 0
while i < 10*n: 
    C = C_mat[:,i:i+n] 
    d = d_mat[:,cnt] 
    A = -C[L:L+p,:]  
    b = d[L:L+p] 
    if use_dir: 
        start = time.time() 
        opt = opt = solve_problem_direct(A,b,C,d) 
        end = time.time() 
    else:
        start = time.time()
        opt = solve_problem_biconjugate(A,b,C,d)
        end = time.time()
    times.append(end - start)
    obj_vals.append(opt)
    i += n
    cnt += 1

if use_dir: 
    obj_vals_total.append(np.mean(np.log(obj_vals))) 
else: 
    obj_vals_total.append(np.mean(-np.array(obj_vals)))
times_total.append(np.mean(times))

# Instance 4
n, L, p = 20, 20, 8 
C_mat = pd.read_csv("Data/LMP_C_20_20_8.csv", header=None).to_numpy()
d_mat = np.squeeze(pd.read_csv("Data/LMP_d_20_20_8.csv", header=None).to_numpy())  
obj_vals, times = [], []
i = 0 
cnt = 0
while i < 10*n: 
    C = C_mat[:,i:i+n] 
    d = d_mat[:,cnt] 
    A = -C[L:L+p,:]  
    b = d[L:L+p] 
    if use_dir: 
        start = time.time() 
        opt = opt = solve_problem_direct(A,b,C,d) 
        end = time.time() 
    else:
        start = time.time()
        opt = solve_problem_biconjugate(A,b,C,d)
        end = time.time()
    times.append(end - start)
    obj_vals.append(opt)
    i += n
    cnt += 1

if use_dir: 
    obj_vals_total.append(np.mean(np.log(obj_vals))) 
else: 
    obj_vals_total.append(np.mean(-np.array(obj_vals)))
times_total.append(np.mean(times))


# Instance 5
n, L, p = 40, 40, 4 
C_mat = pd.read_csv("Data/LMP_C_40_40_4.csv", header=None).to_numpy()
d_mat = np.squeeze(pd.read_csv("Data/LMP_d_40_40_4.csv", header=None).to_numpy())  
obj_vals, times = [], []
i = 0 
cnt = 0
while i < 10*n: 
    C = C_mat[:,i:i+n] 
    d = d_mat[:,cnt] 
    A = -C[L:L+p,:]  
    b = d[L:L+p] 
    if use_dir: 
        start = time.time() 
        opt = opt = solve_problem_direct(A,b,C,d) 
        end = time.time() 
    else:
        start = time.time()
        opt = solve_problem_biconjugate(A,b,C,d)
        end = time.time()
    times.append(end - start)
    obj_vals.append(opt)
    i += n
    cnt += 1

if use_dir: 
    obj_vals_total.append(np.mean(np.log(obj_vals))) 
else: 
    obj_vals_total.append(np.mean(-np.array(obj_vals)))
times_total.append(np.mean(times))

results = pd.DataFrame({'Instance': [1, 2, 3, 4, 5], 'Obj_Val': obj_vals_total, 'Time': times_total})
print(results)

