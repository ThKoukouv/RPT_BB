import numpy as np
from pyomo.environ import *
import pandas as pd
import time 

def solve_problem_X1(A,b,C,d,m,K):
    model = ConcreteModel() 
    model.n = RangeSet(1, C.shape[1])
    model.l = RangeSet(1, b.shape[0])
    model.x = Var(model.n, domain=NonNegativeReals) 
    model.y = Var(model.l, domain=NonNegativeReals)

    model.cons_x = ConstraintList()
    for i in range(C.shape[0]):
        model.cons_x.add( sum( C[i,j-1]*model.x[j] for j in model.n) <= d[i] )
        
    counter = 1 
    model.cons_y = ConstraintList()
    for k in range(K):
        model.cons_y.add( sum(model.y[q] for q in range(counter,counter+m)) == 1 ) 
        counter += m 
        
    def obj_fun(model): 
        return sum((sum(A[k-1,q-1]*model.x[k] for k in model.n) + b[q-1]) * model.y[q] for q in model.l)

    model.obj = Objective(rule=obj_fun, sense=maximize)
    solver = SolverFactory('baron')
    solver.options['maxtime'] = 3600 
    solver.options['epsa'] = 1e-4
    solver.solve(model)
    return model.obj() 

def solve_problem_X2(A,b,C,d,alpha,m,K):
    model = ConcreteModel()
    model.n = RangeSet(1, C.shape[1]) 
    model.l = RangeSet(1, b.shape[0])
    model.x = Var(model.n, domain=NonNegativeReals)
    model.y = Var(model.l, domain=NonNegativeReals)

    model.cons_x = ConstraintList()
    for i in range(C.shape[0]):
        model.cons_x.add( sum( C[i,j-1]*model.x[j] for j in model.n) <= d[i] )
    model.cons_x.add(log(sum( exp(model.x[j]) for j in model.n)) <= alpha)

    counter = 1 
    model.cons_y = ConstraintList()
    for k in range(K):
        model.cons_y.add( sum(model.y[q] for q in range(counter,counter+m)) == 1 ) 
        counter += m

    def obj_fun(model): 
        return sum((sum(A[i-1,j-1]*model.x[i] for i in model.n) + b[j-1]) * model.y[j] for j in model.l)

    model.obj = Objective(rule=obj_fun, sense=maximize)
    solver = SolverFactory('baron')
    solver.options['maxtime'] = 3600 
    solver.options['epsa'] = 1e-4
    solver.solve(model)
    return model.obj()

def solve_problem_X3(A,b,C,d,c,m,K):
    model = ConcreteModel()
    model.n = RangeSet(1, C.shape[1]) 
    model.l = RangeSet(1, b.shape[0])
    model.x = Var(model.n, domain=NonNegativeReals)
    model.y = Var(model.l, domain=NonNegativeReals)

    model.cons_x = ConstraintList()
    for i in range(C.shape[0]):
        model.cons_x.add( sum( C[i,j-1]*model.x[j] for j in model.n) <= d[i] )
    model.cons_x.add(sqrt(sum(model.x[j]**2 for j in model.n)) + sum(sqrt(model.x[j]) for j in model.n) <= c)

    counter = 1 
    model.cons_y = ConstraintList()
    for k in range(K):
        model.cons_y.add( sum(model.y[i] for i in range(counter,counter+m)) == 1 ) 
        counter += m

    def obj_fun(model): 
        return sum((sum(A[i-1,j-1]*model.x[i] for i in model.n) + b[j-1]) * model.y[j] for j in model.l)

    model.obj = Objective(rule=obj_fun, sense=maximize)
    solver = SolverFactory('baron')
    solver.options['maxtime'] = 3600 
    solver.options['epsa'] = 1e-4
    solver.solve(model)
    return model.obj() 

run_x1 = False 
run_x2 = False 
run_x3 = False 

obj_vals_total, times_total = [], [] 

# Instance 1 
n, L, m, K, alpha, c, M = 5, 5, 5, 1, 3, 6, 100
A = pd.read_csv("Data/SOMOL/RPTproblem1.csv", header=None).to_numpy()
b = np.zeros(A.shape[0])
C = np.identity(n)
d = [n/i for i in range(1,n+1)]  

if run_x1: 
    start = time.time() 
    opt = solve_problem_X1(A.T,b,C,d,m,K) 
    end = time.time()  
if run_x2: 
    start = time.time()
    opt = solve_problem_X2(A.T,b,C,d,alpha,m,K) 
    end = time.time()
if run_x3: 
    start = time.time()
    opt = solve_problem_X3(A.T,b,C,d,c,m,K) 
    end = time.time()
    
obj_vals_total.append(opt) 
times_total.append(end-start)
 
# Instance 1a 
n, L, m, K, alpha, c, M = 5, 5, 5, 1, 3, 6, 100
A_mat = pd.read_csv("Data/SOMOL/RPTproblem1aAA.csv", header=None).to_numpy()
b = np.zeros(A.shape[0])
C = np.identity(n)
d = [n/i for i in range(1,n+1)]  

times, obj_vals = [], []
i = 0
while i < 10*n: 
    A = A_mat[:,i:i+n] 
    if run_x1: 
        start = time.time() 
        opt = solve_problem_X1(A.T,b,C,d,m,K) 
        end = time.time() 
    if run_x2: 
        start = time.time() 
        opt = solve_problem_X2(A.T,b,C,d,alpha,m,K) 
        end = time.time() 
    if run_x3: 
        start = time.time() 
        opt = solve_problem_X3(A.T,b,C,d,c,m,K)  
        end = time.time() 
    times.append(end - start)
    obj_vals.append(opt)
    i += n  

obj_vals_total.append(np.mean(obj_vals)) 
times_total.append(np.mean(times)) 

# Instance 2 
n, L, m, K, alpha, c, M = 5, 5, 5, 10, 3, 6, 100
A = pd.read_csv("Data/SOMOL/RPTproblem2.csv", header=None).to_numpy()
b = np.zeros(A.shape[0])
C = np.identity(n)
d = [n/i for i in range(1,n+1)]  

if run_x1: 
    start = time.time() 
    opt = solve_problem_X1(A.T,b,C,d,m,K) 
    end = time.time()  
if run_x2: 
    start = time.time()
    opt = solve_problem_X2(A.T,b,C,d,alpha,m,K) 
    end = time.time()
if run_x3: 
    start = time.time()
    opt = solve_problem_X3(A.T,b,C,d,c,m,K) 
    end = time.time()

obj_vals_total.append(opt) 
times_total.append(end-start)

# Instance 2a 
n, L, m, K, alpha, c, M = 5, 5, 5, 10, 3, 6, 100
A_mat = pd.read_csv("Data/SOMOL/RPTproblem2aAA.csv", header=None).to_numpy()
b = np.zeros(A.shape[0])
C = np.identity(n)
d = [n/i for i in range(1,n+1)]  

times, obj_vals = [], []
i = 0
while i < 10*n: 
    A = A_mat[:,i:i+n] 
    if run_x1: 
        start = time.time() 
        opt = solve_problem_X1(A.T,b,C,d,m,K) 
        end = time.time() 
    if run_x2: 
        start = time.time() 
        opt = solve_problem_X2(A.T,b,C,d,alpha,m,K) 
        end = time.time() 
    if run_x3: 
        start = time.time() 
        opt = solve_problem_X3(A.T,b,C,d,c,m,K) 
        end = time.time() 
    times.append(end - start)
    obj_vals.append(opt)
    i += n 

obj_vals_total.append(np.mean(obj_vals)) 
times_total.append(np.mean(times)) 

# Instance 3
n, L, m, K, alpha, c, M = 20, 5, 10, 10, 11, 25, 1000
A = pd.read_csv("Data/SOMOL/RPTproblem3.csv", header=None).to_numpy()
b = np.zeros(A.shape[0])
C = np.identity(n)
d = [n/i for i in range(1,n+1)]  

if run_x1: 
    start = time.time() 
    opt = solve_problem_X1(A.T,b,C,d,m,K) 
    end = time.time()  
if run_x2: 
    start = time.time()
    opt = solve_problem_X2(A.T,b,C,d,alpha,m,K) 
    end = time.time()
if run_x3: 
    start = time.time()
    opt = solve_problem_X3(A.T,b,C,d,c,m,K) 
    end = time.time() 

obj_vals_total.append(opt) 
times_total.append(end-start) 

# Instance 3a 
n, L, m, K, alpha, c, M = 20, 5, 10, 10, 11, 25, 1000
A_mat = pd.read_csv("Data/SOMOL/RPTproblem3aAA.csv", header=None).to_numpy()
b = np.zeros(A.shape[0])
C = np.identity(n)
d = [n/i for i in range(1,n+1)]  

times, obj_vals = [], []
i = 0
while i < 10*n: 
    A = A_mat[:,i:i+n] 
    if run_x1: 
        start = time.time() 
        opt = solve_problem_X1(A.T,b,C,d,m,K) 
        end = time.time() 
    if run_x2: 
        start = time.time() 
        opt = solve_problem_X2(A.T,b,C,d,alpha,m,K) 
        end = time.time() 
    if run_x3: 
        start = time.time() 
        opt = solve_problem_X3(A.T,b,C,d,c,m,K)  
        end = time.time() 
    times.append(end - start)
    obj_vals.append(opt)
    i += n  

obj_vals_total.append(np.mean(obj_vals)) 
times_total.append(np.mean(times)) 

# Instance 4
n, L, m, K, alpha, c, M = 10, 10, 5, 2, 3, 7, 1000
A = pd.read_csv("Data/SOMOL/RPTproblem7A.csv", header=None).to_numpy() 
b = pd.read_csv("Data/SOMOL/RPTproblem7b.csv", header=None).to_numpy()[:,0] 
C = pd.read_csv("Data/SOMOL/RPTproblem7matrixD.csv", header=None).to_numpy() 
d = pd.read_csv("Data/SOMOL/RPTproblem7D.csv", header=None).to_numpy()[:,0]

if run_x1: 
    start = time.time() 
    opt = solve_problem_X1(A.T,b,C,d,m,K) 
    end = time.time()  
if run_x2: 
    start = time.time()
    opt = solve_problem_X2(A.T,b,C,d,alpha,m,K) 
    end = time.time()
if run_x3: 
    start = time.time()
    opt = solve_problem_X3(A.T,b,C,d,c,m,K) 
    end = time.time()

obj_vals_total.append(opt) 
times_total.append(end-start) 

# Instance 4a 
n, L, m, K, alpha, c, M = 10, 10, 5, 2, 3, 7, 1000
A_mat = pd.read_csv("Data/SOMOL/RPTproblem7aAA.csv", header=None).to_numpy()
b_mat = pd.read_csv("Data/SOMOL/RPTproblem7abb.csv", header=None).to_numpy()
C_mat = pd.read_csv("Data/SOMOL/RPTproblem7aDD.csv", header=None).to_numpy()
d_mat = pd.read_csv("Data/SOMOL/RPTproblem7addvector.csv", header=None).to_numpy() 

times, obj_vals = [], []
i, cnt = 0, 0
while i < 10*n: 
    A = A_mat[:,i:i+n] 
    b = b_mat[:,cnt] 
    C = C_mat[:,i:i+n] 
    d = d_mat[:,cnt]
    if run_x1: 
        start = time.time() 
        opt = solve_problem_X1(A.T,b,C,d,m,K) 
        end = time.time() 
    if run_x2: 
        start = time.time() 
        opt = solve_problem_X2(A.T,b,C,d,alpha,m,K) 
        end = time.time() 
    if run_x3: 
        start = time.time() 
        opt = solve_problem_X3(A.T,b,C,d,c,m,K) 
        end = time.time() 
    times.append(end - start)
    obj_vals.append(opt)
    i += n 
    cnt += 1

obj_vals_total.append(np.mean(obj_vals)) 
times_total.append(np.mean(times)) 

# Instance 5
n, L, m, K, alpha, c, M = 20, 10, 10, 10, 5, 30, 1000
A = pd.read_csv("Data/SOMOL/RPTproblem11A.csv", header=None).to_numpy()
b = pd.read_csv("Data/SOMOL/RPTproblem11b.csv", header=None).to_numpy()[:,0] 
C1 = np.array([-3,7,0,-5,1,1,0,2,-1,1])
C2 = np.array([7,0,-5,1,1,0,2,-1,-1,1])
C3 = np.array([0,-5,1,1,0,2,-1,-1,-9,1])
C4 = np.array([-5,1,1,0,2,-1,-1,-9,3,1])
C5 = np.array([1,1,0,2,-1,-1,-9,3,5,1])
C6 = np.array([1,0,2,-1,-1,-9,3,5,0,1])
C7 = np.array([0,2,-1,-1,-9,3,5,0,0,1])
C8 = np.array([2,-1,-1,-9,3,5,0,0,1,1])
C9 = np.array([-1,-1,-9,3,5,0,0,1,7,1])
C10 = np.array([-1,-9,3,5,0,0,1,7,-7,1])
C11 = np.array([-9,3,5,0,0,1,7,-7,-4,1])
C12 = np.array([3,5,0,0,1,7,-7,-4,-6,1])
C13 = np.array([5,0,0,1,7,-7,-4,-6,-3,1])
C14 = np.array([0,0,1,7,-7,-4,-6,-3,7,1])
C15 = np.array([0,1,7,-7,-4,-6,-3,7,0,1])
C16 = np.array([1,7,-7,-4,-6,-3,7,0,-5,1])
C17 = np.array([7,-7,-4,-6,-3,7,0,-5,1,1])
C18 = np.array([-7,-4,-6,-3,7,0,-5,1,1,1])
C19 = np.array([-4,-6,-3,7,0,-5,1,1,0,1])
C20 = np.array([-6,-3,7,0,-5,1,1,0,2,1])
C = np.array([C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20]) 
d = [-5,2,-1,-3,5,4,-1,0,9,40]  
C = C.T

if run_x1: 
    start = time.time() 
    opt = solve_problem_X1(A.T,b,C,d,m,K) 
    end = time.time()  
if run_x2: 
    start = time.time()
    opt = solve_problem_X2(A.T,b,C,d,alpha,m,K) 
    end = time.time()
if run_x3: 
    start = time.time()
    opt = solve_problem_X3(A.T,b,C,d,c,m,K) 
    end = time.time()

obj_vals_total.append(opt) 
times_total.append(end-start) 

# Instance 5a 
n, L, m, K, alpha, c, M = 20, 10, 10, 10, 5, 30, 1000
A_mat = CSV.read("Data/SOMOL/RPTproblem11aAA.csv", header=None).to_numpy()
b_mat = CSV.read("Data/SOMOL/RPTproblem11abb.csv", header=None).to_numpy()

times, obj_vals = [], []
i, cnt = 0, 0
while i < 10*n: 
    A = A_mat[:,i:i+n] 
    b = b_mat[:,cnt] 
    if run_x1: 
        start = time.time() 
        opt = solve_problem_X1(A.T,b,C,d,m,K) 
        end = time.time() 
    if run_x2: 
        start = time.time() 
        opt = solve_problem_X2(A.T,b,C,d,alpha,m,K) 
        end = time.time() 
    if run_x3: 
        start = time.time() 
        opt = solve_problem_X3(A.T,b,C,d,c,m,K) 
        end = time.time() 
    times.append(end - start)
    obj_vals.append(opt)
    i += n 
    cnt += 1 

obj_vals_total.append(np.mean(obj_vals)) 
times_total.append(np.mean(times)) 


instances_sml = ["1", "1a", "2", "2a", "3", "3a", "4", "4a", "5", "5a"]

results = pd.DataFrame({'Instance': instances_sml, 'Obj_Val': obj_vals_total, 'Time': times_total})
print(results)

