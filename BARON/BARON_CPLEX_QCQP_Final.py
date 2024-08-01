import numpy as np
from pyomo.environ import *
import pandas as pd
import time
import docplex
from docplex.mp.model import Model

def solve_problem_cplex(C,d,P,q,r):
    qcqp_model = Model('qcqp')
    n = C.shape[1]
    x = qcqp_model.continuous_var_list(n, lb=0)
    for i in range(C.shape[0]):
        qcqp_model.add_constraint(sum(C[i,j]*x[j] for j in range(n)) <= d[i]) 
    
    obj_fn = sum(P[i,j]*x[i]*x[j] for i in range(n) for j in range(n)) + sum(q[i]*x[i] for i in range(n)) + r
    qcqp_model.set_objective("max", obj_fn) 
    qcqp_model.parameters.optimalitytarget.set(3)
    qcqp_model.parameters.mip.tolerances.mipgap = 1e-4
    sol = qcqp_model.solve()  
    return sol 

def solve_problem(C,d,P,q,r,quad_cons_data):
    model = ConcreteModel()
    model.n = RangeSet(1, C.shape[1]) 
    model.x = Var(model.n, domain=NonNegativeReals)

    model.cons_x = ConstraintList()
    for i in range(C.shape[0]):
        model.cons_x.add( sum( C[i,q-1]*model.x[q] for q in model.n) <= d[i] )
    
    for k in quad_cons_data: 
        P1, q1, r1 = k[0], k[1], k[2] 
        model.cons_x.add( sum( P1[i-1,j-1]*model.x[i]*model.x[j] for i in model.n for j in model.n ) 
                        + sum( q1[i-1]*model.x[i] for i in model.n) 
                        + r1 <= 0)
    def obj_fun(model): 
        return sum( P[i-1,j-1]*model.x[i]*model.x[j] for i in model.n for j in model.n ) + sum( q[i-1]*model.x[i] for i in       model.n) + r
                        

    model.obj = Objective(rule=obj_fun, sense=minimize)
    solver = SolverFactory('baron')
    solver.options['maxtime'] = 3600 
    solver.options['epsa'] = 1e-2
    solver.solve(model)
    return model.obj()

use_cplex = False

obj_vals_total, times_total = [], []

# Problem 1
n, L = 20, 10
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

P = -0.5*np.eye(n) 
q = 2*np.ones(n) 
r = -2*n 
quad_cons_data = []  

if use_cplex: 
    start = time.time() 
    opt = solve_problem_cplex(C,d,-P,-q,-r) 
    end = time.time() 
else: 
    start = time.time() 
    opt = solve_problem(C,d,P,q,r,quad_cons_data) 
    end = time.time() 
    
if use_cplex:  
    obj_vals_total.append(opt)
else:  
    obj_vals_total.append(-opt)

times_total.append(end - start)

# Problem 2
n, L = 20, 10
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

P = -0.5*np.eye(n) 
q = -5*np.ones(n) 
r = -12.5*n 
quad_cons_data = []

if use_cplex: 
    start = time.time() 
    opt = solve_problem_cplex(C,d,-P,-q,-r) 
    end = time.time() 
else: 
    start = time.time() 
    opt = solve_problem(C,d,P,q,r,quad_cons_data) 
    end = time.time() 

if use_cplex:  
    obj_vals_total.append(opt)
else: 
    obj_vals_total.append(-opt)

times_total.append(end - start)

# Problem 3
P = pd.read_csv("Data/QCQP/QCQP_P3_P.csv", header=None).to_numpy() 
q = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P3_q.csv", header=None).to_numpy()) 
C = pd.read_csv("Data/QCQP/QCQP_P3_C.csv", header=None).to_numpy() 
d = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P3_d.csv", header=None).to_numpy()) 
r = 0 
quad_cons_data = []

if use_cplex: 
    start = time.time() 
    opt = solve_problem_cplex(C,d,-P,-q,-r) 
    end = time.time() 
else: 
    start = time.time() 
    opt = solve_problem(C,d,P,q,r,quad_cons_data) 
    end = time.time() 

if use_cplex:  
    obj_vals_total.append(opt)
else: 
    obj_vals_total.append(-opt)

times_total.append(end - start)

# Problem 4
P = pd.read_csv("Data/QCQP/QCQP_P4_P.csv", header=None).to_numpy() 
q = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P4_q.csv", header=None).to_numpy()) 
C = pd.read_csv("Data/QCQP/QCQP_P4_C.csv", header=None).to_numpy() 
d = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P4_d.csv", header=None).to_numpy()) 
r = 0 
quad_cons_data = []  

if use_cplex: 
    start = time.time() 
    opt = solve_problem_cplex(C,d,-P,-q,-r) 
    end = time.time() 
else: 
    start = time.time() 
    opt = solve_problem(C,d,P,q,r,quad_cons_data) 
    end = time.time() 

if use_cplex:  
    obj_vals_total.append(opt)
else: 
    obj_vals_total.append(-opt)

times_total.append(end - start)

# Problem 5
P = pd.read_csv("Data/QCQP/QCQP_P5_P.csv", header=None).to_numpy() 
q = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P5_q.csv", header=None).to_numpy()) 
C = pd.read_csv("Data/QCQP/QCQP_P5_C.csv", header=None).to_numpy() 
d = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P5_d.csv", header=None).to_numpy()) 
r = 0 
quad_cons_data = [] 

if use_cplex: 
    start = time.time() 
    opt = solve_problem_cplex(C,d,-P,-q,-r) 
    end = time.time() 
else: 
    start = time.time() 
    opt = solve_problem(C,d,P,q,r,quad_cons_data) 
    end = time.time() 

if use_cplex:  
    obj_vals_total.append(opt)
else: 
    obj_vals_total.append(-opt)

times_total.append(end - start)


# Problem 6
n, m = 8, 4
P_mat = pd.read_csv("Data/QCQP/QCQP_P6_P.csv", header=None).to_numpy() 
q_mat = pd.read_csv("Data/QCQP/QCQP_P6_q.csv", header=None).to_numpy()
r_mat = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P6_r.csv", header=None).to_numpy())
C = pd.read_csv("Data/QCQP/QCQP_P6_C.csv", header=None).to_numpy() 
d = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P6_d.csv", header=None).to_numpy()) 
P = P_mat[:,:n]
q = q_mat[:,0] 
r = r_mat[0]
quad_cons_data = [] 
cnt = 1
for i in range(1,m+1): 
    P1 = P_mat[:,i*n:(i+1)*n] 
    q1 = q_mat[:,cnt] 
    r1 = r_mat[cnt]  
    quad_cons_data.append([P1,q1,r1]) 
    cnt += 1
    
start = time.time() 
opt = solve_problem(C,d,P,q,r,quad_cons_data) 
end = time.time() 

obj_vals_total.append(-opt)
times_total.append(end - start)

# Problem 7 
n, m = 12, 6
P_mat = pd.read_csv("Data/QCQP/QCQP_P7_P.csv", header=None).to_numpy() 
q_mat = pd.read_csv("Data/QCQP/QCQP_P7_q.csv", header=None).to_numpy()
r_mat = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P7_r.csv", header=None).to_numpy())
C = pd.read_csv("Data/QCQP/QCQP_P7_C.csv", header=None).to_numpy() 
d = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P7_d.csv", header=None).to_numpy()) 
P = P_mat[:,:n]
q = q_mat[:,0] 
r = r_mat[0]
quad_cons_data = [] 
cnt = 1
for i in range(1,m+1): 
    P1 = P_mat[:,i*n:(i+1)*n] 
    q1 = q_mat[:,cnt] 
    r1 = r_mat[cnt]  
    quad_cons_data.append([P1,q1,r1]) 
    cnt += 1
    
start = time.time() 
opt = solve_problem(C,d,P,q,r,quad_cons_data) 
end = time.time() 

obj_vals_total.append(-opt)
times_total.append(end - start)


# Problem 8 
n, m = 16, 8
P_mat = pd.read_csv("Data/QCQP/QCQP_P8_P.csv", header=None).to_numpy() 
q_mat = pd.read_csv("Data/QCQP/QCQP_P8_q.csv", header=None).to_numpy()
r_mat = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P8_r.csv", header=None).to_numpy())
C = pd.read_csv("Data/QCQP/QCQP_P8_C.csv", header=None).to_numpy() 
d = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P8_d.csv", header=None).to_numpy()) 
P = P_mat[:,:n]
q = q_mat[:,0] 
r = r_mat[0]
quad_cons_data = [] 
cnt = 1
for i in range(1,m+1): 
    P1 = P_mat[:,i*n:(i+1)*n] 
    q1 = q_mat[:,cnt] 
    r1 = r_mat[cnt]  
    quad_cons_data.append([P1,q1,r1]) 
    cnt += 1 
    
start = time.time() 
opt = solve_problem(C,d,P,q,r,quad_cons_data) 
end = time.time() 

obj_vals_total.append(-opt)
times_total.append(end - start) 

# Problem 9 
n, m = 30, 15
P_mat = pd.read_csv("Data/QCQP/QCQP_P9_P.csv", header=None).to_numpy() 
q_mat = pd.read_csv("Data/QCQP/QCQP_P9_q.csv", header=None).to_numpy()
r_mat = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P9_r.csv", header=None).to_numpy())
C = pd.read_csv("Data/QCQP/QCQP_P9_C.csv", header=None).to_numpy() 
d = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P9_d.csv", header=None).to_numpy()) 
P = P_mat[:,:n]
q = q_mat[:,0] 
r = r_mat[0]
quad_cons_data = [] 
cnt = 1
for i in range(1,m+1): 
    P1 = P_mat[:,i*n:(i+1)*n] 
    q1 = q_mat[:,cnt] 
    r1 = r_mat[cnt]  
    quad_cons_data.append([P1,q1,r1]) 
    cnt += 1 
    
start = time.time() 
opt = solve_problem(C,d,P,q,r,quad_cons_data) 
end = time.time() 

obj_vals_total.append(-opt)
times_total.append(end - start)


# Problem 10 
n, m = 40, 20
P_mat = pd.read_csv("Data/QCQP/QCQP_P10_P.csv", header=None).to_numpy() 
q_mat = pd.read_csv("Data/QCQP/QCQP_P10_q.csv", header=None).to_numpy()
r_mat = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P10_r.csv", header=None).to_numpy())
C = pd.read_csv("Data/QCQP/QCQP_P10_C.csv", header=None).to_numpy() 
d = np.squeeze(pd.read_csv("Data/QCQP/QCQP_P10_d.csv", header=None).to_numpy()) 
P = P_mat[:,:n]
q = q_mat[:,0] 
r = r_mat[0]
quad_cons_data = [] 
cnt = 1
for i in range(1,m+1): 
    P1 = P_mat[:,i*n:(i+1)*n] 
    q1 = q_mat[:,cnt] 
    r1 = r_mat[cnt]  
    quad_cons_data.append([P1,q1,r1]) 
    cnt += 1 

start = time.time() 
opt = solve_problem(C,d,P,q,r,quad_cons_data) 
end = time.time() 

obj_vals_total.append(-opt)
times_total.append(end - start)

if use_cplex:  
    instances_qcqp = np.arange(1,6)
else: 
    instances_qcqp = np.arange(1,11)

results = pd.DataFrame({'Instance': instances_qcqp, 'Obj_Val': obj_vals_total, 'Time': times_total})
print(results)

