import numpy as np
from pyomo.environ import *
import pandas as pd
import time 

def solve_problem(t,a,a1,delta,lmbd,theta,beta,c,b,T): 
    K = len(t) - 2
    model = ConcreteModel()
    model.n = RangeSet(1,K+1) 
    model.x = Var(model.n, domain=NonNegativeReals) 
    model.h = Var(model.n, domain=NonNegativeReals)

    model.cons_x = ConstraintList()
    for k in range(1,K+2):
        model.cons_x.add( sum( model.x[i] for i in range(1,k+1)) == model.h[k] )

    def obj_fun(model): 
        inv_costs = sum((c+b*model.x[k])*exp(lmbd*model.h[k] - delta*t[k-1]) for k in range(1,K+2)) 
        damage_costs = sum(a[k-1]*exp(-theta*model.h[k]) for k in range(1,K+2)) + a1*exp(beta*T-theta*model.h[K+1]) 
        return inv_costs + damage_costs
                        

    model.obj = Objective(rule=obj_fun, sense=minimize)
    solver = SolverFactory('baron')
    solver.options['maxtime'] = 3600 
    solver.options['epsa'] = 1e-4
    solver.solve(model)
    return model.obj() 

obj_vals_total, times_total = [], []

# 10 rings
alpha = 0.033027
eta = 0.32
zeta = 0.003774
gamma = 0.02
delta = 0.04
lmbd = 0.0014
theta = alpha - zeta
beta1 = alpha*eta + gamma 
beta = beta1 - delta
c = 16.6939
b = 0.6258
T = 300
S0 = 1564.9/2270

# step=25 
t = [0,25,50,75,100,125,150,175,200,225,250,275,T]
K = len(t) - 2
a = [(S0/beta)*(exp(beta*t[k+1]) - exp(beta*t[k])) for k in range(K+1)]
a1 = S0/delta 

start = time.time()
opt = solve_problem(t,a,a1,delta,lmbd,theta,beta,c,b,T)
end = time.time()

obj_vals_total.append(opt)
times_total.append(end - start)

# step=50 
t = [0,50,100,150,200,250,T]
K = len(t) - 2
a = [(S0/beta)*(exp(beta*t[k+1]) - exp(beta*t[k])) for k in range(K+1)]
a1 = S0/delta 

start = time.time()
opt = solve_problem(t,a,a1,delta,lmbd,theta,beta,c,b,T)
end = time.time() 

obj_vals_total.append(opt)
times_total.append(end - start)

# step=irregular 
t = [0,20,50,90,130,155,180,210,255,270,T]
K = len(t) - 2
a = [(S0/beta)*(exp(beta*t[k+1]) - exp(beta*t[k])) for k in range(K+1)]
a1 = S0/delta 

start = time.time()
opt = solve_problem(t,a,a1,delta,lmbd,theta,beta,c,b,T)
end = time.time() 

obj_vals_total.append(opt)
times_total.append(end - start)


# 15 rings 
alpha = 0.0502
eta = 0.76
zeta = 0.003764
gamma = 0.02
delta = 0.04
lmbd = 0.0098
theta = alpha - zeta
beta1 = alpha*eta + gamma 
beta = beta1 - delta
c = 125.6422
b = 1.1268
T = 300
S0 = 11810.4/729

# step=25 
t = [0,25,50,75,100,125,150,175,200,225,250,275,T]
K = len(t) - 2
a = [(S0/beta)*(exp(beta*t[k+1]) - exp(beta*t[k])) for k in range(K+1)]
a1 = S0/delta 

start = time.time()
opt = solve_problem(t,a,a1,delta,lmbd,theta,beta,c,b,T)
end = time.time() 

obj_vals_total.append(opt)
times_total.append(end - start)

# step=50 
t = [0,50,100,150,200,250,T]
K = len(t) - 2
a = [(S0/beta)*(exp(beta*t[k+1]) - exp(beta*t[k])) for k in range(K+1)]
a1 = S0/delta 

start = time.time()
opt = solve_problem(t,a,a1,delta,lmbd,theta,beta,c,b,T)
end = time.time() 

obj_vals_total.append(opt)
times_total.append(end - start)


# step=irregular 
t = [0,20,50,90,130,155,180,210,255,270,T]
K = len(t) - 2
a = [(S0/beta)*(exp(beta*t[k+1]) - exp(beta*t[k])) for k in range(K+1)]
a1 = S0/delta 

start = time.time()
opt = solve_problem(t,a,a1,delta,lmbd,theta,beta,c,b,T)
end = time.time() 

obj_vals_total.append(opt)
times_total.append(end - start)


#16 rings 
alpha = 0.0574
eta = 0.76
zeta = 0.002032
gamma = 0.02
delta = 0.04
lmbd = 0.01
theta = alpha - zeta
beta1 = alpha*eta + gamma 
beta = beta1 - delta
c = 324.6287
b = 2.1304
T = 300
S0 = 22656.5/906

# step=25 
t = [0,25,50,75,100,125,150,175,200,225,250,275,T]
K = len(t) - 2
a = [(S0/beta)*(exp(beta*t[k+1]) - exp(beta*t[k])) for k in range(K+1)]
a1 = S0/delta 

start = time.time()
opt = solve_problem(t,a,a1,delta,lmbd,theta,beta,c,b,T)
end = time.time() 

obj_vals_total.append(opt)
times_total.append(end - start)

# step=50 
t = [0,50,100,150,200,250,T]
K = len(t) - 2
a = [(S0/beta)*(exp(beta*t[k+1]) - exp(beta*t[k])) for k in range(K+1)]
a1 = S0/delta 

start = time.time()
opt = solve_problem(t,a,a1,delta,lmbd,theta,beta,c,b,T)
end = time.time() 

obj_vals_total.append(opt)
times_total.append(end - start)

# step=irregular 
t = [0,20,50,90,130,155,180,210,255,270,T]
K = len(t) - 2
a = [(S0/beta)*(exp(beta*t[k+1]) - exp(beta*t[k])) for k in range(K+1)]
a1 = S0/delta 

start = time.time()
opt = solve_problem(t,a,a1,delta,lmbd,theta,beta,c,b,T)
end = time.time() 

obj_vals_total.append(opt)
times_total.append(end - start)

instances_dh = ["10,25", "10,50", "10,Ir", "15,25", "15,50", "15,Ir", "16,25", "16,50", "16,Ir"]

results = pd.DataFrame({'Instance': instances_dh, 'Obj_Val': obj_vals_total, 'Time': times_total})
print(results)

