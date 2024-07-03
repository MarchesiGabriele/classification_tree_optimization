import pyomo.environ as pyo
import numpy as np
import sys
from codice.mytree import Tree

def modello_albero(alpha, beta, delta, u, umax, umin, n_features, xtrain, ytrain, tree:Tree, lcap, yclass, mvalue) -> pyo.ConcreteModel:
    model = pyo.ConcreteModel(name = "(H)")


    # SETS
    P = [i for i in range(len(xtrain[0]))] # features index
    print(len(xtrain[0]))
    TB = [i for i in range(1, np.power(2, delta))]
    TB1 = [i for i in range(2, np.power(2, delta))]
    TL = [i for i in range(np.power(2, delta), np.power(2, delta + 1))]
    I = [i for i in range(len(xtrain))] # datapoints index
    K = [i for i in range(len(np.unique(ytrain)))] # index classes
    mm = [tree.get_left_ancestors(t) for t in TL]
    mm1 = [tree.get_right_ancestors(t) for t in TL]

    print(mm)

    TOT1 =  [(i, TL[t], m) for i in I for t in range(len(TL)) for m in mm[t]]
    TOT2 =  [(i, TL[t], m) for i in I for t in range(len(TL)) for m in mm1[t]]


    # VARIABLES
    model.d = pyo.Var(TB, domain=pyo.Binary) 
    model.a = pyo.Var(P, TB, domain=pyo.Binary) 
    model.b = pyo.Var(TB, domain=pyo.NonNegativeReals) 
    model.l = pyo.Var(TL, domain=pyo.NonNegativeIntegers) 
    model.z = pyo.Var(I, TL, domain=pyo.Binary) 
    model.lsmall = pyo.Var(TL, domain=pyo.Binary) 
    model.n1 = pyo.Var(K, TL, domain=pyo.NonNegativeIntegers) # Nkt
    model.n2 = pyo.Var(TL, domain=pyo.NonNegativeIntegers) # Nt
    model.c = pyo.Var(K, TL, domain=pyo.Binary) 

    print([model.d[i].value for i in TB])

    # PARAMETERS  
    def x_init(mdl, i, j):
        return xtrain[i, j]
    model.x = pyo.Param(I, P, initialize=x_init, within=pyo.Any)
    def y_init(mdl, i):
        return ytrain[i]
    model.y = pyo.Param(I, initialize=y_init, within=pyo.Any)
    def yclass_init(mdl, i, k):
        return yclass[i][k]
    model.yclass = pyo.Param(I, K, initialize=yclass_init, within=pyo.Any)

    model.alpha = pyo.Param(initialize=alpha)
    model.beta = pyo.Param(initialize=beta)
    model.delta = pyo.Param(initialize=delta)
    def u_init(mdl, p):
        return u[p]
    model.u= pyo.Param(P, initialize=u_init, within=pyo.Any)
    model.umax = pyo.Param(initialize=umax)
    model.umin = pyo.Param(initialize=umin)
    model.lcap = pyo.Param(initialize=lcap)
    model.mvalue = pyo.Param(initialize=mvalue)


    # CONSTRAINTS
    def c1_rule(mdl, t):
        return np.sum(model.a[j,t] for j in P) == model.d[t]
    model.c1 = pyo.Constraint(TB, rule=c1_rule)

    def c2_rule(mdl, t):
        return model.b[t] >= 0
    model.c2 = pyo.Constraint(TB, rule=c2_rule)   

    def c3_rule(mdl, t):
        return model.b[t] <= model.d[t] 
    model.c3 = pyo.Constraint(TB, rule=c3_rule)   

    """
    # TODO: REMOVE
    def c4_rule(mdl, t):
        return model.d[t] <= model.d[tree.get_parent(t)]
    model.c4 = pyo.Constraint(TB1, rule=c4_rule)   
    """
    def c5_rule(mdl, t):
        return model.d[t] <= model.d[tree.get_parent(t)]
    model.c5 = pyo.Constraint(TB1, rule=c5_rule)   

    def c6_rule(mdl, i, t):
        return model.z[i,t] <= model.lsmall[t] 
    model.c6 = pyo.Constraint(I, TL, rule=c6_rule)   

    def c7_rule(mdl, t):
        return np.sum(model.z[i,t] for i in I)  >= model.beta*model.lsmall[t]
    model.c7 = pyo.Constraint(TL, rule=c7_rule)   

    def c8_rule(mdl, i):
        return np.sum(model.z[i,t] for t in TL) == 1
    model.c8 = pyo.Constraint(I,  rule=c8_rule)   

    def c9_rule(mdl, i, t, m):
        return  np.sum(model.a[j,m]*(model.x[i,j] + model.u[j] - model.umin) for j in P) + model.umin <= model.b[m] + (1+model.umax)*(1-model.z[i,t])
    model.c9 = pyo.Constraint(TOT1, rule=c9_rule)   

    def c10_rule(mdl, i, t, m):
        return np.sum(model.a[j,m]*(model.x[i,j]) for j in P) >= model.b[m] - (1-model.z[i,t])
    model.c10 = pyo.Constraint(TOT2, rule=c10_rule)   

    def c11_rule(mdl, t, k):
        return model.n1[k,t] == np.sum(model.z[i,t]*model.yclass[i,k] for i in I)
    model.c11 = pyo.Constraint(TL, K,  rule=c11_rule)   

    def c12_rule(mdl, t):
        return model.n2[t] == np.sum(model.z[i,t] for i in I)
    model.c12 = pyo.Constraint(TL, rule=c12_rule)   

    def c13_rule(mdl, t):
        return np.sum(model.c[k,t] for k in K) == model.lsmall[t]
    model.c13 = pyo.Constraint(TL, rule=c13_rule)   

    def c14_rule(mdl, t, k):
        return model.l[t] >= model.n2[t] - model.n1[k,t] -  mvalue*(1-model.c[k,t])
    model.c14 = pyo.Constraint(TL, K, rule=c14_rule)   

    def c15_rule(mdl, t, k):
        return model.l[t] <= model.n2[t] - model.n1[k,t] +  mvalue*model.c[k,t]
    model.c15 = pyo.Constraint(TL, K, rule=c15_rule)   

    # OBJECTIVE FUNCTION
    def obj_rule(mdl):
        return (1/model.lcap)*np.sum(model.l[t] for t in TL) + model.alpha*sum(model.d[t] for t in TB)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

