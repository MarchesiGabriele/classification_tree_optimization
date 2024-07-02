import pyomo.environ as pyo
import numpy as np
import sys
from codice.mytree import Tree

EPSILON = 0.000001 

def modello_albero(alpha, beta, delta, u, umax, umin, n_features, xtrain, ytrain, tree:Tree, lcap, yclass) -> pyo.ConcreteModel:
    model = pyo.ConcreteModel(name = "(H)")


    # SETS
    P = [i for i in range(len(xtrain[0]))] # features index
    TB = [i for i in range(1, np.power(2, delta))]
    TB1 = [i for i in range(2, np.power(2, delta))]
    TL = [i for i in range(np.power(2, delta), np.power(2, delta + 1))]
    I = [i for i in range(len(xtrain))] # datapoints index
    K = [i for i in range(len(np.unique(ytrain)))] # index classes
    T = TB + TL

    # TODO CREARE SET ANTENATI SINISTRA E DESTRA ALBERO ?????



    # VARIABLES
    model.d = pyo.Var(TB, domain=pyo.Binary) 
    model.a = pyo.Var(P, TB, domain=pyo.Binary) 
    model.b = pyo.Var(TB, domain=pyo.NonNegativeReals) 
    model.l = pyo.Var(TL, domain=pyo.NonNegativeIntegers) 
    model.z = pyo.Var(I, T, domain=pyo.Binary) 
    model.l = pyo.Var(T, domain=pyo.Binary) 
    model.n1 = pyo.Var(K, T, domain=pyo.NonNegativeIntegers) # Nkt
    model.n2 = pyo.Var(T, domain=pyo.NonNegativeIntegers) # Nt


    # PARAMETERS  
    def x_init(mdl, i):
        return xtrain[i]
    model.x = pyo.Param(I, initialize=x_init, within=pyo.Any)
    def y_init(mdl, i):
        return ytrain[i]
    model.y = pyo.Param(I, initialize=y_init, within=pyo.Any)
    def yclass_init(mdl, i, k):
        return yclass[i][k]
    model.yclass = pyo.Param(I, K, initialize=yclass_init, within=pyo.Any)

    model.alpha = pyo.Param(initialize=alpha)
    model.beta = pyo.Param(initialize=beta)
    model.delta = pyo.Param(initialize=delta)
    model.u = pyo.Param(initialize=u)
    model.umax = pyo.Param(initialize=umax)
    model.umin = pyo.Param(initialize=umin)
    model.tree= pyo.Param(initialize=tree)
    model.lcap= pyo.Param(initialize=lcap)


    # CONSTRAINTS
    def c1_rule(mdl, t):
        return np.sum(model.a[j][t] for j in P) == model.d[t]
    model.c1 = pyo.Constraint(TB, rule=c1_rule)

    def c2_rule(mdl, t):
        return model.b[t] >= 0
    model.c2 = pyo.Constraint(TB, rule=c2_rule)   

    def c3_rule(mdl, t):
        return model.b[t] <= model.d[t] 
    model.c3 = pyo.Constraint(TB, rule=c3_rule)   

    def c4_rule(mdl, t):
        return model.d[t] <= model.d[tree.get_parent(t)]
    model.c4 = pyo.Constraint(TB1, rule=c4_rule)   

    def c5_rule(mdl, t):
        return model.d[t] <= model.d[tree.get_parent(t)]
    model.c5 = pyo.Constraint(TB1, rule=c5_rule)   

    def c6_rule(mdl, i, t):
        return model.z[i][t] <= model.lsmall[t] 
    model.c6 = pyo.Constraint(I, TL, rule=c6_rule)   

    def c7_rule(mdl, t):
        return np.sum(model.z[i][t] for i in I)  >= model.beta*model.lsmall[t]
    model.c7 = pyo.Constraint(TL, rule=c7_rule)   

    def c8_rule(mdl, t):
        return np.sum(model.z[i][t] for i in TL)  == 1
    model.c8 = pyo.Constraint(I,  rule=c8_rule)   

    # TODO VINCOLO 8-9 SLIDE

    def c11_rule(mdl, t, k):
        return model.n1[k][t] == np.sum(model.z[i][t]*model.yclass[i][k] for i in I)
    model.c11 = pyo.Constraint(TL, K,  rule=c11_rule)   









    # OBJECTIVE FUNCTION
    def obj_rule(mdl):
        return (1/model.lcap)*np.sum(model.l[t] for t in TL) + model.alpha*sum(model.d[t] for t in TB)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

