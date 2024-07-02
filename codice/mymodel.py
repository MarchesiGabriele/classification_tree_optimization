import pyomo.environ as pyo
import numpy as np
import sys

EPSILON = 0.000001 

def modello_albero(x, y, m, n, v) -> pyo.ConcreteModel:
    """
    x: training data matrix
    y: target vector
    m: n° of datapoints
    n: n° of features
    v: regularization parameter
    """
    model = pyo.ConcreteModel(name = "(H)")

    # SETS
    I = [i for i in range(m)] # set of datapoints
    F = [i for i in range(n)] # set of features

    # VARIABLES
    model.w = pyo.Var(F) # vector of n elements
    model.gamma = pyo.Var() # scalar
    model.xi = pyo.Var(I, domain = pyo.NonNegativeReals) # vector of m errors

    # PARAMETERS  
    def x_init(mdl, i):
        return x[i]
    model.x = pyo.Param(I, initialize=x_init, within=pyo.Any)
    def y_init(mdl, i):
        return y[i]
    model.y = pyo.Param(I, initialize=y_init, within=pyo.Any)

    # CONSTRAINTS
    def c1_rule(mdl, i):
        return mdl.y[i]*(np.dot(mdl.w, mdl.x[i])-mdl.gamma) >= 1 - mdl.xi[i]
    model.c1 = pyo.Constraint(I, rule=c1_rule)
    def c2_rule(mdl, i):
        return mdl.xi[i] >= 0
    model.c2 = pyo.Constraint(I, rule=c2_rule)   

    # OBJECTIVE FUNCTION
    def obj_rule(mdl):
        return my_norm(mdl.w) + v*np.sum(mdl.xi[i] for i in I)            
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    return model

