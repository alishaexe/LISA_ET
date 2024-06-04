import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import *
from sympy import Array, Symbol
#%%
#Differentiation of LogNs
f0 = Symbol('f0')
om = Symbol('om')
f = Symbol('f')
s = Symbol('s')
rho = Symbol('rho')
def diff(param1, param2):
    res = ogw.diff(param1)*ogw.diff(param2)
    return res

#A logarithm function with any base b can be written as the natural log over the
#log of that base: log10(x) = ln(x)/log(10) or log6(x) = ln(x)/log(6)

ogw = om*sp.exp(-1/(2*rho**2)*(sp.log(f/f0))**2)
A = Matrix([[om], [rho]])
params = np.array(np.meshgrid(A, A)).T.reshape(-1,2)
rows_to_remove = [2]
params = np.delete(params, rows_to_remove, axis=0)

diffs = np.array(list(map(lambda args: diff(*args), params)))