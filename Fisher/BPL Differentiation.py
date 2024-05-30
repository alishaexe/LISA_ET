import sympy as sp
from sympy import *
from sympy import Array, Symbol
import numpy as np
#%%
# (f/fstar)**n1 * (1+(f/fstar)**s)**(-(n1-n2)/s)
#%%
f0 = Symbol('f0')
om = Symbol('om')
f = Symbol('f')
n1 = Symbol('n1')
n2 = Symbol('n2')
astar = Symbol('astar')
s = Symbol('s')
nt = Symbol('nt')

# ogw = (10**astar) * ((f/f0)**n1 * (1/2 + (1/2)*(f/f0)**s)**(-(n1-n2)/s))
# ogw = omegstar * ((f/f0)**n1 * (1+(f/f0)**10)**(-(n1-n2)/10))
ogw = om *(f/f0)**nt

def diff(param1, param2):
    res = ogw.diff(param1)*ogw.diff(param2)
    return res

# paramlist = np.array([[om], [n1], [n2]])
paramlist = np.array([[om], [nt]])

# A = Matrix([[om], [n1], [n2]])
A = Matrix([[om], [nt]])

params = np.array(np.meshgrid(A, A)).T.reshape(-1,2)
# rows_to_remove = [3, 6, 7]
rows_to_remove = [2]

# # Remove rows
# params = np.delete(params, rows_to_remove, axis=0)
params = np.delete(params, rows_to_remove, axis=0)

diffs = np.array(list(map(lambda args: diff(*args), params)))

