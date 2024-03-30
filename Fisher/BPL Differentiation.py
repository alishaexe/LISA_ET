import sympy as sp
from sympy import *
from sympy import Array, Symbol
import numpy as np
#%%
# (f/fstar)**n1 * (1+(f/fstar)**s)**(-(n1-n2)/s)
#%%
f0 = Symbol('f0')
#omegstar = Symbol('omegstar')
f = Symbol('f')
n1 = Symbol('n1')
n2 = Symbol('n2')
astar = Symbol('astar')

ogw = (10**astar) * ((f/f0)**n1 * (1+(f/f0)**10)**(-(n1-n2)/10))
#ogw = omegstar * ((f/f0)**n1 * (1+(f/f0)**10)**(-(n1-n2)/10))

def diff(param1, param2):
    res = ogw.diff(param1)*ogw.diff(param2)
    return res

paramlist = np.array([[astar], [n1], [n2]])


A = Matrix([[astar], [n1], [n2]])

params = np.array(np.meshgrid(A, A)).T.reshape(-1,2)
rows_to_remove = [3, 6, 7]

# Remove rows
params = np.delete(params, rows_to_remove, axis=0)

diffs = np.array(list(map(lambda args: diff(*args), params)))

