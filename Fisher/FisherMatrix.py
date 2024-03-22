import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import sympy as sp
from sympy import *
from sympy import Array, Symbol
import getdist
from getdist import plots, MCSamples
from scipy.interpolate import UnivariateSpline
import time
#%%
#Now going to define constants
yr = 365*24*60*60 #in seconds
H0 = 100*0.67*10**(3)/(3.086*10**(22)) #1/seconds
pi = np.pi

fetstar = 10**(-2)
fi = 0.4*10**(-3)
#For the LISA mission they have designed the
#arms to be length L = 2.5*10^(6)
T = 3*yr
snr5 = 5

ffmin = 10**(-5)
ffmax = 445
###############

#%%
#All the functions for ET and LISA
#Not combined ones though
def SI(f):
    si = 5.76*10**(-48)*(1+(fi/f)**2)
    return si
def Sa(f):
    sa = 1/4 *SI(f)/((2*pi*f)**4)
    return sa

SII = 3.6*10**(-41)
Ss = SII

f2 = 25*10**(-3)

####LOW FREQUENCY APPROXIMATION
#Now make the low frequency approximation
#this is equation 63 in the paper

def sigI(f):#Sigma_I
    sig = np.sqrt(2)*20/3 * (SI(f)/(2*pi*f)**4 + SII)*(1+(f/f2)**2)
    return sig

def SigmaLisaApprox(f):#Sigma_Ohm approx
    const = ((4*pi**2/(3*H0**2)))
    res = const * f**3 *sigI(f)
    return res

L = 25/3
fLisa = 1/(2*pi*L)
#%%


f0 = Symbol('f0')
omegstar = Symbol('omegstar')
f = Symbol('f')
nt = Symbol('nt')


ogw = omegstar * (f/f0)**(nt)
siget = (9*f**(-30)) + (5*10**(-6)*f**(-4.5)) + (3*10**(-11)*f**2.1)

def diff(param1, param2):
    res = ogw.diff(param1)*ogw.diff(param2)
    return res

paramlist = np.array([omegstar, nt])
params = np.array(np.meshgrid(paramlist, paramlist)).T.reshape(-1,2)
diffs = np.array(list(map(lambda args: diff(*args), params)))

#%%
#This calculates the fisher matrix for LISA
start = time.time()

def Fisher(differential):
    os = 10**(-10)
    ntv = 2/3
    f0v = 0.1
    integrand = lambda f: differential.subs({omegstar: os, nt: ntv, f0:f0v})/SigmaLisaApprox(f)**2
    print('after integrand', differential)
    res = integrate(integrand(f), (f, 1e-5, 10**(-1)))
    print('finished this int', differential)
    return res

    
FM = np.array(list(map(Fisher, diffs))).reshape(2,2)
FM = FM.astype(np.float64)
FM2 = T * FM
np.save("FMLISAA.npy", FM2)
pause = time.time()
print(pause-start, "Just LISA")


#%%
#This calculates the Fishermatrix for ET if it ever finishes running

# def sigp(f):
#     res = 1.3*((3*30*10**(-1)*f**(-30)+5*10**(-6)*f**(-4.5)+0.6*10**(-11)*f**(2.8))
#                 *(1/2-1/2*sp.tanh(0.1*(f-42)))+(1/2*sp.tanh(0.1*(f-42)))*(2*10**(-11)*f**2.25 
#                                                                           +10**(-13)*f**3))
#     return res

def sigp(f):
    res = (9*f**(-30)) + (5*10**(-6)*f**(-4.5)) + (3*10**(-11)*f**2.1)
    return res 

#%%
secstart = time.time()
def FisherET(differential):
    os = 10**(-10)
    ntv = 2/3
    f0v = 0.1
    integrand = lambda f: differential.subs({omegstar: os, nt: ntv, f0:f0v})/sigp(f)**2
    print('after integrand', differential)
    res = integrate(integrand(f), (f,1.6,445))
    print('finished this int', differential)
    return res.subs({omegstar: os, nt: ntv, f0: f0v})

FMET = np.array(list(map(FisherET, diffs))).reshape(2,2)
FMET = FMET.astype(np.float64)
FMet2 = T* FMET
end = time.time()
np.save("FMETA.npy", FMet2)
#&&
###############################
#Now doing the scenario B

def Fisher(differential):
    os = 10**(-9)
    ntv = 0.01
    f0v = 0.1
    integrand = lambda f: differential.subs({omegstar: os, nt: ntv, f0:f0v})/SigmaLisaApprox(f)**2
    print('after integrand', differential)
    res = integrate(integrand(f), (f, 1e-5, 10**(-1)))
    print('finished this int', differential)
    return res

    
FMB = np.array(list(map(Fisher, diffs))).reshape(2,2)
FMB = FM.astype(np.float64)
FM2B = T * FM
np.save("FMLISAB.npy", FM2B)
pause = time.time()
print(pause-start, "Just LISA")


#%%
#This calculates the Fishermatrix for ET if it ever finishes running
secstart = time.time()
def FisherET(differential):
    os = 10**(-9)
    ntv = 2/3
    f0v = 0.01
    integrand = lambda f: differential.subs({omegstar: os, nt: ntv, f0:f0v})/sigp(f)**2
    print('after integrand', differential)
    res = integrate(integrand(f), (f,1.6,445))
    print('finished this int', differential)
    return res.subs({omegstar: os, nt: ntv, f0: f0v})

FMETB = np.array(list(map(FisherET, diffs))).reshape(2,2)
FMETB = FMET.astype(np.float64)
FMet2B = T* FMET
end = time.time()
np.save("FMETB.npy", FMet2B)
