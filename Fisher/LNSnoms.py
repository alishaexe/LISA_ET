import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


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
#Change this value for how many 'steps' you want in the range of values

itera = 10000

elminL = (np.log10(ffmin))
elmaxL = (np.log10(10**(-0.95)))
elminet = np.log10(1.6)
elmaxet = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera

#%%
L = 25/3
fLisa = 1/(2*pi*L)
c = 3e8

P = 12
A = 3

def P_acc(f):
    res = A**2 *(1e-15)**2 * (1+(0.4e-3 / f)**2)*(1+(f/8e-3)**4)*1/(2*pi*f)**(4)*(2*pi*f/c)**2
    return res

def P_ims(f):
    res = P**2 * (1e-12)**2 *(1+(2e-3/f)**4)*(2*pi*f/c)**2
    return res

def N_xx(f):
    con = 2*pi*f*L
    res = 16 * (np.sin(con))**2 * ((3+np.cos(2*con))*P_acc(f)+P_ims(f))
    return res

def R_XX(f):#this is Rxx
    res = 16*(np.sin(2*pi*f*L))**2  * (2*pi*f*L)**2 * 3/10 * 1/(1+0.6*(2*pi*f*L)**2)
    return res

def S_n(f):
    res = 1/np.sqrt(2)*N_xx(f)/R_XX(f)
    return res

def Ohms(f):
    const = 4*pi**2/(3*H0**2)
    res = const*f**3*S_n(f)
    return res


freqvals = np.logspace(elminL, elmaxL, itera)   
sigvals = np.array(list(map(Ohms, freqvals)))

#%%
def sigp(f):
    f0 = 1
    t1 = ((9*((f/f0)**(-30))) + (5.5e-6*((f/f0)**(-4.5))) +(0.28e-11*((f/f0)**3.2)))*(0.5-0.5*(np.tanh(0.06*((f/f0)-42))))
    t2 = ((1e-13*((f/f0)**(1.9))) + (20e-13*((f/f0)**(2.8))))*0.5*(np.tanh(0.06*((f/f0)-42)))
    t3 = 1-(0.475*np.exp(-(((f/f0)-25)**2)/50))
    t4 = 1-(5e-4*np.exp(-(((f/f0)-20)**2)/100))
    t5 = 1-(0.2*np.exp(-((((f/f0)-47)**2)**0.85)/100))
    t6 = 1-(0.12*np.exp(-((((f/f0)-50)**2)**0.7)/100))-(0.2*np.exp(-(((f/f0)-45)**2)/250))+(0.15*np.exp(-(((f/f0)-85)**2)/400))
    res = 0.88*(t1+t2)*t3*t4*t5*t6
    return 0.816**2 * res



def etnomonly(f):
    res = sigp(f)
    if res > 10**(-5):
        return
    return res

def sigETapp(f):#Sigma_Ohm approx
    res = sigp(f)
    return res


fvalsET = np.logspace(np.log10(1), np.log10(445),itera)#frequency values
sigETvals = np.array(list(map(etnomonly, fvalsET)))#The Omega_gw values from the ET data

#%%
def omegatog(f):
    if f <= 10**(-0.9):
        return Ohms(f)
    if f > 1.6:
        return sigETapp(f)


def nomtog(f):
    if f <= 10**(-0.9):
        res = Ohms(f)
        if res > 1e-5:
            return
        return res
    if f > 1:
        res = sigETapp(f)
        if res > 1e-5:
            return
        return res

step = (ffmax-ffmin)/itera
freqs = np.linspace(np.log10(ffmin), np.log10(ffmax), itera)
fvalscomb = np.logspace(np.log10(ffmin), np.log10(ffmax),itera)
#%%
# def logncomb(f, fstar, rho):
#     res = np.exp(-1/(2*rho)*(np.log10(f/fstar))**2)
#     return res

# def aminlogcomb(fstar, rho):
#     integrand = lambda f, fstar, rho: (logncomb(f, fstar, rho)/Ohms(f))**2
#     I1 = quad(integrand, ffmin, 10**(-3), args=(fstar, rho))[0]
#     I2 = quad(integrand, 10**(-3), 10**(-1), args=(fstar, rho))[0]
#     I3 = quad(integrand, 10**(-1), 1e0, args=(fstar, rho))[0]
#     I4 = quad(integrand, 1e0, 0.5e1, args=(fstar, rho))[0]
#     I5 = quad(integrand, 0.5e1, ffmax, args=(fstar, rho))[0]
#     integrand2 = lambda f, fstar, rho: (logncomb(f, fstar, rho)/sigp(f))**2
#     I6 = quad(integrand2, ffmin, 1e-1, args=(fstar, rho))[0]
#     I7 = quad(integrand2, 1e-1, ffmax, args=(fstar, rho))[0]
#     res = snr5/np.sqrt(2*T*sum((I1, I2, I3, I4, I5, I6, I7)))
#     return res 

# rhom = np.linspace(0.1, 10, 15)

# fLisa = 1/(2*pi*L)
# ffmin = 10**(-5)
# mm = 1e-1
# ffmax = 445
# elmin = np.log10(ffmin)
# elmax = np.log10(ffmax)

# lsc = np.linspace(elmin, elmax,100)
# flsc = 10**lsc



# valslogcom = np.array(np.meshgrid(flsc, rhom)).T.reshape(-1,2)
# aminvalscom = np.array(list(map(lambda args: aminlogcomb(*args), valslogcom)))

# atablogcom = np.vstack((valslogcom.T, aminvalscom)).T.reshape(-1,len(rhom), 3)

# def fLlogtabc(i, j, k):
#     res = atablogcom[i, j, 2]*logncomb(flsc[k], atablogcom[i, j, 0], atablogcom[i, j, 1])
#     return res

# i = range(len(flsc))
# j = range(len(rhom))
# k = range(len(flsc))

# coords = np.array(np.meshgrid(i, j, k)).T.reshape(-1,3)

# Ftablogcomb = np.array(list(map(lambda args: fLlogtabc(*args), coords))).reshape(len(flsc),len(rhom),len(flsc))

# def maxlogvalsc(l):
#     maximslog = np.log(np.max(Ftablogcomb[l]))
#     return maximslog


# maxposlog = range(len(Ftablogcomb))
# maxlogcom = np.array(list(map(maxlogvalsc, maxposlog)))
# flogplotc = np.vstack((np.log(flsc), maxlogcom)).T

#%%

om1 = 3e-10
fs1 = 0.2
r1 = 0.45

def log1(f):
    res = om1*np.exp(-(1/(2*r1**2)) * (np.log10(f/fs1))**2)
    return res


om2 = 7e-13
fs2 = 0.25
r2 = 1.6

elmin = np.log10(ffmin)

lsc = np.linspace(elmin, 10**(-0.95),1000)
tf = 10**lsc


def log2(f):
    res = om2*np.exp(-1/(2*r2**2)*np.log10(f/fs2)**2)
    return res

phase1 = np.array(list(map(log1, fvalscomb)))
combine = np.array(list(map(omegatog, fvalscomb)))
nom = np.array(list(map(nomtog, fvalscomb)))
otog = np.vstack((fvalscomb, combine)).T
phase2 = np.array(list(map(log2, fvalscomb)))

flogplotc = np.load("flogplotc.npy")
plt.figure(figsize=(6, 8))
plt.loglog(otog[:,0], nom,\
           color = "indigo", linewidth=1.5, \
           label = "Nominal Curves")
plt.loglog(fvalscomb, phase1,linewidth=2.5,\
     color = "darkgreen",\
           label = "Inf 1")
plt.loglog(fvalscomb, phase2,linewidth=2.5,\
     color = "blue",linestyle='--',\
           label = "Inf 2")
# plt.loglog(np.exp(flogplotc[:,0]), np.exp(flogplotc[:,1]), color = "aqua", label = " Combined LogNs", linewidth=2.5)

# plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1])/0.67**2, label = "BPLS curve", color = "lime", linewidth=2.5)

plt.legend(fontsize = 12, loc = (0.35,0.04))
plt.title('SGWB from Inflation', fontsize = 16)#plt.title(r"Values: \
#n1 = {nom1}, n2 = {nom2}, $f_\star$ = {fbreak}, $\sigma$= {sig}, $\alpha\star$ = {astar}".format(nom1=n1, nom2=n2, fbreak=fstar, sig = s, astar = a), fontsize = 10)
plt.grid(True) 
plt.xlim(ffmin, ffmax) 
plt.ylim(10**(-17),10**(-5))
plt.xlabel('f (Hz)', fontsize = 16)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/Inflation.png', bbox_inches='tight')
plt.show()












