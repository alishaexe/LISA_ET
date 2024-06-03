import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
#%%

#Now going to define constants
yr = 365*24*60*60 #in seconds
H0 = 100*0.67*10**(3)/(3.086*10**(22)) #1/seconds
# H0 = 3.24e-18 #Debikas value
#setting h = 0.67
pi = np.pi
c = 3e8
fetstar = 10**(-2)
fi = 0.4*10**(-3)

#For the LISA mission they have designed the
#arms to be length L = 2.5*10^(6)
T = 3*yr
snr5 = 5
#L = 2.5e9
L = 25/3

c = 3e8
fetstar = 10**(-2)
fi = 0.4*10**(-3)

fLisa = 1/(2*pi*L)
ffmin = 10**(-5)
ffmax = 445
elmin = np.log10(ffmin)
elmax = np.log10(ffmax)
###############
#Change this value for how many 'steps' you want in the range of values

itera = 200

##########

elminL = (np.log10(ffmin))
elmaxL = (np.log10(10**(-0.95)))
elminet = np.log10(1.6)
elmaxet = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera
#%%
P = 12
A = 3


def P_acc(f):
    res = A**2 *(1e-15)**2 * (1+(0.4e-3 / f)**2)*(1+(f/8e-3)**4)*(2*pi*f)**(-4)*(2*pi*f/c)**2
    return res

def P_ims(f):#* (1e-12)**2 after P
    res = P**2 * (1e-12)**2 *(1+(2e-3/f)**4)*(2*pi*f/c)**2
    return res

def N_aa(f):
    con = 2*pi*f*L
    res = 8 * (np.sin(con))**2 * (4*(1+np.cos(con)+(np.cos(con))**2)*P_acc(f)+(2+np.cos(con))*P_ims(f))
    return res

def R(f):
    res = 16*(np.sin(2*pi*f*L))**2  * (2*pi*f*L)**2 * 9/20 * 1/(1+0.7*(2*pi*f*L)**2)
    return res

def S_n(f):
    res = N_aa(f)/R(f)
    return res

def Ohms(f):
    const = 4*pi**2/(3*H0**2)
    res = const *f**3*S_n(f)
    return res

freqvals = np.logspace(elminL, elmaxL, itera)   
sigvals = np.array(list(map(Ohms, freqvals)))


#%%
######################
#LogNS curves
######################
#LISA logns plots

    
def lognL(f, fstar, sig):
    res = np.exp(-1/(2*10**sig)*(np.log(f/fstar))**2)
    return res

def aminlogL(fstar, sig):
    integrand = lambda f, fstar, sig: (lognL(f, fstar, sig)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-3), args=(fstar, sig))[0]
    I2 = quad(integrand, 10**(-3), 10**(-1), args=(fstar, sig))[0]
    res = snr5/np.sqrt(2*T*sum((I1,I2)))
    return res

   

ls = np.linspace(elminL, elmaxL, itera)
fls = 10**ls
sigma = np.linspace(-1, 0, 5)


valslog = np.array(np.meshgrid(fls, sigma)).T.reshape(-1,2)
aminvals = np.array(list(map(lambda args: aminlogL(*args), valslog)))

atablog = np.vstack((valslog.T, aminvals)).T.reshape(-1,len(sigma), 3)

def fLlogtab(i, j, k):
    res = atablog[i, j, 2]*lognL(fls[k], atablog[i, j, 0], atablog[i, j, 1])
    return res


i = range(len(fls))
j = range(len(sigma))
k = range(len(fls))

coords = np.array(np.meshgrid(i, j, k)).T.reshape(-1,3)

Ftabloglisa = np.array(list(map(lambda args: fLlogtab(*args), coords))).reshape(len(fls),len(fls),len(sigma))

def maxlogvalsL(l):
    maximslog = np.log(np.max(Ftabloglisa[l]))
    return maximslog


maxposlog = range(len(Ftabloglisa))
maxlogvals = np.array(list(map(maxlogvalsL, maxposlog)))
maxlogL = maxlogvals

flogplotL = np.vstack((np.log(fls), maxlogL)).T


#%%
plt.figure(figsize=(6, 9))
plt.loglog(np.exp(np.log(fls)), np.exp(maxlogL), color = "aqua", linewidth=2.5)
plt.title("LogNS curve LISA")
plt.xlabel('f (Hz)')
plt.ylabel(r"$\Omega_{gw}$")
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()



plt.figure(figsize=(6, 9))
plt.loglog(freqvals, sigvals, color = "indigo", label = "Nominal", linewidth=2.5)
plt.loglog(np.exp(np.log(fls)), np.exp(maxlogL), color = "aqua", label = "LogNs", linewidth=2.5)
plt.title("Nominal and LogNs curve for LISA")
plt.xlabel('f (Hz)', fontsize = 16)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/LISAnomlognset.png', bbox_inches='tight')
plt.show()


#%%
def sigp(f):
    f0 = 1
    t1 = ((9.0*((f/f0)**(-30.0))) + (5.5e-6*((f/f0)**(-4.5e0))) +(0.28e-11*((f/f0)**3.2)))*(0.5-0.5*(np.tanh(0.06*((f/f0)-42.0))))
    t2 = ((0.01e-11*((f/f0)**(1.9))) + (20.0e-13*((f/f0)**(2.8))))*0.5*(np.tanh(0.06*((f/f0)-42.0)))
    t3 = 1.0-(0.475*np.exp(-(((f/f0)-25.0)**2.0)/50.0))
    t4 = 1.0-(5.0e-4*np.exp(-(((f/f0)-20.0)**2.0)/100.0))
    t5 = 1.0-(0.2*np.exp(-((((f/f0)-47.0)**2.0)**0.85)/100.0))
    t6 = 1.0-(0.12*np.exp(-((((f/f0)-50.0)**2.0)**0.7)/100.0))-(0.2*np.exp(-(((f/f0)-45.0)**2.0)/250.0))+(0.15*np.exp(-(((f/f0)-85.0)**2.0)/400.0))
    res = 0.88*(t1+t2)*t3*t4*t5*t6
    return res



def etnomonly(f):
    res = sigp(f)
    if res > 10**(-5):
        return
    return res

#%%
fvalsET = np.logspace(0, 3,itera)#frequency values
sigETvals = np.array(list(map(etnomonly, fvalsET)))


#ET Logns

def logn(f, fstar, sig):
    res = np.exp(-1/(2*10**sig)*(np.log(f/fstar))**2)
    return res


def AlogETmin(fstar, sig):
    integrand = lambda f, fstar, sig :(logn(f, fstar, sig)/sigp(f))**2
    I1 = quad(integrand, 1.6, 100, args=(fstar, sig))[0]
    I2 = quad(integrand, 100, ffmax, args = (fstar, sig))[0]
    res = snr5/np.sqrt(2*T*sum((I1,I2)))
    return res


lf = np.linspace(elminet, elmaxet, itera)
flf = 10**lf
sigma = np.linspace(-1, 0, 5)




vals = np.array(np.meshgrid(flf, sigma)).T.reshape(-1,2)
aminvals = np.array(list(map(lambda args: AlogETmin(*args), vals)))

atab = np.vstack((vals.T, aminvals)).T.reshape(-1,len(sigma), 3)

def ftab(i, j, k):
    res = atab[i, j, 2]*logn(flf[k], atab[i, j, 0], atab[i, j, 1])
    return res

i = range(len(flf))
j = range(len(sigma))
k = range(len(flf))

coords = np.array(np.meshgrid(i, j, k)).T.reshape(-1,3)

Ftabetlog = np.array(list(map(lambda args: ftab(*args), coords))).reshape(len(flf),len(flf),len(sigma))

def maxlogvals(l):
    maximslog = np.log(np.max(Ftabetlog[l]))
    return maximslog


maxposlog = range(len(Ftabetlog))
maxlog = np.array(list(map(maxlogvals, maxposlog)))
flogplot = np.vstack((np.log(flf), maxlog)).T

#%%
plt.figure(figsize=(6, 9))
plt.loglog(np.exp(flogplot[:,0]), np.exp(flogplot[:,1]), color = "aqua", linewidth=2.5)
plt.title("LogNS curve ET")
plt.xlabel('f (Hz)')
plt.ylabel(r"$\Omega_{gw}$")
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()



plt.figure(figsize=(6, 9))
plt.loglog(fvalsET, sigETvals, color = "indigo", label = "Nominal", linewidth=2.5)
plt.loglog(np.exp(flogplot[:,0]), np.exp(flogplot[:,1]), color = "aqua", label = "LogNs", linewidth=2.5)
plt.legend()
plt.title("Nominal and LogNs curve for ET")
plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
plt.ylim(10**(-13),10**(-5))
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/ETnomlognslisa.png', bbox_inches='tight')
plt.show()

#%%
######################
#Combined Log curves
#####################
def omegatog(f):
    if f <= 10**(-0.95):
        return Ohms(f)
    if f >= 1.6:
        return sigp(f)
    
def nomtog(f):
    if f <= 10**(-0.95):
        res = Ohms(f)
        if res > 1e-5:
            return
        return res
    if f >= 1:
        res = sigp(f)
        if res > 1e-5:
            return
        return res
 
fvalscomb = np.logspace(np.log10(ffmin), np.log10(ffmax),2000)
combine = np.array(list(map(omegatog, fvalscomb)))
nom = np.array(list(map(nomtog, fvalscomb)))
otog = np.vstack((fvalscomb, combine)).T

plt.figure(figsize=(6, 9))
plt.loglog(otog[:,0], nom, color = "indigo", linewidth=2.5)
plt.title("Nominal curves for ET and LISA")
plt.grid(True) 
plt.xlim(ffmin, ffmax) 
plt.xlabel('f (Hz)')
plt.ylabel(r"$\Omega_{gw}$")
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/ETLISAnom.png', bbox_inches='tight')
plt.show()



#%%
def logncomb(f, fstar, sig):
    res = np.exp(-1/(2*10**sig)*(np.log(f/fstar))**2)
    return res

def aminlogcomb(fstar, sig):
    integrand = lambda f, fstar, sig: (logncomb(f, fstar, sig)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-3), args=(fstar, sig))[0]
    I2 = quad(integrand, 10**(-3), 10**(-1), args=(fstar, sig))[0]
    I3 = quad(integrand, 10**(-1), 1e0, args=(fstar, sig))[0]
    I4 = quad(integrand, 1e0, 0.5e1, args=(fstar, sig))[0]
    I5 = quad(integrand, 0.5e1, ffmax, args=(fstar, sig))[0]
    integrand2 = lambda f, fstar, sig: (logncomb(f, fstar, sig)/sigp(f))**2
    I6 = quad(integrand2, ffmin, 1e-1, args=(fstar, sig))[0]
    I7 = quad(integrand2, 1e-1, ffmax, args=(fstar, sig))[0]
    res = snr5/np.sqrt(2*T*sum((I1, I2, I3, I4, I5, I6, I7)))
    return res 


lsc = np.linspace(elmin, elmax,itera)
flsc = 10**lsc
sigma = np.linspace(-1, 0, 5)


valslogcom = np.array(np.meshgrid(flsc, sigma)).T.reshape(-1,2)
aminvalscom = np.array(list(map(lambda args: aminlogcomb(*args), valslogcom)))

atablogcom = np.vstack((valslogcom.T, aminvalscom)).T.reshape(-1,len(sigma), 3)

def fLlogtabc(i, j, k):
    res = atablogcom[i, j, 2]*logncomb(flsc[k], atablogcom[i, j, 0], atablogcom[i, j, 1])
    return res

i = range(len(flsc))
j = range(len(sigma))
k = range(len(flsc))

coords = np.array(np.meshgrid(i, j, k)).T.reshape(-1,3)

Ftablogcomb = np.array(list(map(lambda args: fLlogtabc(*args), coords))).reshape(len(flsc),len(flsc),len(sigma))

def maxlogvalsc(l):
    maximslog = np.log(np.max(Ftablogcomb[l]))
    return maximslog


maxposlog = range(len(Ftablogcomb))
maxlogcom = np.array(list(map(maxlogvalsc, maxposlog)))
flogplotc = np.vstack((np.log(flsc), maxlogcom)).T
#%%


plt.figure(figsize=(6, 9))
plt.loglog(otog[:,0], nom, label = "Nominal", color = "indigo", linewidth=2.5)
plt.loglog(np.exp(flogplotc[:,0]), np.exp(flogplotc[:,1]), color = "aqua", label = " Combined LogNs", linewidth=2.5)
plt.loglog(np.exp(flogplot[:,0]), np.exp(flogplot[:,1]), ':',color = "black", label = "ET LogNs", linewidth=2.5)
plt.loglog(np.exp(np.log(fls)), np.exp(maxlogL),':', color = "teal", label = "LISA LogNs", linewidth=2.5)
plt.title(" Combined LogNS curve for LISA and ET",fontsize = 14)
plt.legend(loc = (0.02,0.8),fontsize = 14)
plt.xlim(ffmin, ffmax)
plt.xlabel('f (Hz)',fontsize = 14)
plt.ylabel(r"$\Omega_{gw}$",fontsize = 16)
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/CombineNomlogNswold.png', bbox_inches='tight')
plt.show()
