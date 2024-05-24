import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import time
#%%
#Now we want to start looking at what we get for broken power laws.
#Let's start by defining our analytical values again

#Getting the data file
Rtab = np.array(np.loadtxt('Rtab.dat'))
Etab = np.array(np.loadtxt('ciao4.dat'))
#These separate the three columns of the given
#data array. The first column is frequency,
#followed by r1, and then r2.

f = Rtab[:,0]#Measured in Hz
r1 = Rtab[:,1]#auto correlation response
r2 = Rtab[:,2]#cross-correlation response.

Rae = r1-r2
#Now going to define constants
yr = 365*24*60*60 #in seconds
H0 = 100*0.67*10**(3)/(3.086*10**(22)) #1/seconds
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
elmaxL = (np.log10(10**(-1)))
elminet = np.log10(1.6)
elmaxet = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera
P = 12
A = 3
alpha = -11.352

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
def pls(f):
    nt = -0.1
    a = -12.4
    f0 = 1e-7
    res = 10**a * (f/f0)**(nt)
    return res

freqvals = np.logspace(elminL, elmaxL, itera)   
sigvals = np.array(list(map(Ohms, freqvals)))

pl = np.array(list(map(pls, freqvals)))
plt.figure(figsize=(6, 9)) 
plt.loglog(freqvals, sigvals, color = "indigo", label = "Approximate", linewidth=2.5)
plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]), color = "orangered", linewidth=2.5)
plt.plot(freqvals, pl, color = "black")
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.grid(True)
plt.legend(fontsize = 16)
plt.xlabel("f (Hz)", fontsize = 16)
plt.title("Nominal Sensitivity curve of LISA", fontsize = 16)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/LISAnoms.png', bbox_inches='tight')
plt.show()

#%%

def sigp(f):
    f0 = 1
    t1 = (1-0.475*np.exp(-(f/f0-25)**2/50))
    t2 = (1-5e-4*np.exp(-(f/f0-20)**2/100))
    t3 = (1-0.2*np.exp(-((f/f0-47)**2)**0.85/100))
    t4 = (1-0.1*np.exp(-((f/f0-50)**2)**0.7/100)-0.2*np.exp(-(f/f0-45)**2/250)+0.15*np.exp(-(f/f0-85)**2/400))
    res = 0.88*((9*(f/f0)**(-30)+5.5e-6 *(f/f0)**(-4.5)+0.28e-11 * (f/f0)**3.2)*(1/2 - 1/2*np.tanh(0.06*(f/f0-42)))
                +(1/2*np.tanh(0.06*(f/f0-42)))*(0.01e-11*(f/f0)**1.9 + 20e-13 *(f/f0)**2.8))*t1*t2*t3*t4*(0.67)**2
    if res > 1.2e-6:
        return
    return res


def pls(f):
    nt = -0.1
    a = -12
    f0 = 1e-7
    res = 10**a * (f/f0)**(nt)
    return res

fvalsET = np.logspace(np.log10(1), np.log10(445),itera)#frequency values
sigETvals = np.array(list(map(sigp, fvalsET)))
plet = np.array(list(map(pls, fvalsET)))

plt.figure(figsize=(6, 9)) 
plt.title("Nominal Sensitivity curve of ET", fontsize = 16)
plt.loglog(fvalsET, sigETvals, '-',label = "Approximate", linewidth = 2.5,color = "indigo" )
plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]), color = "orangered", linewidth=2.5)
plt.plot(fvalsET, plet, color = "black")
plt.legend(fontsize = 16)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.xlabel("f (Hz)", fontsize = 16)
#plt.ylim(10**(-9), 10**(-5))
plt.yscale('log')
plt.xscale('log')
# plt.xlim(10**0, 400)
plt.grid(True)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/ETnoms.png', bbox_inches='tight')
plt.show()

#%%


plt.figure(figsize=(6, 9))
plt.loglog(otog[:,0], nom , color = "indigo", label = "Nominal", linewidth=2.5)
plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]), color = "orangered", label = "Combined PLS", linewidth=2.5)
plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]), ':',color = "teal", label = 'LISA PLS', linewidth=2.5)
plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]), ':',color = "black", label = 'ET PLS', linewidth=2.5)
plt.title("Nominal and PLS curves ", fontsize = 16)
plt.legend(loc = (1.05,0.5), fontsize = 14)
plt.grid(True) 
plt.xlim(ffmin, ffmax) 
plt.xlabel(r'$f$ (Hz)', fontsize = 16)
plt.ylabel(r'$\Omega_{gw}$', fontsize = 16)
plt.show()





























