import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import getdist
from getdist import plots, MCSamples
from scipy.interpolate import UnivariateSpline
import time
#%%
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


#benchmarks LISA case 1
om1 = 7e-12
nt1 = -0.1
fs1 = 1e-7


#%%
P = 12
A = 3

def P_acc(f):
    res = A**2 *(1e-15)**2 * (1+(0.4e-3 / f)**2)*(1+(f/8e-3)**4)*(2*pi*f)**(-4)*(2*pi*f/c)**2
    return res

def P_ims(f):
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


#%%


def f00(f0, nt, om):
    integrand = lambda f, f0, nt, om: ((f/f0)**(2*nt))/Ohms(f)**2
    res = quad(integrand, ffmin, ffmax, args=(f0, nt, om))[0]
    return 2*T*res

def f01(f0, nt, om):
    integrand = lambda f, f0, nt, om: (om*(f/f0)**(2*nt)*np.log(f/f0))/Ohms(f)**2
    res = quad(integrand, ffmin, ffmax, args=( f0, nt, om))[0]
    return 2*T*res

def f11(f0, nt, om):
    integrand = lambda f, f0, nt, om: (om**2*(f/f0)**(2*nt)*np.log(f/f0)**2)/Ohms(f)**2
    res = quad(integrand, ffmin, ffmax, args=( f0, nt, om))[0]
    return 2*T*res


def fisher(f0, nt, om):
    res = np.array(((f00(f0, nt, om), f01(f0, nt, om)), 
                        (f01(f0, nt, om), f11(f0, nt, om))))
    return res

lisa = np.array((fs1, nt1, om1))
LISAfm = fisher(fs1, nt1, om1)#np.array(list(map(lambda args: fisher(*args), lisa)))

FMLA = LISAfm
covmA = np.linalg.inv((FMLA))



meansA = np.array((om1,nt1))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp)
names = [r'\Omega_*',r'nt']
labels =  [r'\Omega_*',r'nt']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Scenario A')


g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=18
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.triangle_plot([samples], contour_colors = ['red'],
                filled=True, markers={r'\Omega_*': meansA[0],'nt': meansA[1]}, title_limit=1)
plt.text(-0.1,2, r'$\Omega_* = {om}$'.format(om = om1), ha='center', size='x-large')
plt.text(-0.1,1.9, r'$nt = {nt}$'.format(nt = nt1), ha='center', size='x-large')
plt.text(-0.1,1.8, r'$f_* = {fs}$'.format(fs = fs1), ha='center', size='x-large')
# plt.suptitle(r'Fisher Analysis of LISA Scenario A', fontsize = 18)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERLISA_A.png')

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

def f00et(f0, nt, om):
    integrand = lambda f, f0, nt, om: (f/f0)**(2*nt)/sigp(f)**2
    res = quad(integrand, ffmin, ffmax, args=(f0, nt, om))[0]
    return 2*T*res

def f01et(f0, nt, om):
    integrand = lambda f, f0, nt, om: (om*(f/f0)**(2*nt)*np.log(f/f0))/sigp(f)**2
    res = quad(integrand, ffmin, ffmax, args=( f0, nt, om))[0]
    return 2*T*res

def f11et(f0, nt, om):
    integrand = lambda f, f0, nt, om: (om**2*(f/f0)**(2*nt)*np.log(f/f0)**2)/sigp(f)**2
    res = quad(integrand, ffmin, ffmax, args=( f0, nt, om))[0]
    return 2*T*res


def fisheret(f0, nt, om):
    res = np.array(((f00et(f0, nt, om), f01et(f0, nt, om)), 
                        (f01et(f0, nt, om), f11et(f0, nt, om))))
    return res

ET = np.array((fs1, nt1, om1))
ETfm = fisheret(fs1, nt1, om1)#np.array(list(map(lambda args: fisheret(*args), ET)))

FMEA = ETfm
covmA = np.linalg.inv((FMEA))


meansA = np.array((om1,nt1))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp)
names = [r'\Omega_*',r'nt']
labels =  [r'\Omega_*',r'nt']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Scenario A')


g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=14
g.settings.legend_fontsize = 16
g.settings.axes_labelsize = 16
g.triangle_plot([samples], contour_colors = ['darkblue'], #param_limits=param_limitsA,
                filled=True, markers={r'\Omega_*': meansA[0],'nt': meansA[1]}, title_limit=1)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERET_A.png')


#%%
########################
#Combining
########################

#all together now
FMA = FMLA + FMEA

covmA = np.linalg.inv((FMA))


meansA = np.array((om1,nt1))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp)
names = [r'\Omega_*',r'nt']
labels =  [r'\Omega_*',r'nt']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Scenario A')


g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=18
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.triangle_plot([samples], contour_colors = ['indigo'],
                filled=True, markers={r'\Omega_*': meansA[0],'nt': meansA[1]}, title_limit=1)
plt.suptitle(r'Fisher Analysis of LISA + ET Scenario A', fontsize = 18)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERcomb_A.png')










