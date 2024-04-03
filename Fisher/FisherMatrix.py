import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
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
alphaA = -10
alphaB = -9

def f00(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: (10**(2*astar)*(f/f0)**(2*nt)*np.log(10)**2)/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=(f0, nt, astar))[0]
    return T*res

def f01(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: ((10**(astar))*(f/f0)**(2*nt)*np.log(10)*np.log(f/f0))/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=( f0, nt, astar))[0]
    return T*res

def f11(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: (10**(2*astar)*(f/f0)**(2*nt)*np.log(f/f0)**2)/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=( f0, nt, astar))[0]
    return T*res


def fisher(f0, nt, astar):
    res = np.array(((f00(f0, nt, astar), f01(f0, nt, astar)), 
                        (f01(f0, nt, astar), f11(f0, nt, astar))))
    return res

lisa = np.array(((0.1, 2/3, alphaA), (0.1, 0.01, alphaB)))
LISAfm = np.array(list(map(lambda args: fisher(*args), lisa)))

FMLA = LISAfm[0]
FMLB = LISAfm[1]
#%%
covmA = np.linalg.inv((FMLA))
covmB = np.linalg.inv((FMLB))

param_limitsA = {'r\alpha_*': (-60, 60), 'nt': (-60, 60)}
param_limitsB = {'r\alpha_*': (-60, 60)}#, 'nt': (0, 0.1)}

meansA = np.array((alphaA,2/3))
meansB = np.array((alphaB,0.01))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp)
names = [r'\alpha_*',r'nt']
labels =  [r'\alpha_*',r'nt']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Scenario A')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Scenario B')

#%%
g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=14
g.settings.legend_fontsize = 16
g.settings.axes_labelsize = 16
g.triangle_plot([samples], contour_colors = ['red'],
                filled=True, markers={r'\alpha_*': meansA[0],'nt': meansA[1]}, title_limit=1)
plt.suptitle(r'Fisher Analysis of LISA Scenario A', fontsize = 18)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERLISA_A.png')
#%%
g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=18
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.triangle_plot([samples2], contour_colors = ['orangered'], 
                #param_limits=param_limitsB,
                filled=True, markers={r'\alpha_*': meansB[0],'nt': meansB[1]}, title_limit=1)
plt.suptitle(r'Fisher Analysis of LISA Scenario B', fontsize = 18)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERLISA_B.png')
#%%
def sigp(f):
    res = 0.9 * ((3 * 30 * 10**(-1) * f**(-30) + 5.5 * 10**(-6) * f**(-4.5) + 
            0.7 * 10**(-11) * f**(2.8)) * (1/2 - 
            1/2 * np.tanh(0.04 * (f - 42))) + (1/2 * np.tanh(0.04 * (f - 42))) * 
            (0.4 * 10**(-11) * f**(1.4) + 7.9 * 10**(-13) * f**(2.98))) * (1 - 
            0.38 * np.exp(-(f - 25)**2/50))
    return res

def f00et(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: (10**(2*astar)*(f/f0)**(2*nt)*np.log(10)**2)/sigp(f)
    res = quad(integrand, 1, 445, args=(f0, nt, astar))[0]
    return T*res

def f01et(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: ((10**(astar))*(f/f0)**(2*nt)*np.log(10)*np.log(f/f0))/sigp(f)
    res = quad(integrand, 1, 445, args=( f0, nt, astar))[0]
    return T*res

def f11et(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: (10**(2*astar)*(f/f0)**(2*nt)*np.log(f/f0)**2)/sigp(f)
    res = quad(integrand, 1, 445, args=( f0, nt, astar))[0]
    return T*res


def fisheret(f0, nt, astar):
    res = np.array(((f00et(f0, nt, astar), f01et(f0, nt, astar)), 
                        (f01et(f0, nt, astar), f11et(f0, nt, astar))))
    return res

ET = np.array(((0.1, 2/3, alphaA), (0.1, 0.01, alphaB)))
ETfm = np.array(list(map(lambda args: fisheret(*args), ET)))

FMEA = ETfm[0]
FMEB = ETfm[0]
#%%
covmA = np.linalg.inv((FMEA))
covmB = np.linalg.inv((FMEB))


meansA = np.array((alphaA,2/3))
meansB = np.array((alphaB,0.01))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp)
names = [r'\alpha_*',r'nt']
labels =  [r'\alpha_*',r'nt']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Scenario A')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Scenario B')

#param_limitsA = {r'\Omega_*': (0, 1e-10), 'nt': (0, 1.2)}

#%%
g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=14
g.settings.legend_fontsize = 16
g.settings.axes_labelsize = 16
g.triangle_plot([samples], contour_colors = ['darkblue'], #param_limits=param_limitsA,
                filled=True, markers={r'\alpha_*': meansA[0],'nt': meansA[1]}, title_limit=1)
plt.suptitle(r'Fisher Analysis of ET Scenario A', fontsize = 18)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERET_A.png')

g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=14
g.settings.legend_fontsize = 16
g.settings.axes_labelsize = 16
g.triangle_plot([samples2], contour_colors = ['blue'], 
                filled=True, markers={r'\alpha_*': meansB[0],'nt': meansB[1]}, title_limit=1)
plt.suptitle(r'Fisher Analysis of ET Scenario B', fontsize = 18)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERET_B.png')
#%%
# g = plots.get_subplot_plotter(subplot_size=5)
# g.settings.axes_fontsize=14
# g.settings.legend_fontsize = 16
# g.settings.axes_labelsize = 16
# g.triangle_plot([samples2], contour_colors = ['green'], 
#                 filled=True, markers={r'\alpha_*': meansB[0],'nt': meansB[1]}, title_limit=1)
# plt.suptitle(r'Fisher Analysis of ET Scenario B')
#%%
#all together now
FMA = FMLA + FMEA
FMB = FMLB + FMEB

covmA = np.linalg.inv((FMA))
covmB = np.linalg.inv((FMB))


meansA = np.array((alphaA,2/3))
meansB = np.array((alphaB,0.01))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp)
names = [r'\alpha_*',r'nt']
labels =  [r'\alpha_*',r'nt']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Scenario A')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Scenario B')

#%%
g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=14
g.settings.legend_fontsize = 16
g.settings.axes_labelsize = 16
g.triangle_plot([samples], contour_colors = ['indigo'],
                filled=True, markers={r'\alpha_*': meansA[0],'nt': meansA[1]}, title_limit=1)
plt.suptitle(r'Fisher Analysis of LISA + ET Scenario A', fontsize = 18)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERcomb_A.png')

g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=14
g.settings.legend_fontsize = 16
g.settings.axes_labelsize = 16
g.triangle_plot([samples2], contour_colors = ['darkviolet'], 
                filled=True, markers={r'\alpha_*': meansB[0],'nt': meansB[1]}, title_limit=1)
plt.suptitle(r'Fisher Analysis of LISA + ET Scenario B', fontsize = 18)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERcomb_B.png')


















 
