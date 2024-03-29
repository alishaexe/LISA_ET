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

def f00(f0, n1, n2, omegstar):
    integrand = lambda f, f0, n1, n2: (f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=(f0, n1, n2))[0]
    return T*res

def f01(f0, n1, n2, omegstar):
    integrand = lambda f, f0, n1, n2, omegstar: ((f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*(omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10))/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=(f0, n1, n2, omegstar))[0]
    return T*res

def f02(f0, n1, n2, omegstar):
    integrand = lambda f, f0, n1, n2, omegstar: (omegstar*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(f**10/f0**10 + 1)/10)/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=(f0, n1, n2, omegstar))[0]
    return T*res

def f11(f0, n1, n2, omegstar):
    integrand = lambda f, f0, n1, n2, omegstar: ((omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)**2)/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=(f0, n1, n2, omegstar))[0]
    return T*res

def f12(f0, n1, n2, omegstar):
    integrand = lambda f, f0, n1, n2, omegstar: (omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*(omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)*np.log(f**10/f0**10 + 1)/10)/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=(f0, n1, n2, omegstar))[0]
    return T*res

def f22(f0, n1, n2, omegstar):
    integrand = lambda f, f0, n1, n2, omegstar:  (omegstar**2*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(f**10/f0**10 + 1)**2/100)/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=(f0, n1, n2, omegstar))[0]
    return T*res

def fisher(f0, n1, n2, omegstar):
    res = np.array(((f00(f0, n1, n2, omegstar), f01(f0, n1, n2, omegstar), f02(f0, n1, n2, omegstar)), 
                    (f01(f0, n1, n2, omegstar), f11(f0, n1, n2, omegstar), f12(f0, n1, n2, omegstar)),
                    (f02(f0, n1, n2, omegstar), f12(f0, n1, n2, omegstar), f22(f0, n1, n2, omegstar))))
    return res

lisa = np.array(((0.1, 2/3, 0.01, 1e-10), (0.1, 0.01, 2/3, 1e-9)))
LISAfm = np.array(list(map(lambda args: fisher(*args), lisa)))

FMLA = LISAfm[0]
FMLB = LISAfm[1]
#%%
covmA = np.linalg.inv((FMLA))
covmB = np.linalg.inv((FMLB))


meansA = np.array((10**(-10),2/3, 0.01))
meansB = np.array((10**(-9),0.01, 2/3))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp)
names = [r'\Omega_*',r'nt', r'n2']
labels =  [r'\Omega_*',r'n1', r'n2']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Scenario A')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Scenario B')


g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=18
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.triangle_plot([samples], contour_colors = ['teal'],
                filled=True, markers={r'\Omega_*': meansA[0],'n1': meansA[1], 'n2':meansA[2]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of LISA Scenario A : 2/3, 0.01', fontsize = 18)

g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=18
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.triangle_plot([samples2], conour_colors = ['green'], 
                filled=True, markers={r'\Omega_*': meansB[0],'n1': meansB[1], 'n2':meansB[2]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of LISA Scenario B: 0.01, 2/3', fontsize = 18)


#%%
def sigp(f):
    res = 1.3*((3*30*10**(-1)*f**(-30)+5*10**(-6)*f**(-4.5)+0.6*10**(-11)*f**(2.8))
               *(1/2-1/2*np.tanh(0.1*(f-42)))+(1/2*np.tanh(0.1*(f-42)))*(2*10**(-11)*f**2.25 
                                                                         +10**(-13)*f**3))
    return res

def f00et(f0, n1, n2, omegstar):
    integrand = lambda f, f0, n1, n2: (f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)/sigp(f)
    res = quad(integrand, 1.6, 445, args=(f0, n1, n2))[0]
    return T*res

def f01et(f0, n1, n2, omegstar):
    integrand = lambda f, f0, n1, n2, omegstar: ((f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*(omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10))/sigp(f)
    res = quad(integrand, 1.6, 445, args=(f0, n1, n2, omegstar))[0]
    return T*res

def f02et(f0, n1, n2, omegstar):
    integrand = lambda f, f0, n1, n2, omegstar: (omegstar*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(f**10/f0**10 + 1)/10)/sigp(f)
    res = quad(integrand, 1.6, 445, args=(f0, n1, n2, omegstar))[0]
    return T*res

def f11et(f0, n1, n2, omegstar):
    integrand = lambda f, f0, n1, n2, omegstar: ((omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)**2)/sigp(f)
    res = quad(integrand, 1.6, 445, args=(f0, n1, n2, omegstar))[0]
    return T*res

def f12et(f0, n1, n2, omegstar):
    integrand = lambda f, f0, n1, n2, omegstar: (omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*(omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - omegstar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)*np.log(f**10/f0**10 + 1)/10)/sigp(f)
    res = quad(integrand, 1.6, 445, args=(f0, n1, n2, omegstar))[0]
    return T*res

def f22et(f0, n1, n2, omegstar):
    integrand = lambda f, f0, n1, n2, omegstar:  (omegstar**2*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(f**10/f0**10 + 1)**2/100)/sigp(f)
    res = quad(integrand, 1.6, 445, args=(f0, n1, n2, omegstar))[0]
    return T*res

def fisheret(f0, n1, n2, omegstar):
    res = np.array(((f00et(f0, n1, n2, omegstar), f01et(f0, n1, n2, omegstar), f02et(f0, n1, n2, omegstar)), 
                    (f01et(f0, n1, n2, omegstar), f11et(f0, n1, n2, omegstar), f12et(f0, n1, n2, omegstar)),
                    (f02et(f0, n1, n2, omegstar), f12et(f0, n1, n2, omegstar), f22et(f0, n1, n2, omegstar))))
    return res

ET = np.array(((0.1, 2/3, 0.01,1e-10), (0.1, 0.01, 2/3, 1e-9)))
ETfm = np.array(list(map(lambda args: fisheret(*args), ET)))

FMEA = ETfm[0]
FMEB = ETfm[0]
#%%
covmA = np.linalg.inv((FMEA))
covmB = np.linalg.inv((FMEB))


meansA = np.array((10**(-10),2/3, 0.01))
meansB = np.array((10**(-9),0.01,2/3))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp)
names = [r'\Omega_*',r'n1', r'n2']
labels =  [r'\Omega_*',r'n1', r'n2']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Scenario A')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Scenario B')


g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=18
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.triangle_plot([samples], contour_colors = ['teal'],
                filled=True, markers={r'\Omega_*': meansA[0],'n1': meansA[1], 'n2':meansA[2]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of ET Scenario A: 2/3, 0.01')

g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=18
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.triangle_plot([samples2], contour_colors = ['green'], 
                filled=True, markers={r'\Omega_*': meansB[0],'n1': meansB[1], 'n2': meansB[2]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of ET Scenario B: 0.01, 2/3')

#%%
#all together now
FMA = FMLA + FMEA
FMB = FMLB + FMEB

covmA = np.linalg.inv((FMA))
covmB = np.linalg.inv((FMB))


meansA = np.array((10**(-10),2/3, 0.01))
meansB = np.array((10**(-9),0.01, 2/3))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp)
names = [r'\Omega_*',r'n1', r'n2']
labels =  [r'\Omega_*',r'n1', r'n2']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Scenario A')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Scenario B')


g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=18
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.triangle_plot([samples], contour_colors = ['teal'],
                filled=True, markers={r'\Omega_*': meansA[0],'n1': meansA[1], 'n2': meansA[2]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of LISA + ET Scenario A: 2/3, 0.01')

g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=18
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.triangle_plot([samples2], contour_colors = ['green'], 
                filled=True, markers={r'\Omega_*': meansB[0],'n1': meansB[1], 'n2':meansB[2]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of LISA + ET Scenario B: 0.01, 2/3')



















 