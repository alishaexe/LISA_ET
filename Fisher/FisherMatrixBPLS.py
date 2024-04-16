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


####LOW FREQUENCY APPROXIMATION
#Now make the low frequency approximation
#this is equation 63 in the paper

def sigI(f):#Sigma_I
    sig = np.sqrt(2)*20/3 * (SI(f)/(2*pi*f)**4 + SII)*(1+(f/(4*fLisa/3))**2)
    return sig

def SigmaLisaApprox(f):#Sigma_Ohm approx
    const = ((4*pi**2/(3*H0**2)))
    res = const * f**3 *sigI(f)
    return res



L = 25/3
fLisa = 1/(2*pi*L)
size = 20
#case 1
a1 = -6
n1 = 1
n2 = -1
fstar = 0.45

#case 2
a2 = -6.5
nom1 = 0.8
nom2 = -0.8
fbreak = 0.05
#%%

def f00(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**(2*astar)*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(10)**2)/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=(f0, n1, n2, astar))[0]
    return T*res

def f01(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*(10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - 10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)*np.log(10))/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=(f0, n1, n2, astar))[0]
    return T*res

def f02(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**(2*astar)*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(10)*np.log(f**10/f0**10 + 1)/10)/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=(f0, n1, n2, astar))[0]
    return T*res

def f11(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: ((10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - 10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)**2)/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=(f0, n1, n2, astar))[0]
    return T*res

def f12(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*(10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - 10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)*np.log(f**10/f0**10 + 1)/10)/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=(f0, n1, n2, astar))[0]
    return T*res

def f22(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar:  (10**(2*astar)*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(f**10/f0**10 + 1)**2/100)/SigmaLisaApprox(f)
    res = quad(integrand, 1e-5, 1e-1, args=(f0, n1, n2, astar))[0]
    return T*res

def fisher(f0, n1, n2, astar):
    res = np.array(((f00(f0, n1, n2, astar), f01(f0, n1, n2, astar), f02(f0, n1, n2, astar)), 
                    (f01(f0, n1, n2, astar), f11(f0, n1, n2, astar), f12(f0, n1, n2, astar)),
                    (f02(f0, n1, n2, astar), f12(f0, n1, n2, astar), f22(f0, n1, n2, astar))))
    return res

lisa = [(fstar, n1, n2, a1), (fbreak, nom1, nom2, a2)]
LISAfm = np.array(list(map(lambda args: fisher(*args), lisa)))
#LISAfm = np.array(list(lambda args: fisher(*args)))

FMLA = LISAfm[0]
FMLB = LISAfm[1]
#%%
covmA = np.linalg.inv((FMLA))
covmB = np.linalg.inv((FMLB))



meansA = np.array((a1, n1, n2))
meansB = np.array((a2, nom1, nom2))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp, tol=1e-6)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp, tol=1e-6)
names = [r'\alpha_*',r'n1', r'n2']
labels =  [r'\alpha_*',r'n1', r'n2']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Case 1')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Case 2')


#%%
g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.triangle_plot([samples], contour_colors = ['red'], 
                filled=True, markers={r'\alpha_*': meansA[0],'n1': meansA[1], 'n2':meansA[2]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of LISA BPL case 1: n1 = {nom1}, n2 = {nom2} $f_\star$ = {fbreak}'.format(nom1=n1, nom2=n2, fbreak=fstar), fontsize=size)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERBPL_LISA1.png')
#%%
g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.triangle_plot([samples2], contour_colors = ['red'], 
                filled=True, markers={r'\alpha_*': meansB[0],'n1': meansB[1], 'n2':meansB[2]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of LISA BPL: n1 = {nom1}, n2 = {nom2} $f_\star$ = {fbreak}'.format(nom1=nom1, nom2=nom2, fbreak=fbreak), fontsize=size)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERBPL_LISA2.png')

#%%
def sigp(f):
    res = 0.9 * ((3 * 30 * 10**(-1) * f**(-30) + 5.5 * 10**(-6) * f**(-4.5) + 
            0.7 * 10**(-11) * f**(2.8)) * (1/2 - 
            1/2 * np.tanh(0.04 * (f - 42))) + (1/2 * np.tanh(0.04 * (f - 42))) * 
            (0.4 * 10**(-11) * f**(1.4) + 7.9 * 10**(-13) * f**(2.98))) * (1 - 
            0.38 * np.exp(-(f - 25)**2/50))
    return res

def f00et(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**(2*astar)*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(10)**2)/sigp(f)
    res = quad(integrand, 1, 445, args=(f0, n1, n2, astar))[0]
    return T*res

def f01et(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*(10**astar*(f/f0)**n1*
                                                    (f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - 10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*
                                                    np.log(f**10/f0**10 + 1)/10)*np.log(10))/sigp(f)
    res = quad(integrand, 1, 445, args=(f0, n1, n2, astar))[0]
    return T*res

def f02et(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**(2*astar)*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(10)*np.log(f**10/f0**10 + 1)/10)/sigp(f)
    res1 = quad(integrand, 1, 100, args=(f0, n1, n2, astar))[0]
    res2 = quad(integrand, 100, 250, args=(f0, n1, n2, astar))[0]
    res3 = quad(integrand, 250, 445, args=(f0, n1, n2, astar))[0]
    return T*sum((res1,res2,res3))

def f11et(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: ((10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - 10**astar*(f/f0)**n1*
                                               (f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)**2)/sigp(f)
    res = quad(integrand, 1, 445, args=(f0, n1, n2, astar))[0]
    return T*res

def f12et(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*(10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*
                                                np.log(f/f0) - 10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)*np.log(f**10/f0**10 + 1)/10)/sigp(f)
    res = quad(integrand, 1, 445, args=(f0, n1, n2, astar))[0]
    return T*res

def f22et(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar:  (10**(2*astar)*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(f**10/f0**10 + 1)**2/100)/sigp(f)
    res = quad(integrand, 1, 445, args=(f0, n1, n2, astar))[0]
    return T*res

def fisheret(f0, n1, n2, astar):
    res = np.array(((f00et(f0, n1, n2, astar), f01et(f0, n1, n2, astar), f02et(f0, n1, n2, astar)), 
                    (f01et(f0, n1, n2, astar), f11et(f0, n1, n2, astar), f12et(f0, n1, n2, astar)),
                    (f02et(f0, n1, n2, astar), f12et(f0, n1, n2, astar), f22et(f0, n1, n2, astar))))
    return res

ET = [(fstar, n1, n2, a1), (fbreak, nom1, nom2, a2)]
ETfm = np.array(list(map(lambda args: fisheret(*args), ET)))

FMEA = ETfm[0]
FMEB = ETfm[1]
#%%
covmA = np.linalg.inv((FMEA))
covmB = np.linalg.inv((FMEB))


meansA = np.array((a1, n1, n2))
meansB = np.array((a2, nom1, nom2))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp, tol=1e-6)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp, tol=1e-6)
names = [r'\alpha_*',r'n1', r'n2']
labels =  [r'\alpha_*',r'n1', r'n2']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Case 1')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Case 2')

#%%
g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.triangle_plot([samples], contour_colors = ['blue'], 
                filled=True, markers={r'\alpha_*': meansA[0],'n1': meansA[1], 'n2':meansA[2]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of ET BPL case 1: n1 = {nom1}, n2 = {nom2} $f_\star$ = {fbreak}'.format(nom1=n1, nom2=n2, fbreak=fstar), fontsize=size)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERBPL_ET1.png')
#%%
g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.triangle_plot([samples2], contour_colors = ['blue'], 
                filled=True, markers={r'\alpha_*': meansB[0],'n1': meansB[1], 'n2': meansB[2]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of ET BPL: n1 = {nom1}, n2 = {nom2} $f_\star$ = {fbreak}'.format(nom1=nom1, nom2=nom2, fbreak=fbreak), fontsize=size)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERBPL_ET2.png')

#%%
#all together now

def f00(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**(2*astar)*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(10)**2)/SigmaLisaApprox(f)
    res = quad(integrand, ffmin, ffmax, args=(f0, n1, n2, astar))[0]
    return T*res

def f01(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*(10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - 10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)*np.log(10))/SigmaLisaApprox(f)
    res = quad(integrand, ffmin, ffmax, args=(f0, n1, n2, astar))[0]
    return T*res

def f02(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**(2*astar)*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(10)*np.log(f**10/f0**10 + 1)/10)/SigmaLisaApprox(f)
    res = quad(integrand, ffmin, ffmax, args=(f0, n1, n2, astar))[0]
    return T*res

def f11(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: ((10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - 10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)**2)/SigmaLisaApprox(f)
    res = quad(integrand, ffmin, ffmax, args=(f0, n1, n2, astar))[0]
    return T*res

def f12(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*(10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - 10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)*np.log(f**10/f0**10 + 1)/10)/SigmaLisaApprox(f)
    res = quad(integrand, ffmin, ffmax, args=(f0, n1, n2, astar))[0]
    return T*res

def f22(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar:  (10**(2*astar)*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(f**10/f0**10 + 1)**2/100)/SigmaLisaApprox(f)
    res = quad(integrand, ffmin, ffmax, args=(f0, n1, n2, astar))[0]
    return T*res

def fisher(f0, n1, n2, astar):
    res = np.array(((f00(f0, n1, n2, astar), f01(f0, n1, n2, astar), f02(f0, n1, n2, astar)), 
                    (f01(f0, n1, n2, astar), f11(f0, n1, n2, astar), f12(f0, n1, n2, astar)),
                    (f02(f0, n1, n2, astar), f12(f0, n1, n2, astar), f22(f0, n1, n2, astar))))
    return res

lisaC = [(fstar, n1, n2, a1), (fbreak, nom1, nom2, a2)]
LISAfmC = np.array(list(map(lambda args: fisher(*args), lisaC)))
#LISAfm = np.array(list(lambda args: fisher(*args)))

FMLAC = LISAfmC[0]
FMLBC = LISAfmC[1]


def f00et(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**(2*astar)*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(10)**2)/sigp(f)
    res = quad(integrand, ffmin, ffmax, args=(f0, n1, n2, astar))[0]
    return T*res

def f01et(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*(10**astar*(f/f0)**n1*
                                                    (f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - 10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*
                                                    np.log(f**10/f0**10 + 1)/10)*np.log(10))/sigp(f)
    res = quad(integrand, ffmin, ffmax, args=(f0, n1, n2, astar))[0]
    return T*res

def f02et(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**(2*astar)*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(10)*np.log(f**10/f0**10 + 1)/10)/sigp(f)
    res1 = quad(integrand, ffmin, 1e-3, args=(f0, n1, n2, astar))[0]
    res2 = quad(integrand, 1e-3, 1e0, args=(f0, n1, n2, astar))[0]
    res3 = quad(integrand, 1e0, ffmax, args=(f0, n1, n2, astar))[0]
    return T*sum((res1,res2,res3))

def f11et(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: ((10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f/f0) - 10**astar*(f/f0)**n1*
                                               (f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)**2)/sigp(f)
    res = quad(integrand, ffmin, ffmax, args=(f0, n1, n2, astar))[0]
    return T*res

def f12et(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar: (10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*(10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*
                                                np.log(f/f0) - 10**astar*(f/f0)**n1*(f**10/f0**10 + 1)**(-n1/10 + n2/10)*np.log(f**10/f0**10 + 1)/10)*np.log(f**10/f0**10 + 1)/10)/sigp(f)
    res = quad(integrand, ffmin, ffmax, args=(f0, n1, n2, astar))[0]
    return T*res

def f22et(f0, n1, n2, astar):
    integrand = lambda f, f0, n1, n2, astar:  (10**(2*astar)*(f/f0)**(2*n1)*(f**10/f0**10 + 1)**(-n1/5 + n2/5)*np.log(f**10/f0**10 + 1)**2/100)/sigp(f)
    res = quad(integrand, ffmin, ffmax, args=(f0, n1, n2, astar))[0]
    return T*res

def fisheret(f0, n1, n2, astar):
    res = np.array(((f00et(f0, n1, n2, astar), f01et(f0, n1, n2, astar), f02et(f0, n1, n2, astar)), 
                    (f01et(f0, n1, n2, astar), f11et(f0, n1, n2, astar), f12et(f0, n1, n2, astar)),
                    (f02et(f0, n1, n2, astar), f12et(f0, n1, n2, astar), f22et(f0, n1, n2, astar))))
    return res

ETC = [(fstar, n1, n2, a1), (fbreak, nom1, nom2, a2)]
ETfmC = np.array(list(map(lambda args: fisheret(*args), ETC)))

FMEAC = ETfmC[0]
FMEBC = ETfmC[1]


FMAC = FMLAC + FMEAC
FMBC = FMLBC + FMEBC

covmA = np.linalg.inv((FMAC))
covmB = np.linalg.inv((FMBC))


meansA = np.array((a1, n1, n2))
meansB = np.array((a2, nom1, nom2))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp, tol=1e-6)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp, tol=1e-6)
names = [r'\alpha_*',r'n1', r'n2']
labels =  [r'\alpha_*',r'n1', r'n2']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Case 1')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Case 2')


g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.triangle_plot([samples], contour_colors = ['indigo'], 
                filled=True, markers={r'\alpha_*': meansA[0],'n1': meansA[1], 'n2':meansA[2]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of LISA + ET BPL: n1 = {nom1}, n2 = {nom2} $f_\star$ = {fbreak}'.format(nom1=n1, nom2=n2, fbreak=fstar), fontsize=size)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERBPL_COMB1.png')

g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=20
g.settings.legend_fontsize = 20
g.settings.axes_labelsize = 20
g.triangle_plot([samples2], contour_colors = ['indigo'], 
                filled=True, markers={r'\alpha_*': meansB[0],'n1': meansB[1], 'n2':meansB[2]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of LISA + ET BPL: n1 = {nom1}, n2 = {nom2} $f_\star$ = {fbreak}'.format(nom1=nom1, nom2=nom2, fbreak=fbreak), fontsize=size)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERBPL_COMB2.png')




#%%


# arr = covmA #you will need to run the cell with the covariance matrix you want
#               #you want to check first since they all have the same name

# #Check if the matrix is symmetric
# is_symmetric = np.allclose(arr, arr.T)
# print("Is symmetric:", is_symmetric)

# #Check if all principal minors are positive
# minors_positive = all(np.linalg.det(arr[:i, :i]) > 0 for i in range(1, arr.shape[0] + 1))
# print("All principal minors are positive:", minors_positive)













 
