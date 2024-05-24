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


#LISA case 1
a1 = -12.4
nt1 = -0.1
fs1 = 1e-3

#LISA case 2
a2 = -20
nt2 = -0.1
fs2 = 1e-3

#ET case 3 
a3 = -12
nt3 = -0.1
fs3 = 1e-3

#ET case 4
a4 = -12
nt4 = -0.1
fs4 = 1e-3

#Combined case 5
a5 = -6
nt5 = -0.1
fs5 = 1e-3

#Combined case 6
a6 = -6
nt6 = -0.1
fs6 = 1e-3

#%%
# #%%
P = 12
A = 3
# alpha = -11.352

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



# #%%


def f00(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: (10**(2*astar)*(f/f0)**(2*nt)*np.log(10)**2)/Ohms(f)**2
    res = quad(integrand, 1e-5, 1e-1, args=(f0, nt, astar))[0]
    return T*res

def f01(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: ((10**(astar))*(f/f0)**(2*nt)*np.log(10)*np.log(f/f0))/Ohms(f)**2
    res = quad(integrand, 1e-5, 1e-1, args=( f0, nt, astar))[0]
    return T*res

def f11(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: (10**(2*astar)*(f/f0)**(2*nt)*np.log(f/f0)**2)/Ohms(f)**2
    res = quad(integrand, 1e-5, 1e-1, args=( f0, nt, astar))[0]
    return T*res


def fisher(f0, nt, astar):
    res = np.array(((f00(f0, nt, astar), f01(f0, nt, astar)), 
                        (f01(f0, nt, astar), f11(f0, nt, astar))))
    return res

lisa = np.array(((fs1, nt1, a1), (fs2, nt2, a2)))
LISAfm = np.array(list(map(lambda args: fisher(*args), lisa)))

FMLA = LISAfm[0]
FMLB = LISAfm[1]
# #%%
covmA = np.linalg.inv((FMLA))
covmB = np.linalg.inv((FMLB))


meansA = np.array((a1,nt1))
meansB = np.array((a2,nt2))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp)
names = [r'\alpha_*',r'nt']
labels =  [r'\alpha_*',r'nt']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Scenario A')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Scenario B')

# #%%
g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=18
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.triangle_plot([samples], contour_colors = ['red'],
                filled=True, markers={r'\alpha_*': meansA[0],'nt': meansA[1]}, title_limit=1)
# plt.suptitle(r'Fisher Analysis of LISA Scenario A', fontsize = 18)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERLISA_A.png')

g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=18
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.triangle_plot([samples2], contour_colors = ['orangered'], 
                filled=True, markers={r'\alpha_*': meansB[0],'nt': meansB[1]}, title_limit=1)
# plt.suptitle(r'Fisher Analysis of LISA Scenario B', fontsize = 18)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERLISA_B.png')
#%%
def sigp(f):
    f0 = 1
    t1 = (1-0.475*np.exp(-(f/f0-25)**2/50))
    t2 = (1-5e-4*np.exp(-(f/f0-20)**2/100))
    t3 = (1-0.2*np.exp(-((f/f0-47)**2)**0.85/100))
    t4 = (1-0.1*np.exp(-((f/f0-50)**2)**0.7/100)-0.2*np.exp(-(f/f0-45)**2/250)+0.15*np.exp(-(f/f0-85)**2/400))
    res = 0.88*((9*(f/f0)**(-30)+5.5e-6 *(f/f0)**(-4.5)+0.28e-11 * (f/f0)**3.2)*(1/2 - 1/2*np.tanh(0.06*(f/f0-42)))
                +(1/2*np.tanh(0.06*(f/f0-42)))*(0.01e-11*(f/f0)**1.9 + 20e-13 *(f/f0)**2.8))*t1*t2*t3*t4*(0.67)**2
    return res

def f00et(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: (10**(2*astar)*(f/f0)**(2*nt)*np.log(10)**2)/sigp(f)**2
    res = quad(integrand, 1, 445, args=(f0, nt, astar))[0]
    return T*res

def f01et(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: ((10**(astar))*(f/f0)**(2*nt)*np.log(10)*np.log(f/f0))/sigp(f)**2
    res = quad(integrand, 1, 445, args=( f0, nt, astar))[0]
    return T*res

def f11et(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: (10**(2*astar)*(f/f0)**(2*nt)*np.log(f/f0)**2)/sigp(f)**2
    res = quad(integrand, 1, 445, args=( f0, nt, astar))[0]
    return T*res


def fisheret(f0, nt, astar):
    res = np.array(((f00et(f0, nt, astar), f01et(f0, nt, astar)), 
                        (f01et(f0, nt, astar), f11et(f0, nt, astar))))
    return res

ET = np.array(((fs3, nt3, a3), (fs4, nt4, a4)))
ETfm = np.array(list(map(lambda args: fisheret(*args), ET)))

FMEA = ETfm[0]
FMEB = ETfm[0]
#%%
covmA = np.linalg.inv((FMEA))
covmB = np.linalg.inv((FMEB))


meansA = np.array((a3,nt3))
meansB = np.array((a4,nt4))
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
g.triangle_plot([samples], contour_colors = ['darkblue'], #param_limits=param_limitsA,
                filled=True, markers={r'\alpha_*': meansA[0],'nt': meansA[1]}, title_limit=1)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERET_A.png')

g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=14
g.settings.legend_fontsize = 16
g.settings.axes_labelsize = 16
g.triangle_plot([samples2], contour_colors = ['blue'], 
                filled=True, markers={r'\alpha_*': meansB[0],'nt': meansB[1]}, title_limit=1)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERET_B.png')
#%%
########################
#Combining
########################
def f00(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: (10**(2*astar)*(f/f0)**(2*nt)*np.log(10)**2)/Ohms(f)**2
    res = quad(integrand, ffmin, ffmax, args=(f0, nt, astar))[0]
    return T*res

def f01(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: ((10**(astar))*(f/f0)**(2*nt)*np.log(10)*np.log(f/f0))/Ohms(f)**2
    res1 = quad(integrand, ffmin, 1e-4, args=( f0, nt, astar))[0]
    res2 = quad(integrand, 1e-4, 1e-0, args=( f0, nt, astar))[0]
    res3 = quad(integrand, 1e-0, ffmax, args=( f0, nt, astar))[0]
    return T*sum((res1, res2, res3))

def f11(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: (10**(2*astar)*(f/f0)**(2*nt)*np.log(f/f0)**2)/Ohms(f)**2
    res = quad(integrand, ffmin, ffmax, args=( f0, nt, astar))[0]
    return T*res


def fisher(f0, nt, astar):
    res = np.array(((f00(f0, nt, astar), f01(f0, nt, astar)), 
                        (f01(f0, nt, astar), f11(f0, nt, astar))))
    return res

lisac = np.array(((fs5, nt5, a5), (fs6, nt6, a6)))
LISAfmC = np.array(list(map(lambda args: fisher(*args), lisa)))

FMLAC = LISAfmC[0]
FMLBC = LISAfmC[1]

def f00et(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: (10**(2*astar)*(f/f0)**(2*nt)*np.log(10)**2)/sigp(f)**2
    res = quad(integrand, ffmin, ffmax, args=(f0, nt, astar))[0]
    return T*res

def f01et(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: ((10**(astar))*(f/f0)**(2*nt)*np.log(10)*np.log(f/f0))/sigp(f)**2
    res = quad(integrand, ffmin, ffmax, args=( f0, nt, astar))[0]
    return T*res

def f11et(f0, nt, astar):
    integrand = lambda f, f0, nt, astar: (10**(2*astar)*(f/f0)**(2*nt)*np.log(f/f0)**2)/sigp(f)**2
    res = quad(integrand, ffmin, ffmax, args=( f0, nt, astar))[0]
    return T*res


def fisheret(f0, nt, astar):
    res = np.array(((f00et(f0, nt, astar), f01et(f0, nt, astar)), 
                        (f01et(f0, nt, astar), f11et(f0, nt, astar))))
    return res

ETC = np.array(((fs5, nt5, a5), (fs6, nt6, a6)))
ETfmC = np.array(list(map(lambda args: fisheret(*args), ET)))

FMEAC = ETfm[0]
FMEBC = ETfm[0]


#all together now
FMA = FMLA+FMEA#FMLAC + FMEAC
FMB = FMLB + FMEB#FMLBC + FMEBC

covmA = np.linalg.inv((FMA))
covmB = np.linalg.inv((FMB))


meansA = np.array((a5,nt5))
meansB = np.array((a6,nt6))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp)
names = [r'\alpha_*',r'nt']
labels =  [r'\alpha_*',r'nt']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Scenario A')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Scenario B')


g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=18
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.triangle_plot([samples], contour_colors = ['indigo'],
                filled=True, markers={r'\alpha_*': meansA[0],'nt': meansA[1]}, title_limit=1)
plt.suptitle(r'Fisher Analysis of LISA + ET Scenario A', fontsize = 18)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERcomb_A.png')

g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=18
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.triangle_plot([samples2], contour_colors = ['darkviolet'], 
                filled=True, markers={r'\alpha_*': meansB[0],'nt': meansB[1]}, title_limit=1)
plt.suptitle(r'Fisher Analysis of LISA + ET Scenario B', fontsize = 18)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHERcomb_B.png')








