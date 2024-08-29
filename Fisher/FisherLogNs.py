import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import getdist
from getdist import plots, MCSamples
from scipy.interpolate import UnivariateSpline
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

itera = 2000

##########

elminL = (np.log10(ffmin))
elmaxL = (np.log10(10**(-0.95)))
elminet = np.log10(1.6)
elmaxet = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera

file = "LNS"
subject = "Inf"

size = 35
tsize = 45


om1 = 3e-10
fs1 = 0.2
rho1 = 0.45

om2 = 7e-13
fs2 = 0.25
rho2 = 1.6

#%%
props = dict(boxstyle='square', facecolor='white', alpha=1)
txt = 40
textstr1 = '\n'.join((
    r'$\Omega_\star = {om}$'.format(om = om1),
    r'$\rho = {r}$'.format(r = rho1),
    r'$f_\star = {fs}$'.format(fs = fs1)))

textstr2 = '\n'.join((
    r'$\Omega_\star = {om}$'.format(om = om2),
    r'$\rho = {r}$'.format(r = rho2),
    r'$f_\star = {fs}$'.format(fs = fs2)))

#%%
P = 12
A = 3

def P_acc(f):
    res = A**2 *(1e-15)**2 * (1+(0.4e-3 / f)**2)*(1+(f/8e-3)**4)*(2*pi*f)**(-4)*(2*pi*f/c)**2
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
#%%
def f00(om, rho, f0):
    integrand = lambda f, f0, rho: (np.exp(-np.log(f/f0)**2/rho))/Ohms(f)**2
    res = quad(integrand, ffmin, ffmax, args=(f0, rho))[0]
    return 2*T*res

def f01(om, rho, f0):
    integrand = lambda f, om, rho, f0: (om*np.exp(-np.log(f/f0)**2/rho)*np.log(f/f0)**2/(2*rho**2))/Ohms(f)**2
    res = quad(integrand, ffmin, ffmax, args=( om, rho, f0))[0]
    return 2*T*res

def f02(om, rho, f0):
    integrand = lambda f, om, rho, f0: (om*np.exp(-np.log(f/f0)**2/rho)*np.log(f/f0)/(f0*rho))/Ohms(f)**2
    res = quad(integrand, ffmin, ffmax, args=( om, rho, f0))[0]
    return 2*T*res

def f11(om, rho, f0):
    integrand = lambda f, om, rho, f0: (om**2*np.exp(-np.log(f/f0)**2/rho)*np.log(f/f0)**4/(4*rho**4))/Ohms(f)**2
    res = quad(integrand, ffmin, ffmax, args=( om, rho, f0))[0]
    return 2*T*res

def f12(om, rho, f0):
    integrand = lambda f, om, rho, f0: (om**2*np.exp(-np.log(f/f0)**2/rho)*np.log(f/f0)**3/(2*f0*rho**3))/Ohms(f)**2
    res = quad(integrand, ffmin, ffmax, args=( om, rho, f0))[0]
    return 2*T*res

def f22(om, rho, f0):
    integrand = lambda f, om, rho, f0: (om**2*np.exp(-np.log(f/f0)**2/rho)*np.log(f/f0)**2/(f0**2*rho**2))/Ohms(f)**2
    res = quad(integrand, ffmin, ffmax, args=( om, rho, f0))[0]
    return 2*T*res

def fisher(om, rho, f0):
    res = np.array(((f00(om, rho, f0), f01(om, rho, f0), f02(om, rho, f0)), 
                        (f01(om, rho, f0), f11(om, rho, f0), f12(om, rho, f0)),
                        (f02(om, rho, f0), f12(om, rho, f0), f22(om, rho, f0))))
    return res

lisa = [(om1, rho1, fs1), (om2, rho2, fs2)]
LISAfm = np.array(list(map(lambda args: fisher(*args), lisa)))

FMLA = LISAfm[0]
FMLB = LISAfm[1]
covmA = np.linalg.inv((FMLA))
covmB = np.linalg.inv((FMLB))


meansA = np.array((om1,rho1, fs1))
meansB = np.array((om2, rho2, fs2))

nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp, tol=1e-6)

names = [r'\Omega_\star',r'\rho', r'f_\star']
labels =  [r'\Omega_\star',r'\rho', r'f_\star']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Case 1')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Case 2')

#%%
g = plots.get_subplot_plotter(subplot_size=6)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.triangle_plot([samples], contour_colors = ['Green'],
                filled=True, markers={r'\Omega_\star': meansA[0],'\rho': meansA[1], r'f_\star': meansA[2]}, title_limit=1)
plt.text(0.7,0.7, textstr1, ha='center', fontsize=txt, bbox = props, transform=plt.gcf().transFigure)
plt.suptitle(r'LISA {sub}1'.format(sub=subject), fontsize=tsize)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHER_LISA_{fl}2.png'.format(fl=file))

g = plots.get_subplot_plotter(subplot_size=6)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.triangle_plot([samples2], contour_colors = ['darkblue'], 
                filled=True, markers={r'\Omega_\star': meansB[0],'\rho': meansB[1], r'f_\star': meansB[2]}, title_limit=1)
plt.text(0.7,0.7, textstr2, ha='center', fontsize=txt, bbox = props, transform=plt.gcf().transFigure)
plt.suptitle(r'LISA {sub}2'.format(sub=subject), fontsize=tsize)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHER_LISA_{fl}2.png'.format(fl=file))

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
    return 0.816**2 * res

def f00et(om, rho, f0):
    integrand = lambda f, f0, rho: (np.exp(-np.log(f/f0)**2/rho))/sigp(f)**2
    res = quad(integrand, ffmin, ffmax, args=(f0, rho))[0]
    return 2*T*res

def f01et(om, rho, f0):
    integrand = lambda f, om, rho, f0: (om*np.exp(-np.log(f/f0)**2/rho)*np.log(f/f0)**2/(2*rho**2))/sigp(f)**2
    res = quad(integrand, ffmin, ffmax, args=( om, rho, f0))[0]
    return 2*T*res

def f02et(om, rho, f0):
    integrand = lambda f, om, rho, f0: (om*np.exp(-np.log(f/f0)**2/rho)*np.log(f/f0)/(f0*rho))/sigp(f)**2
    res = quad(integrand, ffmin, ffmax, args=( om, rho, f0))[0]
    return 2*T*res

def f11et(om, rho, f0):
    integrand = lambda f, om, rho, f0: (om**2*np.exp(-np.log(f/f0)**2/rho)*np.log(f/f0)**4/(4*rho**4))/sigp(f)**2
    res = quad(integrand, ffmin, ffmax, args=( om, rho, f0))[0]
    return 2*T*res

def f12et(om, rho, f0):
    integrand = lambda f, om, rho, f0: (om**2*np.exp(-np.log(f/f0)**2/rho)*np.log(f/f0)**3/(2*f0*rho**3))/sigp(f)**2
    res = quad(integrand, ffmin, ffmax, args=( om, rho, f0))[0]
    return 2*T*res

def f22et(om, rho, f0):
    integrand = lambda f, om, rho, f0: (om**2*np.exp(-np.log(f/f0)**2/rho)*np.log(f/f0)**2/(f0**2*rho**2))/sigp(f)**2
    res = quad(integrand, ffmin, ffmax, args=( om, rho, f0))[0]
    return 2*T*res

def fisheret(om, rho, f0):
    res = np.array(((f00et(om, rho, f0), f01et(om, rho, f0), f02et(om, rho, f0)), 
                        (f01et(om, rho, f0), f11et(om, rho, f0), f12et(om, rho, f0)),
                        (f02et(om, rho, f0), f12et(om, rho, f0), f22et(om, rho, f0))))
    return res

ET = [(om1, rho1, fs1), (om2, rho2, fs2)]
ETfm = np.array(list(map(lambda args: fisheret(*args), ET)))

FMEA = ETfm[0]
FMEB = ETfm[1]
#%%
covmA = np.linalg.inv((FMEA))
covmB = np.linalg.inv((FMEB))

meansA = np.array((om1, rho1, fs1))
meansB = np.array((om2, rho2, fs2))

nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp, tol=1e-6)
names = [r'\Omega_\star',r'\rho', r'f_\star']
labels =  [r'\Omega_\star',r'\rho', r'f_\star']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Case 1')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Case 2')

#%%

g = plots.get_subplot_plotter(subplot_size=6)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.triangle_plot([samples], contour_colors = ['Green'], 
                filled=True, markers={r'\Omega_\star': meansA[0],'\rho': meansA[1], r'f_\star': meansA[2]}, title_limit=1)
plt.text(0.7,0.7, textstr1, ha='center', fontsize=txt, bbox = props, transform=plt.gcf().transFigure)
plt.suptitle(r'ET {sub}1'.format(sub=subject), fontsize=tsize)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHER_ET_{fl}1.png'.format(fl=file)


g = plots.get_subplot_plotter(subplot_size=6)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.triangle_plot([samples2], contour_colors = ['mediumblue'], 
                filled=True, markers={r'\Omega_\star': meansB[0],'\rho': meansB[1], r'f_\star': meansB[2]}, title_limit=1)
plt.text(0.7,0.7, textstr2, ha='center', fontsize=txt, bbox = props, transform=plt.gcf().transFigure)
plt.suptitle(r'ET {sub}2'.format(sub=subject), fontsize=tsize)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHER_ET_{fl}2.png'.format(fl=file))

#%%
#all together now
FMAC = FMLA + FMEA
FMBC = FMLB + FMEB

covmA = np.linalg.inv((FMAC))
covmB = np.linalg.inv((FMBC))


meansA = np.array((om1, rho1, fs1))
meansB = np.array((om2, rho2, fs2))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp, tol=1e-6)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp, tol=1e-6)
names = [r'\Omega_\star',r'\rho', r'f_\star']
labels =  [r'\Omega_\star',r'\rho', r'f_\star']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Case 1')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Case 2')

#%%
g = plots.get_subplot_plotter(subplot_size=6)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.triangle_plot([samples], contour_colors = ['Green'],
                filled=True, markers={r'\Omega_\star': meansA[0],'\rho': meansA[1], r'f_\star': meansA[2]}, title_limit=1)
plt.text(0.7,0.7, textstr1, ha='center', fontsize=txt, bbox = props, transform=plt.gcf().transFigure)
plt.suptitle(r'LISA+ET {sub}1'.format(sub=subject), fontsize=tsize)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHER_COMB_{fl}1.png'.format(fl=file))


g = plots.get_subplot_plotter(subplot_size=6)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.triangle_plot([samples2], contour_colors = ['blue'], 
                filled=True, markers={r'\Omega_\star': meansB[0],'\rho': meansB[1], r'f_\star': meansB[2]}, title_limit=1)
plt.text(0.7,0.7, textstr2, ha='center', fontsize=txt, bbox = props, transform=plt.gcf().transFigure)
plt.suptitle(r'LISA+ET {sub}2'.format(sub=subject), fontsize=tsize)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHER_COMB_{fl}2.png'.format(fl=file))


































