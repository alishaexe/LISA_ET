import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import getdist
from getdist import plots, MCSamples
from scipy.interpolate import UnivariateSpline
from fractions import Fraction
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

tsize = 45
size=40

subject = "PT"

file = "PT"

#case 1
o1 = 1e-10
n1 = 3
n2 = -1
s1 = 7.2
fstar = 0.04

#case 2
o2 = 1e-8
nom1 = 2.4
nom2 = -2.4
s2 = 1.2
fbreak = 0.2


props = dict(boxstyle='square', facecolor='white', alpha=1)
txt = 50
def leg(tele, scen):
    if scen ==1:
        textstr1 = '\n'.join((
            r'{tel} {sub}{case}'.format(tel=tele, case = scen, sub=subject),
            r'$\Omega_\star = {om}$'.format(om = o1),
            r'$n_1 = {n1}$'.format(n1 = n1),
            r'$n_2 = {n2}$'.format(n2 = n2), #Fraction(n2).limit_denominator()),
            r'$\sigma = {s}$'.format(s = s1),
            r'$f_\star = {fs}$'.format(fs = fstar)))
        return textstr1
    if scen ==2:
        textstr2 = '\n'.join((
            r'{tel} {sub}{case}'.format(tel=tele, case = scen, sub=subject),
            r'$\Omega_\star = {om}$'.format(om = o2),
            r'$n_1 = {n1}$'.format(n1 = nom1),
            r'$n_2 = {n2}$'.format(n2 = nom2), #Fraction(nom2).limit_denominator()),
            r'$\sigma = {s}$'.format(s = s2),
            r'$f_\star = {fs}$'.format(fs = fbreak)))
        return textstr2

#%%
P = 15
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

def f00(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, s: (f/f0)**(2*n1)*(0.5*(f/f0)**s + 0.5)**(2*(-n1 + n2)/s)/Ohms(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, s))[0]
    I2 = quad(integrand, 1e-1, 1e1, args=(f0, n1, n2, s))[0]
    I3 = quad(integrand, 1e1, ffmax, args=(f0, n1, n2, s))[0]
    return 2*T*sum((I1,I2,I3))

def f01(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s: ((f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*(om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(f/f0) - om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(0.5*(f/f0)**s + 0.5)/s))/Ohms(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def f02(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s: (om*(f/f0)**(2*n1)*(0.5*(f/f0)**s + 0.5)**(2*(-n1 + n2)/s)*np.log(0.5*(f/f0)**s + 0.5)/s)/Ohms(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def f03(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s: (om*(f/f0)**(2*n1)*(0.5*(f/f0)**s + 0.5)**(2*(-n1 + n2)/s)*(0.5*(f/f0)**s*(-n1 + n2)*np.log(f/f0)/(s*(0.5*(f/f0)**s + 0.5)) - (-n1 + n2)*np.log(0.5*(f/f0)**s + 0.5)/s**2))/Ohms(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def f11(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s: ((om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(f/f0) - om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(0.5*(f/f0)**s + 0.5)/s)**2)/Ohms(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def f12(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s: (om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*(om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(f/f0) - om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(0.5*(f/f0)**s + 0.5)/s)*np.log(0.5*(f/f0)**s + 0.5)/s)/Ohms(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def f13(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s: (om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*(om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(f/f0) - om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(0.5*(f/f0)**s + 0.5)/s)*(0.5*(f/f0)**s*(-n1 + n2)*np.log(f/f0)/(s*(0.5*(f/f0)**s + 0.5)) - (-n1 + n2)*np.log(0.5*(f/f0)**s + 0.5)/s**2))/Ohms(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def f22(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s:  (om**2*(f/f0)**(2*n1)*(0.5*(f/f0)**s + 0.5)**(2*(-n1 + n2)/s)*np.log(0.5*(f/f0)**s + 0.5)**2/s**2)/Ohms(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def f23(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s:  (om**2*(f/f0)**(2*n1)*(0.5*(f/f0)**s + 0.5)**(2*(-n1 + n2)/s)*(0.5*(f/f0)**s*(-n1 + n2)*np.log(f/f0)/(s*(0.5*(f/f0)**s + 0.5)) - (-n1 + n2)*np.log(0.5*(f/f0)**s + 0.5)/s**2)*np.log(0.5*(f/f0)**s + 0.5)/s)/Ohms(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def f33(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s:  (om**2*(f/f0)**(2*n1)*(0.5*(f/f0)**s + 0.5)**(2*(-n1 + n2)/s)*(0.5*(f/f0)**s*(-n1 + n2)*np.log(f/f0)/(s*(0.5*(f/f0)**s + 0.5)) - (-n1 + n2)*np.log(0.5*(f/f0)**s + 0.5)/s**2)**2)/Ohms(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def fisher(f0, n1, n2, om, s):
    res = np.array(((f00(f0, n1, n2, om, s), f01(f0, n1, n2, om, s), f02(f0, n1, n2, om, s), f03(f0, n1, n2, om, s)), 
                    (f01(f0, n1, n2, om, s), f11(f0, n1, n2, om, s), f12(f0, n1, n2, om, s), f13(f0, n1, n2, om, s)),
                    (f02(f0, n1, n2, om, s), f12(f0, n1, n2, om, s), f22(f0, n1, n2, om, s), f23(f0, n1, n2, om, s)),
                    (f03(f0, n1, n2, om, s), f13(f0, n1, n2, om, s), f23(f0, n1, n2, om, s), f33(f0, n1, n2, om, s))))
    return res

lisa = [(fstar, n1, n2, o1, s1), (fbreak, nom1, nom2, o2, s2)]
LISAfm = np.array(list(map(lambda args: fisher(*args), lisa)))

FMLA = LISAfm[0]
FMLB = LISAfm[1]
#%%
covmA = np.linalg.inv((FMLA))
covmB = np.linalg.inv((FMLB))



meansA = np.array((o1, n1, n2,s1))
meansB = np.array((o2, nom1, nom2,s2))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp, tol=1e-6)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp, tol=1e-6)
names = [r'\Omega_\star',r'n1', r'n2',r'\sigma']
labels =  [r'\Omega_\star',r'n1', r'n2',r'\sigma']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Case 1')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Case 2')


#%%
g = plots.get_subplot_plotter(subplot_size=6)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.settings.scaling = False
g.triangle_plot([samples], contour_colors = ['Green'], fig_width_inch = 7,
                filled=True, markers={r'\Omega_\star': meansA[0],'n1': meansA[1], 'n2':meansA[2], r'\sigma':meansA[3]}, title_limit=1)
plt.text(0.7,0.7, leg("LISA", 1), ha='center', fontsize=txt, bbox = props, transform=plt.gcf().transFigure)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHER_LISA_{fl}1.png'.format(fl=file), bbox_inches='tight')

# #%%
g = plots.get_subplot_plotter(subplot_size=6)
g.settings.axes_fontsize=33
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.settings.scaling = False
g.triangle_plot([samples2], contour_colors = ['darkblue'], 
                filled=True, markers={r'\Omega_\star': meansB[0],'n1': meansB[1], 'n2':meansB[2], r'\sigma':meansB[3]}, title_limit=1)
plt.text(0.7,0.7, leg("LISA", 2), ha='center', fontsize=txt, bbox = props, transform=plt.gcf().transFigure)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHER_LISA_{fl}2.png'.format(fl=file), bbox_inches='tight')

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

def f00et(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, s: (f/f0)**(2*n1)*(0.5*(f/f0)**s + 0.5)**(2*(-n1 + n2)/s)/sigp(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, s))[0]
    I2 = quad(integrand, 1e-1, 1e1, args=(f0, n1, n2, s))[0]
    I3 = quad(integrand, 1e1, ffmax, args=(f0, n1, n2, s))[0]
    return 2*T*sum((I1,I2,I3))

def f01et(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s: ((f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*(om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(f/f0) - om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(0.5*(f/f0)**s + 0.5)/s))/sigp(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def f02et(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s: (om*(f/f0)**(2*n1)*(0.5*(f/f0)**s + 0.5)**(2*(-n1 + n2)/s)*np.log(0.5*(f/f0)**s + 0.5)/s)/sigp(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def f03et(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s: (om*(f/f0)**(2*n1)*(0.5*(f/f0)**s + 0.5)**(2*(-n1 + n2)/s)*(0.5*(f/f0)**s*(-n1 + n2)*np.log(f/f0)/(s*(0.5*(f/f0)**s + 0.5)) - (-n1 + n2)*np.log(0.5*(f/f0)**s + 0.5)/s**2))/sigp(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def f11et(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s: ((om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(f/f0) - om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(0.5*(f/f0)**s + 0.5)/s)**2)/sigp(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def f12et(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s: (om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*(om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(f/f0) - om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(0.5*(f/f0)**s + 0.5)/s)*np.log(0.5*(f/f0)**s + 0.5)/s)/sigp(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))
def f13et(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s: (om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*(om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(f/f0) - om*(f/f0)**n1*(0.5*(f/f0)**s + 0.5)**((-n1 + n2)/s)*np.log(0.5*(f/f0)**s + 0.5)/s)*(0.5*(f/f0)**s*(-n1 + n2)*np.log(f/f0)/(s*(0.5*(f/f0)**s + 0.5)) - (-n1 + n2)*np.log(0.5*(f/f0)**s + 0.5)/s**2))/sigp(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))
def f22et(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s:  (om**2*(f/f0)**(2*n1)*(0.5*(f/f0)**s + 0.5)**(2*(-n1 + n2)/s)*np.log(0.5*(f/f0)**s + 0.5)**2/s**2)/sigp(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))
def f23et(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s:  (om**2*(f/f0)**(2*n1)*(0.5*(f/f0)**s + 0.5)**(2*(-n1 + n2)/s)*(0.5*(f/f0)**s*(-n1 + n2)*np.log(f/f0)/(s*(0.5*(f/f0)**s + 0.5)) - (-n1 + n2)*np.log(0.5*(f/f0)**s + 0.5)/s**2)*np.log(0.5*(f/f0)**s + 0.5)/s)/sigp(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))
def f33et(f0, n1, n2, om, s):
    integrand = lambda f, f0, n1, n2, om, s:  (om**2*(f/f0)**(2*n1)*(0.5*(f/f0)**s + 0.5)**(2*(-n1 + n2)/s)*(0.5*(f/f0)**s*(-n1 + n2)*np.log(f/f0)/(s*(0.5*(f/f0)**s + 0.5)) - (-n1 + n2)*np.log(0.5*(f/f0)**s + 0.5)/s**2)**2)/sigp(f)**2
    I1 = quad(integrand, ffmin, 1e-1, args=(f0, n1, n2, om, s))[0]
    I2 = quad(integrand, 1e-1, ffmax, args=(f0, n1, n2, om, s))[0]
    return 2*T*sum((I1,I2))

def fisheret(f0, n1, n2, om, s):
    res = np.array(((f00et(f0, n1, n2, om, s), f01et(f0, n1, n2, om, s), f02et(f0, n1, n2, om, s), f03et(f0, n1, n2, om, s)), 
                    (f01et(f0, n1, n2, om, s), f11et(f0, n1, n2, om, s), f12et(f0, n1, n2, om, s), f13et(f0, n1, n2, om, s)),
                    (f02et(f0, n1, n2, om, s), f12et(f0, n1, n2, om, s), f22et(f0, n1, n2, om, s), f23et(f0, n1, n2, om, s)),
                    (f03et(f0, n1, n2, om, s), f13et(f0, n1, n2, om, s), f23et(f0, n1, n2, om, s), f33et(f0, n1, n2, om, s))))
    return res

ET = [(fstar, n1, n2, o1, s1), (fbreak, nom1, nom2, o2, s2)]
ETfm = np.array(list(map(lambda args: fisheret(*args), ET)))

FMEA = ETfm[0]
FMEB = ETfm[1]
#%%
covmA = np.linalg.inv((FMEA))
covmB = np.linalg.inv((FMEB))


meansA = np.array((o1, n1, n2,s1))
meansB = np.array((o2, nom1, nom2,s2))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp, tol=1e-6)
# cutsamps = samps[samps[:,0]>-0.5,:]
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp, tol=1e-6)
names = [r'\Omega_\star',r'n1', r'n2', r'\sigma']
labels =  [r'\Omega_\star',r'n1', r'n2', r'\sigma']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Case 1', ranges={'\\Omega_\\star': (0, None)})
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Case 2', ranges={'\\Omega_\\star': (0, None)})

#%%
g = plots.get_subplot_plotter(subplot_size=6)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.settings.scaling = False
g.triangle_plot([samples], contour_colors = ['forestgreen'], 
                filled=True, markers={r'\Omega_\star': meansA[0],'n1': meansA[1], 'n2':meansA[2], r'\sigma':meansA[3]}, title_limit=1)
plt.text(0.7,0.7, leg("ET", 1), ha='center', fontsize=txt, bbox = props, transform=plt.gcf().transFigure)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHER_ET_{fl}1.png'.format(fl=file), bbox_inches='tight')

# #%%
g = plots.get_subplot_plotter(subplot_size=6)
g.settings.axes_fontsize=34
g.settings.legend_fontsize = size
g.settings.axes_labelsize = size
g.settings.scaling = False
g.triangle_plot([samples2], contour_colors = ['mediumblue'], 
                filled=True, markers={r'\Omega_\star': meansB[0],'n1': meansB[1], 'n2': meansB[2], r'\sigma':meansB[3]}, title_limit=1)
plt.text(0.7,0.7, leg("ET", 2), ha='center', fontsize=txt, bbox = props, transform=plt.gcf().transFigure)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHER_ET_{fl}2.png'.format(fl=file), bbox_inches='tight')


#%%
#all together now
FMAC = FMLA + FMEA
FMBC = FMLB + FMEB

covmA = np.linalg.inv((FMAC))
covmB = np.linalg.inv((FMBC))


meansA = np.array((o1, n1, n2,s1))
meansB = np.array((o2, nom1, nom2,s2))
nsamp = int(1E6)
samps = np.random.multivariate_normal(meansA, covmA, size=nsamp, tol=1e-6)
samps2 = np.random.multivariate_normal(meansB, covmB, size=nsamp, tol=1e-6)
names = [r'\Omega_\star',r'n1', r'n2',r'\sigma']
labels =  [r'\Omega_\star',r'n1', r'n2',r'\sigma']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'Case 1')
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Case 2')

#%%
g = plots.get_subplot_plotter(subplot_size=6)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = 45
g.settings.scaling = False
g.triangle_plot([samples], contour_colors = ['limegreen'], 
                filled=True, markers={r'\Omega_\star': meansA[0],'n1': meansA[1], 'n2':meansA[2],r'\sigma':meansA[3]}, title_limit=1)
plt.text(0.7,0.65, leg("LISA+ET", 1), ha='center', fontsize=txt, bbox = props, transform=plt.gcf().transFigure)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHER_Comb_{fl}1.png'.format(fl=file), bbox_inches='tight')

#%%
g = plots.get_subplot_plotter(subplot_size=6)
g.settings.axes_fontsize=size
g.settings.legend_fontsize = size
g.settings.axes_labelsize = 45
g.settings.scaling = False
g.triangle_plot([samples2], contour_colors = ['blue'], 
                filled=True, markers={r'\Omega_\star': meansB[0],'n1': meansB[1], 'n2':meansB[2], r'\sigma':meansB[3]}, title_limit=1)
plt.text(0.7,0.65, leg("LISA+ET", 2), ha='center', fontsize=txt, bbox = props, transform=plt.gcf().transFigure)
plt.savefig('/Users/alisha/Documents/LISA_ET/Fisher graphs/FISHER_Comb_{fl}2.png'.format(fl=file), bbox_inches='tight')



#%%


# arr = covmA #you will need to run the cell with the covariance matrix you want
#               #you want to check first since they all have the same name

# #Check if the matrix is symmetric
# is_symmetric = np.allclose(arr, arr.T)
# print("Is symmetric:", is_symmetric)

# #Check if all principal minors are positive
# minors_positive = all(np.linalg.det(arr[:i, :i]) > 0 for i in range(1, arr.shape[0] + 1))
# print("All principal minors are positive:", minors_positive)













 
