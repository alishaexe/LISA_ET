import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import interpolate
#%%

#Now going to define constants
yr = 365*24*60*60 #in seconds
H0 = 100*0.67*10**(3)/(3.086*10**(22)) #1/seconds and setting h = 0.67
# H0 = 3.24e-18 #Debikas value

pi = np.pi
T = 3*yr
snr5 = 5
c = 3e8

#For the LISA mission they have designed the
#arms to be length L = 2.5*10^(6)

#L = 2.5e9
L = 25/3

#%%
P = 15
A = 3

def P_acc(f):
    res = A**2 *(1e-15)**2 * (1+(0.4e-3 / f)**2)*(1+(f/8e-3)**4)*1/(2*pi*f)**(4)*(2*pi*f/c)**2
    return res

def P_ims(f):
    res = P**2 * (1e-12)**2 *(1+(2e-3/f)**4)*(2*pi*f/c)**2
    return res

def N_AA(f):
    con = 2*pi*f*L
    res = 8 * np.sin(con)**2 * ((4*(1+np.cos(con)+(np.cos(con))**2)*P_acc(f))+((2+np.cos(con))*P_ims(f)))
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

def sigl(f):
    res = (4*pi**2)/(3*H0**2)*f**3*S_n(f)
    return res

print (sigl(1e-1)) 

#This is Sigma Omega_ET
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
#%%

def PT1(f):
    ostar =1e-10
    n1 = 3
    n2 = -1
    sigma = 7.2
    fstar = 0.04
    
    ogw = ostar*(f/fstar)**n1*(1/2+1/2*(f/fstar)**sigma)**((n2-n1)/sigma)
    return ogw


def SNRL():
    integrand = lambda f:(PT1(f)**2/sigl(f)**2)
    I1 = quad(integrand, 1e-5, 1e-1)[0]
    I2 = quad(integrand, 1e-1, 445)[0]
    return np.sqrt(T*(I1+I2))

LISA_SNR = SNRL()
print("PT1 case LISA",LISA_SNR)

def SNRE():
    integrand = lambda f:(PT1(f)**2/sigp(f)**2)
    I1 = quad(integrand, 1e-5, 445)[0]
    return np.sqrt(T*I1)

ET_SNR = SNRE()
print("PT1 case ET",ET_SNR)
#%%

def PT2(f):
    ostar =1e-8
    n1 = 2.4
    n2 = -2.4
    sigma = 1.2
    fstar = 0.2
    
    ogw = ostar*(f/fstar)**n1*(1/2+1/2*(f/fstar)**sigma)**((n2-n1)/sigma)
    return ogw

def SNRL():
    integrand = lambda f:(PT2(f)**2/Ohms(f)**2)
    I1 = quad(integrand, 1e-5, 1e-1)[0]
    I2 = quad(integrand, 1e-1, 445)[0]
    return np.sqrt(T*(I1+I2))

LISA_SNR = SNRL()
print("PT2 case LISA",LISA_SNR)

def SNRE():
    integrand = lambda f:(PT2(f)**2/sigp(f)**2)
    I1 = quad(integrand, 1e-5, 445)[0]
    return np.sqrt(T*I1)

ET_SNR = SNRE()
print("PT2 case ET",ET_SNR)