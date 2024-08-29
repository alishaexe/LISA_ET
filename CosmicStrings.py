#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import time


# In[2]:


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
#Change this value for how many 'steps' you want in the range of values

itera = 100000


# In[3]:


elminL = (np.log10(ffmin))
elmaxL = (np.log10(10**(-1)))
elminet = np.log10(1.6)
elmaxet = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera


# In[4]:


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



def etnomonly(f):
    res = sigp(f)
    if res > 10**(-5):
        return
    return res

def sigETapp(f):#Sigma_Ohm approx
    res = sigp(f)
    return res


# In[5]:


fvalsET = np.logspace(np.log10(1), np.log10(445),2000)#frequency values
sigETvals = np.array(list(map(etnomonly, fvalsET)))#The Omega_gw values from the ET data



# In[6]:
L = 25/3
fLisa = 1/(2*pi*L)
c = 3e8

P = 12
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


# In[7]:


freqvals = np.logspace(elminL, elmaxL, itera)   
sigvals = np.array(list(map(Ohms, freqvals)))


# In[66]:


s1 = 1
nA1 = -0.1
nB1 = -0.1
fS1 = 2e-2
a1 = 4*10**(-13)

def bpl1(f):
    res = a1 \
    * (f/fS1)**nA1 \
    * (1/2+1/2*(f/fS1)**s1)**(-(nA1-nB1)/s1)
    return res


# In[67]:


s2 = 3
nA2 = -0.1
nB2 = -0.5
fS2 = 2
a2 = 2.5*10**(-12)

def bpl2(f):
    res = a2 \
    * (f/fS2)**nA2 \
    * (1/2+1/2*(f/fS2)**s2)**(-(nA2-nB2)/s2)
    return res


# In[68]:


def omegatog(f):
    if f <= 10**(-0.9):
        return Ohms(f)
    if f > 1.6:
        return sigETapp(f)


# In[69]:


def nomtog(f):
    if f <= 10**(-0.9):
        res = Ohms(f)
        if res > 1e-5:
            return
        return res
    if f > 1:
        res = sigETapp(f)
        if res > 1e-5:
            return
        return res


# In[70]:


step = (ffmax-ffmin)/itera
freqs = np.arange(ffmin, ffmax, step)
fvalscomb = np.logspace(np.log10(ffmin), np.log10(ffmax),10000)


# In[71]:


phase1 = np.array(list(map(bpl1, freqs)))
combine = np.array(list(map(omegatog, fvalscomb)))
nom = np.array(list(map(nomtog, fvalscomb)))
otog = np.vstack((fvalscomb, combine)).T


# In[72]:


phase2 = np.array(list(map(bpl2, freqs)))

flogomcomb = np.load('/Users/alisha/Documents/LISA_ET/PLS.npy')

# In[75]:

combfbplo = np.loadtxt('/Users/alisha/Documents/LISA_ET/combo.txt')

plt.figure(figsize=(6, 8))
plt.loglog(otog[:,0], nom,color = "indigo", linewidth=1.5, label = "Nominal Curves")
plt.loglog((combfbplo[:,0]), (combfbplo[:,1]*0.67**2), label = "BPLS curve", color = "lime", linewidth=2.5)
plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]), color = "orangered", label = "PLS curve", linewidth=2.5)
plt.loglog(freqs, phase1,linewidth=2.5,color = "darkgreen",label = "CS 1")
plt.loglog(freqs, phase2,linewidth=2.5,color = "blue",linestyle='--',label = "CS 2")

# plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1])/0.67**2, label = "BPLS curve", color = "lime", linewidth=2.5)

plt.legend(fontsize = 12, loc = (0.08,0.75))
plt.title('SGWB from Cosmic Strings', fontsize = 16)
plt.grid(True) 
plt.xlim(ffmin, ffmax) 
plt.ylim(2e-14,1e-5)
plt.xlabel('f (Hz)', fontsize = 16)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/fig2b.png', bbox_inches='tight')
plt.show()



