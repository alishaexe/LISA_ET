import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import time
#%%
#Now we want to start looking at what we get for broken power laws.
#Let's start by defining our analytical values again

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
###############
#Change this value for how many 'steps' you want in the range of values

itera = 75

##########
elminL = (np.log10(ffmin))
elmaxL = (np.log10(10**(-1)))
elminet = np.log10(1.6)
elmaxet = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera

#%%
###########################
#LISA

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


#%%    
freqvals = np.logspace(elminL, elmaxL, itera)   
sigvals = np.array(list(map(Ohms, freqvals)))
elstep = (elmaxL-elminL)/itera
elLISA = np.arange(elminL, elmaxL+elstep, elstep)


fbplo = np.load("/Users/alisha/Documents/LISA_ET/Datafiles/BPLS/BPLS_lisa.npy")

plt.figure(figsize=(6, 9))
plt.loglog(freqvals, sigvals, label = "Nominal curve", color = "indigo", linewidth=2.5)
plt.loglog(fbplo[:,0][fbplo[:,0]<1e-1], fbplo[:,1][fbplo[:,0]<1e-1], label = "BPLS curve", color = "lime", linewidth=2.5)
plt.grid(True)
plt.legend(fontsize = 18)
plt.xlabel(r'$f$ (Hz)', fontsize = 20)
plt.ylabel(r'$\Omega_{gw}$', fontsize = 20)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.ylim(2e-14,1e-6)
plt.xlim(1e-5,9.9e-2)
plt.xscale('log')
plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/LISA_BPLS.png', bbox_inches='tight')
plt.show()
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
                                                                           
                                                                           

def etnomonly(f):
    res = sigp(f)
    if res > 10**(-5):
        return
    return res


#%%
fvalsET = np.logspace(np.log10(1e-5), np.log10(445),2000)#frequency values
sigETvals = np.array(list(map(etnomonly, fvalsET)))#The Omega_gw values from the ET data

fbploET = np.load("/Users/alisha/Documents/LISA_ET/Datafiles/BPLS/BPLS_ET.npy")


#plots all 3 graphs on same plot
plt.figure(figsize=(6, 9)) 
plt.loglog(fvalsET, sigETvals, color = "indigo",linewidth=2.5, label = "Nominal curve")
plt.loglog(fbploET[:,0][fbploET[:,0]>1e0], fbploET[:,1][fbploET[:,0]>1e0], linewidth=2.5,color = "lime", label = "BPLS curve")
plt.ylabel(r"$\Omega_{gw}$", fontsize = 20)
plt.xlabel(r'$f$ (Hz)', fontsize = 20)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.legend(fontsize=18, loc=9)
plt.grid(True)
plt.xlim(1,440)
plt.ylim(4e-14,7e-6)
plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/ET_BPLS.png', bbox_inches='tight')
plt.show()

#%%
#######################
#Combining LISA and ET curves
#######################

def omegatog(f):
    if f <= 10**(-1):
        return Ohms(f)
    if f > 1.6:
        return sigp(f)
    
def nomtog(f):
    if f <= 10**(-1):
        res = Ohms(f)
        if res > 1e-5:
            return
        return res
    if f > 1:
        res = sigp(f)
        if res > 1e-5:
            return
        return res
 
fvalscomb = np.logspace(np.log10(ffmin), np.log10(ffmax),750)
combine = np.array(list(map(omegatog, fvalscomb)))
nom = np.array(list(map(nomtog, fvalscomb)))
otog = np.vstack((fvalscomb, combine)).T

combbplo=np.load("/Users/alisha/Documents/LISA_ET/Datafiles/BPLS/BPLS_comb.npy")

#%%
plt.figure(figsize=(6, 9))
plt.loglog(otog[:,0], nom, label = "Nominal Curve", color = "indigo", linewidth=2.5)
plt.loglog(combbplo[:,0], combbplo[:,1], label = "Combined BPLS", color = "lime", linewidth=2.5)
plt.loglog(fbplo[:,0][fbplo[:,0]<1e-1], fbplo[:,1][fbplo[:,0]<1e-1],':', label = "LISA BPLS", color = "teal", linewidth=2.5)
plt.loglog(fbploET[:,0][fbploET[:,0]>1e0], fbploET[:,1][fbploET[:,0]>1e0],':', linewidth=2.5,color = "black", label = "ET BPLS")
plt.ylabel(r"$\Omega_{gw}$", fontsize = 20)
plt.xlabel(r'$f$ (Hz)', fontsize = 20)
plt.legend(loc = 2, fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.xlim(ffmin, ffmax)
plt.ylim(4e-14,1e-6)
plt.grid(True)
plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/Comb_BPLS.png', bbox_inches='tight')
plt.show() 
