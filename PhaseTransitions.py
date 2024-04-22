import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import time
#%%
#Now we want to start looking at what we get for broken power laws.
#Let's start by defining our analytical values again

#Getting the data file
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

itera = 750

##########
elminL = (np.log10(ffmin))
elmaxL = (np.log10(10**(-1)))
elminet = np.log10(1.6)
elmaxet = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera


#%%
def sigp(f):
    res = 0.9 * ((3 * 30 * 10**(-1) * f**(-30) + 5.5 * 10**(-6) * f**(-4.5) + 
            0.7 * 10**(-11) * f**(2.8)) * (1/2 - 
            1/2 * np.tanh(0.04 * (f - 42))) + (1/2 * np.tanh(0.04 * (f - 42))) * 
            (0.4 * 10**(-11) * f**(1.4) + 7.9 * 10**(-13) * f**(2.98))) * (1 - 
            0.38 * np.exp(-(f - 25)**2/50))
    return res



def etnomonly(f):
    res = sigp(f)
    if res > 10**(-5):
        return
    return res

def sigETapp(f):#Sigma_Ohm approx
    res = sigp(f)
    return res
#%%
fvalsET = np.logspace(np.log10(1), np.log10(445),2000)#frequency values
sigETvals = np.array(list(map(etnomonly, fvalsET)))#The Omega_gw values from the ET data


plt.loglog(fvalsET, sigETvals, color = "indigo", linewidth=2.5)
plt.title("Nominal sensitivity curve ET")
plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
#plt.ylim(10**(-9), 10**(-5))
plt.yscale('log')
plt.xscale('log')
plt.xlim(10**0, 400)
plt.grid(True)
plt.show()


#%%
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
    sig = np.sqrt(2)*20/3 * (SI(f)/(2*pi*f)**4 + SII)*(1+(f/(4*fLisa/3))**2)
    return sig

def SigmaLisaApprox(f):#Sigma_Ohm approx
    const = ((4*pi**2/(3*H0**2)))
    res = const * f**3 *sigI(f)
    return res
def SigmaLisaApproxnom(f):#Sigma_Ohm approx
    const = ((4*pi**2/(3*H0**2)))
    res = const * f**3 *sigI(f)
    if res >10**(-5):
        return
    return res
L = 25/3
fLisa = 1/(2*pi*L)
#%%
def n1(f):
    res = 4*Ss +8*Sa(f)*(1+np.cos(f/fLisa)**2)
    return res

def n2(f):
    res = -((2*Ss+8*Sa(f))*np.cos(f/fLisa))
    return res

def sigtab(f, r1, r2):
    res = 1/np.sqrt((3*H0**2/(4*pi**2*f**3))**2 * (2*((r1-r2)/(n1(f)-n2(f)))**2))
    if res > 1e-5:
        return
    return res

# og = np.array(list(map(lambda args: sigtab(*args), Rtab)))

# plt.figure(figsize=(6, 9)) 
# plt.loglog(f, og, color = "indigo", linewidth=2.5)
# plt.ylabel(r"$\Omega_{gw}$")
# plt.grid(True)
# plt.xlabel("f (Hz)")
# plt.title("Nominal Sensitivity curve of LISA")
# plt.show()
#%%    
freqvals = np.logspace(elminL, elmaxL, itera)   
sigvals = np.array(list(map(SigmaLisaApproxnom, freqvals)))

plt.figure(figsize=(6, 9)) 
# plt.loglog(f, og,'--' ,color = "darkviolet", label = "Numerical", linewidth=2.5)
plt.loglog(freqvals, sigvals, color = "indigo", label = "Approximate", linewidth=2.5)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.grid(True)
plt.legend(fontsize = 16)
plt.xlabel("f (Hz)", fontsize = 16)
plt.title("Nominal Sensitivity curve of LISA", fontsize = 16)
plt.show()

#%%
s = 1.2
n1 = 2.4
n2 = -2.4
fstar = 1e-1
a = -6

def bpl(f):
    res = 10**a * (f/fstar)**n1 * (1+(f/fstar)**s)**(-(n1-n2)/s)
    return res



def omegatog(f):
    if f <= 10**(-1):
        return SigmaLisaApprox(f)
    if f > 1.6:
        return sigETapp(f)
    
def nomtog(f):
    if f <= 10**(-1):
        res = SigmaLisaApprox(f)
        if res > 1e-5:
            return
        return res
    if f > 1:
        res = sigETapp(f)
        if res > 1e-5:
            return
        return res


step = (ffmax-ffmin)/itera
freqs = np.arange(ffmin, ffmax, step)
fvalscomb = np.logspace(np.log10(ffmin), np.log10(ffmax),750)



phase = np.array(list(map(bpl, freqs)))
combine = np.array(list(map(omegatog, fvalscomb)))
nom = np.array(list(map(nomtog, fvalscomb)))
otog = np.vstack((fvalscomb, combine)).T


#%%
plt.figure(figsize=(6, 9))
plt.loglog(otog[:,0], nom, color = "indigo", linewidth=2, label = "Nominal Curves")
# plt.loglog(freqvals, sigvals, color = "indigo", label = "LISA", linewidth=2)
# plt.loglog(fvalsET, sigETvals, '-.',color = "indigo",label = "ET", linewidth=2)
plt.loglog(freqs, phase, color = "darkviolet", label = "Phase Transition")
plt.legend()
plt.suptitle("Phase Transition")
plt.title(r"Values: n1 = {nom1}, n2 = {nom2}, $f_\star$ = {fbreak}, $\sigma$= {sig}, $\alpha\star$ = {astar}".format(nom1=n1, nom2=n2, fbreak=fstar, sig = s, astar = a), fontsize = 10)
plt.grid(True) 
plt.xlim(ffmin, ffmax) 
plt.xlabel('f (Hz)')
plt.ylabel(r"$\Omega_{gw}$")
plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/phase.png', bbox_inches='tight')
plt.show()


























