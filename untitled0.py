import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
#%%
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

itera = 200

##########

elminL = (np.log10(ffmin))
elmaxL = (np.log10(10**(-0.9)))
elminet = np.log10(1.6)
elmaxet = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera

#%%
def sigp(f):
    f0 = 1
    t1 = (1-0.475*np.exp(-(f/f0-25)**2/50))
    t2 = (1-5e-4*np.exp(-(f/f0-20)**2/100))
    t3 = (1-0.2*np.exp(-((f/f0-47)**2)**0.85/100))
    t4 = (1-0.12*np.exp(-((f/f0-50)**2)**0.7/100)-0.2*np.exp(-(f/f0-45)**2/250)+0.15*np.exp(-(f/f0-85)**2/400))
    
    res = 0.88*((9*(f/f0)**(-30)+5.5e-6 *(f/f0)**(-4.5)+0.28e-11 * (f/f0)**3.2)*(1/2 - 1/2*np.tanh(0.06*(f/f0-42)))
                +(1/2*np.tanh(0.06*(f/f0-42)))*(0.01e-11*(f/f0)**1.9 + 20e-13 *(f/f0)**2.8))*t1*t2*t3*t4*(0.67)**2
    return res

def etnomonly(f):
    res = sigp(f)
    if res > 10**(-5):
        return
    return res
#%%
fvalsET = np.logspace(0, 3,2000)#frequency values
sigETvals = np.array(list(map(etnomonly, fvalsET)))#The Omega_gw values from the ET data

plt.figure(figsize=(6, 9)) 
plt.loglog(fvalsET, sigETvals, color = "indigo", linewidth=2.5)
plt.title("Nominal sensitivity curve ET")
plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
#plt.ylim(10**(-9), 10**(-5))
plt.yscale('log')
plt.xscale('log')
# plt.xlim(10**0, 400)
plt.grid(True)
plt.show()

#%%
#I separate the integration into 4 to help with the accuracy and avoid warnings
def AETmin(nt):
    integrand = lambda f, nt: ((f/fetstar)**(nt)/sigp(f))**2
    I1 = quad(integrand, 1.6, 100, args=(nt))[0]
    I2 = quad(integrand, 100, 445, args = (nt))[0]
    res = snr5/np.sqrt(2*T*sum((I1, I2)))
    return 2*res



ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera
ntetvals = np.arange(ntmin, ntmax+step, step)

#this array/list/map function does the same as earlier by mapping the nt values across
#without having to iterate
aetvals = np.array(list(map(AETmin, ntetvals)))
AETtab = np.vstack((ntetvals, aetvals)).T

def FETtab(i, j):
    res = AETtab[j,1]*(10**elET[i]/fetstar)**AETtab[j,0]
    return res



#%%

elstep = (elmaxet-elminet)/itera
elET = np.arange(elminet, elmaxet, elstep)
i = range(len(elET))
j = range(len(AETtab))
coordset = np.array(np.meshgrid(i,j)).T.reshape(-1,2)
Ftabetpls = np.array(list(map(lambda args: FETtab(*args), coordset))).reshape(len(elET), len(AETtab))
maxedET = []

def maxETpls(i):
    maxedET = np.log(np.max(Ftabetpls[i]))
    return maxedET

maxposet = range(len(Ftabetpls))
maxvals = np.array(list(map(maxETpls, maxposet)))
maxplsvals = maxvals
flogomET = np.vstack((np.log(10**elET), maxplsvals)).T
#%%
plt.figure(figsize=(6, 9)) 
plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]), color = "orangered", linewidth=2.5)
plt.title("PLS curve for Einstein Telescope")
#plt.axhline(AETmin(0)[0], color='r', linestyle='--')
plt.xlabel('f (Hz)')
plt.ylabel(r"$\Omega_{gw}$")
plt.grid(True)
plt.xscale('log')
plt.show()


plt.figure(figsize=(6, 9)) 
plt.loglog(fvalsET, sigETvals, color = "indigo", label = "Nominal", linewidth = 2.5)
plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]), color = "orangered", label = "PLS", linewidth = 2.5)
plt.legend(fontsize = 16)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.title("Nominal and PLS curve for ET", fontsize = 16)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.xlabel("f (Hz)", fontsize = 16)
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/ETnomandPLS.png', bbox_inches='tight')
plt.show()
#%%
P = 12
A = 3


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



#%%
freqvals = np.logspace(elminL, elmaxL, itera)   
sigvals = np.array(list(map(Ohms, freqvals)))

#%%
#Now finding the PLS for LISA
def Almin(nt):
    integrand = lambda f, nt:((f/fLisa)**(nt)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-3), args=(nt))[0]
    I2 = quad(integrand, 10**(-3), 10**(0), args=(nt))[0]
    res = snr5/np.sqrt(2*T*sum((I1,I2)))
    return nt, 2*res


#determining the envelop of the functions that bound Omega_gw for all the values of nt

ntvals = np.arange(ntmin, ntmax, step)
#we can now formulate these into a table by mapping these values together
#We now want to map our values of nt and Amin we can do this using the list() and map() operations in python
#map computes the function Amin with the iterable values of ntvals list turns the mapped values into a list, and array turns these
#into an array - have to do it this way can't just do map -> array


Aminvals = np.array(list(map(Almin, ntvals)))

#We vertically stack the two arrays together and transpose them to get (41,2) dim array
#which is the table in mathematica.
Atab = Aminvals[:,(0,1)]
#Our Ftab is slightly different from the mathematica script as they have the table
def Ftab(i, j):
    res = Atab[j,1]*(10**elLISA[i]/fLisa)**Atab[j,0]
    return res




elstep = (elmaxL-elminL)/itera
elLISA = np.arange(elminL, elmaxL+elstep, elstep)
i = range(len(elLISA))
j = range(len(Atab))
coordsl1 = np.array(np.meshgrid(i, j)).T.reshape(-1,2)

FtabLISA = np.array(list(map(lambda args: Ftab(*args), coordsl1))).reshape(len(elLISA), len(Atab))
maxed = []

def maxtablisa(i):
    maxed = np.log(np.max(FtabLISA[i]))
    return maxed
   
maxposlisa = range(len(FtabLISA))
maxplvals = np.array(list(map(maxtablisa, maxposlisa)))
maxpls = maxplvals
flogom = np.vstack((np.log(10**elLISA), maxpls)).T

#%%
plt.figure(figsize=(6, 9)) 
plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]), color = "orangered", linewidth=2.5)
# plt.axhline(Almin(0)[1], color='r', linestyle='--')
plt.xlabel('f (Hz)')
plt.ylabel(r"$\Omega_{gw}$")
plt.title('PLS curve LISA')
plt.grid(True)
plt.xscale('log')
plt.show()

#%%
plt.figure(figsize=(6, 9)) 
plt.loglog(freqvals, sigvals, color = "indigo", label = "Nominal", linewidth=2.5)
plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]), color = "orangered", label = "PLS", linewidth=2.5)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.legend(fontsize = 16)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.grid(True)
plt.xlabel("f (Hz)", fontsize = 16)
plt.title("Nominal and PLS curve LISA", fontsize = 16)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/LISAnomPLS.png', bbox_inches='tight')
plt.show()



#%%
fstar = 1.1
ntmin = -9/2
ntmax = 9/2

def Amincomb(nt):
    integrand = lambda f, nt:((f/fstar)**(nt)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-4), args=(nt))[0]
    I2 = quad(integrand, 10**(-4), 10**(0), args=(nt))[0]
    I3 = quad(integrand, 10**(0), 10, args=(nt), limit = 1000)[0]
    I4 = quad(integrand, 10, 100, args = (nt), limit = 10000)[0]
    I5 = quad(integrand, 100, ffmax, args=(nt), limit = 10000)[0]
    integrand2 = lambda f, nt:((f/fstar)**(nt)/sigp(f))**2
    I6 = quad(integrand2, ffmin, 10**(0), args=(nt))[0]
    I7 = quad(integrand2, 10**(0), 100, args=(nt))[0]
    I8 = quad(integrand2, 100, ffmax, args=(nt))[0]
    res = snr5/np.sqrt(2*T*sum((I1,I2,I3,I4,I5,I6,I7,I8)))
    return res    

Amin3 = []
ntcombvals = np.linspace(ntmin, ntmax, itera)
Amin3 = np.array(list(map(Amincomb, ntcombvals)))


Atab3 = np.vstack((ntcombvals, Amin3)).T


def ftab3(i, j):
    res = Atab3[j,1]*(10**combel[i]/fstar)**Atab3[j,0]
    return res

elmin = np.log10(ffmin)
elmax = np.log10(ffmax)

combel = np.linspace(elmin, elmax, itera)
i = range(len(combel))
j = range(len(Atab3))
coordsc1 = np.array(np.meshgrid(i, j)).T.reshape(-1,2)

Ftabcomb = np.array(list(map(lambda args: ftab3(*args), coordsc1))).reshape(len(combel), len(Atab3))


combmaxed = []
def maxtabcomb(i):
    combmaxed = np.log(np.max(Ftabcomb[i]))
    return combmaxed

maxposco = range(len(Ftabcomb))
maxcompls = np.array(list(map(maxtabcomb, maxposco)))
flogomcomb = np.vstack((np.log(10**combel), maxcompls)).T


plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]), color = "orangered", linewidth=2.5)
plt.xscale('log')
plt.xlim(ffmin, ffmax) 
plt.xlabel('f (Hz)')
plt.ylabel(r"$\Omega_{gw}$")
plt.title('Combined PLS curve for LISA and ET')
plt.grid(True)
plt.show()


plt.figure(figsize=(6, 9))
plt.loglog(fvalsET, sigETvals, color = "indigo", label = "Nominal", linewidth = 2.5)
plt.loglog(freqvals, sigvals, color = "indigo", label = "Nominal", linewidth=2.5)
plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]), ':',color = "teal", label = 'LISA PLS', linewidth=2.5)
plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]), ':',color = "black", label = 'ET PLS', linewidth=2.5)
plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]), color = "orangered", label = "Combined PLS", linewidth=2.5)
plt.title("Nominal and PLS curves ", fontsize = 16)
plt.legend(loc = (1.05,0.5), fontsize = 14)
plt.grid(True) 
plt.xlim(ffmin, ffmax) 
plt.xlabel(r'$f$ (Hz)', fontsize = 16)
plt.ylabel(r'$\Omega_{gw}$', fontsize = 16)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/CombineNomPLSwold.png', bbox_inches='tight')
plt.show()








