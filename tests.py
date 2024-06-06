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
P = 12
A = 3
alpha = -11.352

def P_acc(f):
    res = A**2 *(1e-15)**2 * (1+(0.4e-3 / f)**2)*(1+(f/8e-3)**4)*(2*pi*f)**(-4)*(2*pi*f/c)**2
    return res

def P_ims(f):
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
def pls(f):
    n = -0.1
    om = 1e-12
    fst = 1e-10
    res = om * (f/fst)**(n)
    return res
#Now finding the PLS for LISA
def Almin(nt):
    integrand = lambda f, nt:((f/fLisa)**(nt)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-3), args=(nt))[0]
    I2 = quad(integrand, 10**(-3), 10**(0), args=(nt))[0]
    res = snr5/np.sqrt(2*T*sum((I1,I2)))
    return nt, res


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

freqvals = np.logspace(elminL, elmaxL, itera)   
sigvals = np.array(list(map(Ohms, freqvals)))

pl = np.array(list(map(pls, freqvals)))
plt.figure(figsize=(6, 9)) 
plt.loglog(freqvals, sigvals, color = "indigo", label = "Approximate", linewidth=2.5)
plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]), color = "orangered", linewidth=2.5)
plt.plot(freqvals, pl, color = "black")
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.grid(True)
plt.legend(fontsize = 16)
plt.xlabel("f (Hz)", fontsize = 16)
plt.title("Nominal Sensitivity curve of LISA", fontsize = 16)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/LISAnoms.png', bbox_inches='tight')
plt.show()

#%%

def sigp(f):
    f0 = 1
    t1 = (1-0.475*np.exp(-(f/f0-25)**2/50))
    t2 = (1-5e-4*np.exp(-(f/f0-20)**2/100))
    t3 = (1-0.2*np.exp(-((f/f0-47)**2)**0.85/100))
    t4 = (1-0.1*np.exp(-((f/f0-50)**2)**0.7/100)-0.2*np.exp(-(f/f0-45)**2/250)+0.15*np.exp(-(f/f0-85)**2/400))
    res = 0.88*((9*(f/f0)**(-30)+5.5e-6 *(f/f0)**(-4.5)+0.28e-11 * (f/f0)**3.2)*(1/2 - 1/2*np.tanh(0.06*(f/f0-42)))
                +(1/2*np.tanh(0.06*(f/f0-42)))*(0.01e-11*(f/f0)**1.9 + 20e-13 *(f/f0)**2.8))*t1*t2*t3*t4

    return res

def etnomonly(f):
    res = sigp(f)
    if res > 10**(-5):
        return
    return res

#%%
fvalsET = np.logspace(0, 3,200)#frequency values
sigETvals = np.array(list(map(etnomonly, fvalsET)))

def pls(f):
    n = -0.1
    om = 1.5e-11
    fst = 1e-2
    res = om * (f/fst)**(n)
    return res

def AETmin(nt):
    integrand = lambda f, nt:(((f/fetstar)**(nt))/sigp(f))**2
    I1 = quad(integrand, 1.6, 100, args=(nt))[0]
    I2 = quad(integrand, 100, 445, args = (nt))[0]
    res = snr5/np.sqrt(2*T*sum((I1, I2)))
    return res

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


plet = np.array(list(map(pls, fvalsET)))

plt.figure(figsize=(6, 9)) 
plt.title("Nominal Sensitivity curve of ET", fontsize = 16)
plt.loglog(fvalsET, sigETvals, '-',label = "Approximate", linewidth = 2.5,color = "indigo" )
plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]), color = "orangered", linewidth=2.5)
plt.plot(fvalsET, plet, color = "black")
plt.legend(fontsize = 16)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.xlabel("f (Hz)", fontsize = 16)
#plt.ylim(10**(-9), 10**(-5))
plt.yscale('log')
plt.xscale('log')
# plt.xlim(10**0, 400)
plt.grid(True)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/ETnoms.png', bbox_inches='tight')
plt.show()

#%%
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
#combine pls curves
fstar = 1.1
ntmin = -9/2
ntmax = 9/2

def Amincomb(nt):
    integrand = lambda f, nt:((f/fstar)**(nt)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-4), args=(nt))[0]
    I2 = quad(integrand, 10**(-4), 10**(0), args=(nt))[0]
    I3 = quad(integrand, 10**(0), 10, args=(nt))[0]
    I4 = quad(integrand, 10, ffmax, args = (nt))[0]
    integrand2 = lambda f, nt:((f/fstar)**(nt)/sigp(f))**2
    I5 = quad(integrand2, ffmin, 10**(0), args=(nt))[0]
    I6 = quad(integrand2, 10**(0), 100, args=(nt))[0]
    I7 = quad(integrand2, 100, ffmax, args=(nt))[0]
    res = snr5/np.sqrt(2*T*sum((I1,I2,I3,I4,I5,I6,I7)))
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

# plt.figure(figsize=(6, 9))
# plt.loglog(otog[:,0], nom , color = "indigo", label = "Nominal", linewidth=2.5)
# plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]), color = "orangered", label = "Combined PLS", linewidth=2.5)
# plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]), ':',color = "teal", label = 'LISA PLS', linewidth=2.5)
# plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]), ':',color = "black", label = 'ET PLS', linewidth=2.5)
# plt.title("Nominal and PLS curves ", fontsize = 16)
# plt.legend(loc = (1.05,0.5), fontsize = 14)
# plt.grid(True) 
# plt.xlim(ffmin, ffmax) 
# plt.xlabel(r'$f$ (Hz)', fontsize = 16)
# plt.ylabel(r'$\Omega_{gw}$', fontsize = 16)
# plt.show()

#%%

def bpls(f):
    om = 5e-10
    n1 = 2.4
    n2 = -2.4
    fstar = 2e-1
    s = 1.2
    res = om*(f/fstar)**n1 * (1/2+(1/2)*(f/fstar)**s)**(-(n1-n2)/s)
    return res

bplet = np.array(list(map(bpls, fvalscomb)))
# phase = np.array(list(map(bpl, freqs)))
et = np.load('/Users/alisha/Documents/LISA/ftabET.npy')
lisa = np.load('/Users/alisha/Documents/LISA/ftablisa.npy')
plt.figure(figsize=(6, 9))
plt.loglog(otog[:,0], nom , color = "indigo", label = "Nominal", linewidth=2.5)
plt.plot(fvalscomb,bplet)
plt.plot(np.exp(et[:,0]), np.exp(et[:,1]))
plt.plot(np.exp(lisa[:,0]),np.exp(lisa[:,1]))
plt.title("Nominal and PLS curves ", fontsize = 16)
# plt.legend(loc = (1.05,0.5), fontsize = 14)
plt.grid(True) 
plt.xlim(ffmin, ffmax) 
plt.xlabel(r'$f$ (Hz)', fontsize = 16)
plt.ylabel(r'$\Omega_{gw}$', fontsize = 16)
plt.show()













