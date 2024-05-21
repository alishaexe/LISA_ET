import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
#%%
yr = 365*24*60*60 #in seconds
H0 = 100*0.67*10**(3)/(3.086*10**(22)) #1/seconds
#setting h = 0.67
pi = np.pi

#For the LISA mission they have designed the
#arms to be length L = 2.5*10^(6)
T = 3*yr
snr5 = 5
#L = 2.5e9
L = 25/3

c = 3e8

fLisa = 1/(2*pi*L)
ffmin = 10**(-5)
ffmax = 445
###############
#Change this value for how many 'steps' you want in the range of values

itera = 10

##########
elminL = (np.log10(ffmin))
elmaxL = (np.log10(10**(-1)))
elminet = np.log10(1.6)
elmaxet = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera



###################
sig1 = 1
sig2 = 12
sigstep = (sig2-sig1)/60
sigma = np.arange(sig1, sig2, sigstep)
#%%
P = 12
A = 3
alpha = -11.352
#Speed of light c = 1
# *(1e-15)**2 after A
def P_acc(f):
    res = A**2 *(1e-15)**2 * (1+(0.4e-3 / f)**2)*(1+(f/8e-3)**4)*(2*pi*f)**(-4)*(2*pi*f/c)**2
    return res

def P_oms(f):#* (1e-12)**2 after P
    res = P**2 * (1e-12)**2 *(1+(2e-3/f)**4)*(2*pi*f/c)**2
    return res

def N_aa(f):
    res = 16*(np.sin(2*pi*f*L))**2 * (P_oms(f)+(3+np.cos(4*pi*f*L))*P_acc(f))
    return res

def R(f):
    res = 16*(np.sin(2*pi*f*L))**2 * 9/20 * 1/(1+0.7*(2*pi*f*L)**2) * (2*pi*f*L)**2
    return res

def S_n(f):
    res = N_aa(f)/R(f)
    return res

def Ohms(f):
    const = 4*pi**2/(3*H0**2)
    res = const *f**3*S_n(f)
    return res
#%%

#Now for BPL
def bpl(f, fstar, n1, n2):
    s = 10
    res =  (f/fstar)**n1 * (1/2+(1/2)*(f/fstar)**s)**(-(n1-n2)/s)
    return res


def Abplmin(fs, n1, n2):
    integrand = lambda f, n1, n2, fs:(bpl(f, fs, n1, n2)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-3), args=(n1, n2, fs))[0]
    I2 = quad(integrand, 10**(-3), 10**(-1), args=(n1, n2, fs))[0]
    res = snr5/np.sqrt(T*sum((I1,I2)))
    return res


elstep = (elmaxL-elminL)/itera
elLISA = np.arange(elminL, elmaxL+elstep, elstep)


n1r = np.linspace(ntmin, ntmax, itera)
n2r = np.linspace(ntmin, ntmax, itera)
fs = 10**elLISA
inputs = np.array(np.meshgrid(fs, n2r, n1r)).T.reshape(-1,3)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputs[:,[0,1,2]] = inputs[:,[0,2,1]]

#%%

Amin2 = np.array(list(map(lambda args: Abplmin(*args), inputs)))
Atab2 = np.vstack((inputs.T, Amin2)).T.reshape(len(fs),len(n1r),len(n2r),4)

#%%
fLvals = 10**elLISA
i = np.array(range(len(fs)))#defining them as arrays here means that in
j = np.array(range(len(n1r)))#the meshgrid they'll stay in order
k = np.array(range(len(n2r)))#i.e 000, 001,002 etc
m = np.array(range(len(fs)))
coords = np.array(np.meshgrid(i,j,k,m)).T.reshape(-1,4)
#here in meshgrid have done ikj purely because this may it will sort j
#like it sorts i and lets k change; we then switch round the columns so that
#it is i,j,k
def fbpltab(i, j, k, m):
    bplres = bpl(fs[m], Atab2[i,j,k,0], Atab2[i,j,k,1], Atab2[i,j,k,2])
    return Atab2[i,j,k,3]*bplres

Ftab2 = np.array(list(map(lambda args: fbpltab(*args), coords))).reshape(len(Atab2),len(fs),len(n1r),len(n2r))
#%%
maxims = []
def maxbplvals(i):
    maxims = np.log(np.max(Ftab2[i]))
    fh = np.log(fLvals[i])
    return fh, maxims

def SigmaLisaApproxnom(f):#Sigma_Ohm approx
    res = Ohms(f)
    if res >1e-5:
        return
    return res
freqvals = np.logspace(elminL, elmaxL, itera)   
sigvals = np.array(list(map(SigmaLisaApproxnom, freqvals)))

maxpos = range(len(Ftab2))
maxbpl = np.array(list(map( maxbplvals, maxpos)))
#%%
fbplo = maxbpl#np.vstack((np.log(fs), maxbpl)).T

plt.figure(figsize=(6, 9))
plt.loglog(freqvals, sigvals, label = "Nominal Curve", color = "indigo", linewidth=2.5)
plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]), label = "BPLS curve", color = "lime", linewidth=2.5)
plt.grid(True)
plt.title("LISA BPLS Curves", fontsize = 16)
plt.legend(fontsize = 16)
plt.xlabel('f (Hz)', fontsize = 16)
plt.ylabel(r'$\Omega_{gw}$', fontsize = 16)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.xscale('log')


#%%
########################
#ET
########################
def sigp(f):
    res = 0.9 * ((3 * 30 * 10**(-1) * f**(-30) + 5.5 * 10**(-6) * f**(-4.5) + 
            0.7 * 10**(-11) * f**(2.8)) * (1/2 - 
            1/2 * np.tanh(0.04 * (f - 42))) + (1/2 * np.tanh(0.04 * (f - 42))) * 
            (0.4 * 10**(-11) * f**(1.4) + 7.9 * 10**(-13) * f**(2.98))) * (1 - 
            0.38 * np.exp(-(f - 25)**2/50))
    return res


#%%
#Now the BPLS for ET
#Now for BPL
elstep = (elmaxet-elminet)/itera
elET = np.arange(elminet, elmaxet+elstep, elstep)


def bplET(f, fstar, n1, n2):
    s = 10
    res = (f/fstar)**n1 * (1/2+(1/2)*(f/fstar)**s)**(-(n1-n2)/s)
    return res

def AbplminET(fs, n1, n2):
    integrand = lambda f, fs, n1, n2:(bplET(f, fs, n1, n2)/sigp(f))**2
    I1 = quad(integrand, 1.6, 100, args=(fs, n1, n2))[0]
    I2 = quad(integrand, 100, 445, args=(fs, n1, n2))[0]
    res = snr5/np.sqrt(T*sum((I1, I2)))
    return res

n1r = np.linspace(ntmin, ntmax, itera)
n2r = np.linspace(ntmin, ntmax, itera)
fs = 10**elET

inputsET = np.array(np.meshgrid(fs, n2r, n1r)).T.reshape(-1,3)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputsET[:,[0,1,2]] = inputsET[:,[0,2,1]]

AminET = np.array(list(map(lambda args: AbplminET(*args), inputsET)))
AtabET = np.vstack((inputsET.T, AminET)).T.reshape(len(fs),len(n1r),len(n2r),4)
#%%
fvals = (10**elET)
i = range(len(fs))
j = range(len(n1r))
k = range(len(n2r))
m = range(len(fs))
coords = np.array(np.meshgrid(i, j, k, m)).T.reshape(-1,4)

#%%
def fbpltabET(i, j, k, m):
    bplres = bplET(fs[m], AtabET[i,j,k,0], AtabET[i,j,k,1], AtabET[i,j,k,2])
    return AtabET[i,j,k,3]*bplres

   
FtabET = np.array(list(map(lambda args: fbpltabET(*args), coords))).reshape(len(AtabET),len(fs),len(n1r),len(n2r))

#%%
def maxETbplvals(i):
    maximsET = np.log(np.max(FtabET[i]))
    return maximsET


maximsET = []
maxposET = range(len(FtabET))
maxbplvals = np.array(list(map(maxETbplvals, maxposET)))
maxbplET = maxbplvals

#%%
fbploET = np.vstack((np.log(fs), maxbplET)).T
def etnomonly(f):
    res = sigp(f)
    if res > 10**(-5):
        return
    return res
fvalsET = np.logspace(np.log10(1), np.log10(445),2000)#frequency values
sigETvals = np.array(list(map(etnomonly, fvalsET)))#The Omega_gw va

#%%
#plots all 3 graphs on same plot
plt.figure(figsize=(6, 9)) 
plt.loglog(fvalsET, sigETvals, color = "indigo",linewidth=2.5, label = "Nominal")
plt.loglog(np.exp(fbploET[:,0]), np.exp(fbploET[:,1]), linewidth=2.5,color = "lime", label = "BPLS")
plt.title('ET Nominal and BPL Curves', fontsize = 16)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.xlabel('f (Hz)', fontsize = 16)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.legend(fontsize="16", loc = 'upper center')
plt.grid(True)
plt.xscale('log')
plt.yscale('log')

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

#%%
#bpls
def combbpl(f, fstar, n1, n2):
    s=10
    res = (f/fstar)**n1 * (1/2+(1/2)*(f/fstar)**s)**(-(n1-n2)/s)
    return res 


def Abplmincomb(fs, n1, n2):
    integrand = lambda f, fs, n1, n2:(combbpl(f, fs, n1, n2)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-4), args=(fs, n1, n2))[0]
    I2 = quad(integrand, 10**(-4), 10**(0), args=(fs, n1, n2))[0]
    I3 = quad(integrand, 1e-1, 10, args=(fs, n1, n2), limit = 5000)[0]
    I4 = quad(integrand, 10, 1e2, args=(fs, n1, n2), limit = 5000)[0]
    I5 = quad(integrand, 1e2, ffmax, args=(fs, n1, n2), limit = 5000)[0]
    integrand2 = lambda f, fs, n1, n2: (combbpl(f, fs, n1, n2)/sigp(f))**2
    I6 = quad(integrand2, ffmin, 10**(0), args=(fs, n1, n2))[0]
    I7 = quad(integrand2, 10**(0), 100, args=(fs, n1, n2))[0]
    I8 = quad(integrand2, 100, ffmax, args=(fs, n1, n2))[0]
    res = snr5/np.sqrt(T*sum((I1,I2,I3,I4,I5,I6,I7,I8)))
    return res       

elmin = np.log10(ffmin)
elmax = np.log10(ffmax)
step = (ntmax-ntmin)/itera
elc = np.arange(elmin, elmax+step, step)

n1c = np.arange(ntmin, ntmax, step)
n2c = np.arange(ntmin, ntmax, step)

fsc = 10**elc



inputsc = np.array(np.meshgrid(fsc, n2c, n1c)).T.reshape(-1,3)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputsc[:,[0,1,2]] = inputsc[:,[0,2,1]]

#%%
Amin4 = np.array(list(map(lambda args: Abplmincomb(*args), inputsc)))
Atab4 = np.vstack((inputsc.T, Amin4)).T.reshape(len(fsc),len(n1c),len(n2c),4)

ic = range(len(fsc))
jc = range(len(n1c))
kc = range(len(n2c))
mc = range(len(fsc))
coordsc = np.array(np.meshgrid(ic,jc,kc,mc)).T.reshape(-1,4)
fcvals = 10**elc
#%%
def fbpltabcomb(i, j, k, m):
    bplres = combbpl(fsc[m], Atab4[i,j,k,0], Atab4[i,j,k,1], Atab4[i,j,k,2])
    return AtabET[i,j,k,3]*bplres


Ftab4 = np.array(list(map(lambda args: fbpltabcomb(*args), coordsc))).reshape(len(Atab4),len(fsc),len(n1c),len(n2c))
   

#%%
maximsc = []
def combmaxbplvals(i):
    maximsc = np.log(np.max(Ftab4[i]))
    fh = np.log(fcvals[i])
    return fh, maximsc

combmaxpos = range(len(Ftab4))
maxbplcomb = np.array(list(map(combmaxbplvals, combmaxpos)))
combfbplo = maxbplcomb


#%%
plt.figure(figsize=(6, 9))
plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]), label = "BPLS curve", color = "lime", linewidth=2.5)
plt.grid(True)
plt.title("Combined BPLS Curves", fontsize = 16)
plt.legend(fontsize = 16)
plt.xlabel('f (Hz)', fontsize = 16)
plt.ylabel(r'$\Omega_{gw}$', fontsize = 16)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.xscale('log')


# #%%
plt.figure(figsize=(6, 9))
plt.loglog(otog[:,0], nom, label = "Nominal Curve", color = "indigo", linewidth=2.5)
plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]), label = "Combined BPLS", color = "lime", linewidth=2.5)
plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]),':', color = "teal", label = 'LISA BPLS', linewidth=2.5)
plt.loglog(np.exp(fbploET[:,0]), np.exp(fbploET[:,1]), ':',color = "black", label = " ET BPLS", linewidth=2.5)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.title("Nominal and BPLS Curve for LISA and ET", fontsize = 16)
plt.xlabel("f (Hz)", fontsize = 16)
plt.legend(loc = (1.05,0.5), fontsize = 14)
plt.yscale('log')
plt.xscale('log')
plt.xlim(ffmin, ffmax)
plt.grid(True)











