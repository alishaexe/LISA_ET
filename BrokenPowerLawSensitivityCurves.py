import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
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

itera = 1000
nitera = 10
##########

elminL = (np.log10(ffmin))
elmaxL = (np.log10(10**(-0.95)))
elminet = np.log10(1.6)
elmaxet = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera
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

freqvals = np.logspace(elminL, elmaxL, 200)   
sigvals = np.array(list(map(Ohms, freqvals)))

plt.loglog(freqvals, sigvals)
#%%
#Now for BPL

def bpl(f, fstar, n1, n2):
    s = 10
    res = (f/fstar)**n1 * (1/2+1/2*(f/fstar)**s)**(-(n1-n2)/s)
    return res


def Abplmin(fs, n1, n2):
    integrand = lambda f, n1, n2, fs:(bpl(f, fs, n1, n2)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-3), args=(n1, n2, fs))[0]
    I2 = quad(integrand, 10**(-3), 10**(-1), args=(n1, n2, fs))[0]
    res = snr5/np.sqrt(2*T*sum((I1,I2)))
    return res


elstep = (elmaxL-elminL)/itera
elbpl = np.arange(elminL, elmaxL, elstep)
n1r = np.linspace(ntmin, ntmax, nitera)
n2r = np.linspace(ntmin, ntmax, nitera)
fs = 10**elbpl

inputs = np.array(np.meshgrid(fs, n2r, n1r)).T.reshape(-1,3)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputs[:,[0,1,2]] = inputs[:,[0,2,1]]

#%%

Amin2 = np.array(list(map(lambda args: Abplmin(*args), inputs)))
Atab2 = np.vstack((inputs.T, Amin2)).T.reshape(len(fs),len(n1r),len(n2r),4)

#%%
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

Ftab2 = np.array(list(map(lambda args: fbpltab(*args), coords))).reshape(len(fs),len(n1r),len(n2r),len(fs))

#%%
maxims = []
def maxbplvals(i):
    maxims = np.log(np.max(Ftab2[i]))
    return maxims

maxpos = range(len(Ftab2))

maxbpl = np.array(list(map(maxbplvals, maxpos)))
#%%
# fbplo = np.vstack((np.log(fs), maxbpl)).T
# np.save("ftablisa.npy", fbplo)
fbplo = np.load("/Users/alisha/Documents/LISA_ET/Datafiles/ftablisa.npy")
plt.figure(figsize=(6, 9))
plt.loglog(freqvals, sigvals, label = "Nominal Curve", color = "indigo", linewidth=2.5)
plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]), label = "BPLS curve", color = "lime", linewidth=2.5)

plt.grid(True)
plt.title("LISA BPLS Curves")
plt.legend()
plt.xlabel('f (Hz)')
plt.ylabel(r'$\Omega_{gw}$', fontsize = 12)
plt.xscale('log')
plt.ylim(1e-14, 1e-4)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/LISAnomBPLS.png', bbox_inches='tight')
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

fvalsET = np.logspace(0, 3,itera)#frequency values
sigETvals = np.array(list(map(etnomonly, fvalsET)))
#%%
#Now for BPL
def bplET(f, fstar, n1, n2):
    s = 10
    res = (f/fstar)**n1 * (1/2+(1/2)*(f/fstar)**s)**(-(n1-n2)/s)
    return res

def AbplminET(fs, n1, n2):
    integrand = lambda f, fs, n1, n2:(bplET(f, fs, n1, n2)/sigp(f))**2
    I1 = quad(integrand, 1.6, 100, args=(fs, n1, n2))[0]
    I2 = quad(integrand, 100, 445, args=(fs, n1, n2))[0]
    res = snr5/np.sqrt(2*T*sum((I1, I2)))
    return res

n1r = np.linspace(ntmin, ntmax, itera)
n2r = np.linspace(ntmin, ntmax, itera)
xelstep = (elmaxet-elminet)/itera
elET = np.arange(elminet, elmaxet, xelstep)
fs = 10**elET

inputsET = np.array(np.meshgrid(fs, n2r, n1r)).T.reshape(-1,3)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputsET[:,[0,1,2]] = inputsET[:,[0,2,1]]

AminET = np.array(list(map(lambda args: AbplminET(*args), inputsET)))
AtabET = np.vstack((inputsET.T, AminET)).T.reshape(len(n1r),len(n1r),len(n1r),4)
#%%
i = range(len(fs))
j = range(len(n1r))
k = range(len(n2r))
m = range(len(fs))
coords = np.array(np.meshgrid(i, j, k, m)).T.reshape(-1,4)


def fbpltabET(i, j, k, m):
    bplres = bplET(fs[m], AtabET[i,j,k,0], AtabET[i,j,k,1], AtabET[i,j,k,2])
    return AtabET[i,j,k,3]*bplres

   
FtabET = np.array(list(map(lambda args: fbpltabET(*args), coords))).reshape(len(fs),len(n1r),len(n2r),len(fs))

def maxETbplvals(i):
    maximsET = np.log(np.max(FtabET[i]))
    return maximsET

maximsET = []
maxposET = range(len(FtabET))
maxbplvals = np.array(list(map(maxETbplvals, maxposET)))
maxbplET = maxbplvals

#%%
fbploET = np.vstack((np.log(fs), maxbplET)).T

# np.save("ftabET.npy", fbploET)
#plots the bpl curve
plt.figure(figsize=(6, 9)) 
plt.loglog(np.exp(fbploET[:,0]), np.exp(fbploET[:,1]), '-',color = "lime", linewidth=2.5)
plt.title("BPLS Curve ET")
plt.xlabel('f (Hz)')
plt.ylabel(r"$\Omega_{gw}$")
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()


#%%
#plots all 3 graphs on same plot
plt.figure(figsize=(6, 9)) 
plt.loglog(fvalsET, sigETvals, color = "indigo",linewidth=2.5, label = "Nominal")
# plt.loglog(np.exp(fbploET[:,0]), np.exp(fbploET[:,1]), linewidth=2.5,color = "lime", label = "BPLS")
plt.title('ET Nominal and BPL Curves')
plt.ylabel(r"$\Omega_{gw}$")
plt.legend(fontsize="10", loc = 'upper center')
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/ETBPLScurves.png', bbox_inches='tight')
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

plt.figure(figsize=(6, 9))
plt.loglog(otog[:,0], nom, color = "indigo", linewidth=2.5)
plt.title("Nominal curves for ET and LISA")
plt.grid(True) 
plt.xlim(ffmin, ffmax) 
plt.xlabel('f (Hz)')
plt.ylabel(r"$\Omega_{gw}$")
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/ETLISAnom.png', bbox_inches='tight')
plt.show()

#%%
#bpls
def combbpl(f, fstar, n1, n2):
    s = 10
    res = (f/fstar)**n1 * (1/2+1/2*(f/fstar)**s)**(-(n1-n2)/s)
    return res 


def Abplmincomb(fs, n1, n2):
    integrand = lambda f, fs, n1, n2:(combbpl(f, fs, n1, n2)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-3), args=(fs, n1, n2))[0]
    I2 = quad(integrand, 10**(-3), 10**(-1), args=(fs, n1, n2))[0]
    integrand2 = lambda f, fs, n1, n2:(combbpl(f, fs, n1, n2)/sigp(f))**2
    I3 = quad(integrand2, 1.6, 10**(1), args=(fs, n1, n2))[0]
    I4 = quad(integrand2, 10**(1), 445, args=(fs, n1, n2))[0]
    res = snr5/np.sqrt(2*T*sum((I1,I2,I3,I4)))
    return res   

step = (ntmax-ntmin)/itera

elc = np.linspace(elmin, elmax, itera)

n1c = np.arange(ntmin, ntmax, step)
n2c = np.arange(ntmin, ntmax, step)

fsc = 10**elc



inputsc = np.array(np.meshgrid(fsc, n2c, n1c)).T.reshape(-1,3)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputsc[:,[0,1,2]] = inputsc[:,[0,2,1]]


Amin4 = np.array(list(map(lambda args: Abplmincomb(*args), inputsc)))
#%%
Atab4 = np.vstack((inputsc.T, Amin4)).T
Atab4 = Atab4.reshape(len(fsc),len(n1c),len(n1c),4)
ic = range(len(fsc))
jc = range(len(n1c))
kc = range(len(n2c))
mc = range(len(fsc))
coordsc = np.array(np.meshgrid(ic,jc,kc, mc)).T.reshape(-1,4)


def fbpltabcomb(i, j, k, m):
    res = []
    bplres = combbpl(fsc[m], Atab4[i,j,k,0], Atab4[i,j,k,1], Atab4[i,j,k,2])
    res.append([Atab4[i,j,k,3]*bplres])
    return res



Ftab4 = np.array(list(map(lambda args: fbpltabcomb(*args), coordsc))).reshape(len(fsc),len(n1c),len(n2c),len(fsc))
   

#%%
maximsc = []
def combmaxbplvals(i):
    maximsc = np.log(np.max(Ftab4[i]))
    return maximsc

combmaxpos = range(len(Ftab4))
maxbplcomb = np.array(list(map(combmaxbplvals, combmaxpos)))
combfbplo = np.vstack((np.log(fsc), maxbplcomb)).T



plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]), color = "lime", linewidth=2.5)
plt.title("BPLS Curve for LISA and ET")
plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
plt.yscale('log')
plt.xscale('log')
plt.xlim(ffmin, ffmax)
plt.grid(True)
plt.show()

#%%
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
plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/CombineNomBPLSwold.png', bbox_inches='tight')
plt.show()

np.save("Ftab4.npy", Ftab4)
np.save("Atab4.npy", Atab4)
np.save("Amin4.npy", Amin4)











