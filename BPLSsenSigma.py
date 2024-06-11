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
#ogs are 100 itera and 12 nitera
itera = 100
nitera = 12
##########

elminL = (np.log10(ffmin))
elmaxL = (np.log10(10**(-0.95)))
elminet = np.log10(1.6)
elmaxet = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera

sig1 = 1
sig2 = 12
sigstep = (sig2-sig1)/5
sigma = np.arange(sig1, sig2, sigstep)
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

freqvals = np.logspace(elminL, elmaxL, 200)   
sigvals = np.array(list(map(Ohms, freqvals)))

#%%
freqvals = np.logspace(elminL, elmaxL, itera)   
sigvals = np.array(list(map(Ohms, freqvals)))
elstep = (elmaxL-elminL)/itera
elLISA = np.arange(elminL, elmaxL, elstep)

#%%
# #Now for BPL
# def bpl(f, fstar, n1, n2, s):
#     res =  (f/fstar)**n1 * (1/2+(1/2)*(f/fstar)**s)**(-(n1-n2)/s)
#     return res


# def Abplmin(fs, n1, n2, s):
#     integrand = lambda f, n1, n2, fs, s:(bpl(f, fs, n1, n2, s)/Ohms(f))**2
#     I1 = quad(integrand, ffmin, 10**(-3), args=(n1, n2, fs, s))[0]
#     I2 = quad(integrand, 10**(-3), 10**(-1), args=(n1, n2, fs, s))[0]
#     res = snr5/np.sqrt(T*sum((I1,I2)))
#     return res

# elbpl = np.arange(elminL, elmaxL, elstep)
# n1r = np.linspace(ntmin, ntmax, itera)
# n2r = np.linspace(ntmin, ntmax, itera)
# fs = 10**elbpl
# inputs = np.array(np.meshgrid(sigma, n2r, n1r, fs)).T.reshape(-1,4)
# #This makes it so n1r is in the second column
# # so inputs(fs, n1r, n2r)
# inputs[:,[0,1,2,3]] = inputs[:,[3,2,1,0]]



# Amin2 = np.array(list(map(lambda args: Abplmin(*args), inputs)))
# Atab2 = np.vstack((inputs.T, Amin2)).T.reshape(len(fs),len(n1r),len(n2r), len(sigma),5)

# #%%
# i = np.array(range(len(fs)))#defining them as arrays here means that in
# j = np.array(range(len(n1r)))#the meshgrid they'll stay in order
# k = np.array(range(len(n2r)))#i.e 000, 001,002 etc
# m = np.array(range(len(sigma)))
# fLvals = 10**elbpl
# coords = np.array(np.meshgrid(m,k,j,i, fLvals)).T.reshape(-1,5)
# coords[:,[0,1,2,3,4]] = coords[:,[4,3,2,1,0]]  
# #here in meshgrid have done ikj purely because this may it will sort j
# #like it sorts i and lets k change; we then switch round the columns so that
# #it is i,j,k
# def fbpltab(f, i, j, k, m):
#     i,j,k,m = i.astype(int), j.astype(int), k.astype(int), m.astype(int)
#     bplres = bpl(f, Atab2[i,j,k,m,0], Atab2[i,j,k,m,1], Atab2[i,j,k,m,2],Atab2[i,j,k,m,3])
#     return Atab2[i,j,k,m,4]*bplres


# Ftab2 = np.array(list(map(lambda args: fbpltab(*args), coords))).reshape(len(Atab2),len(fs),len(n1r),len(n2r),len(sigma))

# #%%
# maxims = []
# def maxbplvals(i):
#     maxims = np.log(np.max(Ftab2[i]))
#     fh = np.log(fLvals[i])
#     return fh, maxims


# maxpos = range(len(Ftab2))
# maxbpl = np.array(list(map( maxbplvals, maxpos)))



# #%%
# fbplo = maxbpl#np.vstack((np.log(fs), maxbpl)).T

# # np.save("FtabbigsigLISA.npy", fbplo)
fbplo = np.load('/Users/alisha/Documents/LISA/FtabbigsigLISA.npy')
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
plt.savefig('LISABPLSbigsigma.png', bbox_inches='tight')
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
    return res
                                                                           
                                                                           

def etnomonly(f):
    res = sigp(f)
    if res > 10**(-5):
        return
    return res

#%%
fvalsET = np.logspace(np.log10(1), np.log10(445),2000)#frequency values
sigETvals = np.array(list(map(etnomonly, fvalsET)))#The Omega_gw values from the ET data
# #%%
# #Now the BPLS for ET
# #Now for BPL
# elstep = (elmaxet-elminet)/itera
# elET = np.arange(elminet, elmaxet, elstep)


# def bplET(f, fstar, n1, n2, s):
#     res = (f/fstar)**n1 * (1/2+(1/2)*(f/fstar)**s)**(-(n1-n2)/s)
#     return res

# def AbplminET(fs, n1, n2, s):
#     integrand = lambda f, fs, n1, n2, s:(bplET(f, fs, n1, n2,s)/sigp(f))**2
#     I1 = quad(integrand, 1.6, 100, args=(fs, n1, n2, s))[0]
#     I2 = quad(integrand, 100, 445, args=(fs, n1, n2, s))[0]
#     res = snr5/np.sqrt(T*sum((I1, I2)))
#     return res

# n1r = np.linspace(ntmin, ntmax, itera)
# n2r = np.linspace(ntmin, ntmax, itera)
# fs = 10**elET

# inputsET = np.array(np.meshgrid(sigma, n2r, n1r, fs)).T.reshape(-1,4)
# #This makes it so n1r is in the second column
# # so inputs(fs, n1r, n2r)
# inputsET[:,[0,1,2,3]] = inputsET[:,[3,2,1,0]]
# #%%
# AminET = np.array(list(map(lambda args: AbplminET(*args), inputsET)))
# AtabET = np.vstack((inputsET.T, AminET)).T.reshape(len(fs),len(n1r),len(n2r), len(sigma),5)
# i = range(len(fs))
# j = range(len(n1r))
# k = range(len(n2r))
# m = range(len(sigma))
# fvals = (10**elET)
# coords = np.array(np.meshgrid(m, k, j, i, fvals)).T.reshape(-1,5)
# coords[:,[0,1,2,3,4]] = coords[:,[4,3,2,1,0]]
# #%%
# def fbpltabET(f, i, j, k, m):
#     i,j,k,m = i.astype(int), j.astype(int), k.astype(int), m.astype(int)
#     bplres = bplET(f, AtabET[i,j,k,m,0], AtabET[i,j,k,m,1], AtabET[i,j,k,m,2], AtabET[i,j,k,m,3])
#     return AtabET[i,j,k,m,4]*bplres

   
# FtabET = np.array(list(map(lambda args: fbpltabET(*args), coords))).reshape(len(AtabET),len(fs),len(n1r),len(n2r),len(sigma))
# #%%
# def maxETbplvals(i):
#     maximsET = np.log(np.max(FtabET[i]))
#     fh = np.log(fvals[i])
#     return fh, maximsET


# maxposET = range(len(FtabET))
# maxbplvals = np.array(list(map(maxETbplvals, maxposET)))


# #%%

# fbploET = maxbplvals

# np.save("FtabbigsigET.npy", fbploET)
fbploET = np.load('/Users/alisha/Documents/LISA/FtabbigsigET.npy')
# # fbploET = np.load('FtabsigET.npy')
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
plt.savefig('ETBPLSbigsigma.png', bbox_inches='tight')


#%%
#######################
#Combining LISA and ET curves
#######################

def omegatog(f):
    if f <= 10**(-0.95):
        return Ohms(f)
    if f > 1.6:
        return sigp(f)
    
def nomtog(f):
    if f <= 10**(-0.95):
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
def combbpl(f, fstar, n1, n2,s):

    res = (f/fstar)**n1 * (1/2+(1/2)*(f/fstar)**s)**(-(n1-n2)/s)
    return res 


def Abplmincomb(fs, n1, n2, s):
    integrand = lambda f, fs, n1, n2, s:(combbpl(f, fs, n1, n2, s)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-4), args=(fs, n1, n2, s))[0]
    I2 = quad(integrand, 10**(-4), 10**(0), args=(fs, n1, n2, s))[0]
    I3 = quad(integrand, 10**(0), 10, args=(fs, n1, n2, s))[0]
    I4 = quad(integrand, 10, ffmax, args = (fs, n1, n2, s))[0]
    integrand2 = lambda f, fs, n1, n2, s:(combbpl(f, fs, n1, n2, s)/sigp(f))**2
    I5 = quad(integrand2, ffmin, 10**(0), args=(fs, n1, n2, s))[0]
    I6 = quad(integrand2, 10**(0), 100, args=(fs, n1, n2, s))[0]
    I7 = quad(integrand2, 100, ffmax, args=(fs, n1, n2, s))[0]
    res = snr5/np.sqrt(T*sum((I1,I2,I3,I4, I5, I6, I7)))
    return res   

elmin = np.log10(ffmin)
elmax = np.log10(ffmax)
step = (ntmax-ntmin)/itera

elc = np.linspace(elmin, elmax, itera)

n1c = np.arange(ntmin, ntmax, step)
n2c = np.arange(ntmin, ntmax, step)

fsc = 10**elc



inputsc = np.array(np.meshgrid(sigma, n2c, n1c, fsc)).T.reshape(-1,4)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputsc[:,[0,1,2,3]] = inputsc[:,[3,2,1,0]]

#%%

Amin4 = np.array(list(map(lambda args: Abplmincomb(*args), inputsc)))
Atab4 = np.vstack((inputsc.T, Amin4)).T.reshape(len(fsc),len(n1c),len(n2c), len(sigma),5)

#%%
ic = range(len(fsc))
jc = range(len(n1c))
kc = range(len(n2c))
mc = range(len(sigma))
fcvals = 10**elc
coordsc = np.array(np.meshgrid(mc,kc,jc,ic, fcvals)).T.reshape(-1,5)
coordsc[:,[0,1,2,3,4]] = coordsc[:,[4,3,2,1,0]]  
#%%
def fbpltabcomb(f, i, j, k, m):
    i,j,k,m = i.astype(int), j.astype(int), k.astype(int), m.astype(int)
    bplres = combbpl(f, Atab4[i,j,k,m,0], Atab4[i,j,k,m,1], Atab4[i,j,k,m,2],Atab4[i,j,k,m,3])
    return Atab4[i,j,k,m,4]*bplres



Ftab4 = np.array(list(map(lambda args: fbpltabcomb(*args), coordsc))).reshape(len(Atab4),len(fsc),len(n1c),len(n2c), len(sigma))
   

#%%
maximsc = []
def combmaxbplvals(i):
    maximsc = np.log(np.max(Ftab4[i]))
    fh = np.log(fcvals[i])
    return fh, maximsc

combmaxpos = range(len(Ftab4))
maxbplcomb = np.array(list(map(combmaxbplvals, combmaxpos)))
combfbplo = maxbplcomb
# np.save("Ftabbigsigcomb.npy", combfbplo)

#%%
combfbplo = np.load('/Users/alisha/Documents/LISA/Ftabbigsigcomb.npy')
plt.figure(figsize=(6, 9))
plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]), label = "BPLS curve", color = "lime", linewidth=2.5)
plt.grid(True)
plt.title("Combined BPLS Curves", fontsize = 16)
plt.legend(fontsize = 16)
plt.xlabel('f (Hz)', fontsize = 16)
plt.ylabel(r'$\Omega_{gw}$', fontsize = 16)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.xscale('log')
# plt.savefig('CombBPLSbigsigma.png', bbox_inches='tight')

# # #%%
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
# plt.savefig('CombineNomBPLSsigma.png', bbox_inches='tight')
 
