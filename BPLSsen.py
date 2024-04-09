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

itera = 10

##########
elminL = (np.log10(ffmin))
elmaxL = (np.log10(10**(-1)))
elminet = np.log10(1.6)
elmaxet = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera



#%%
#function for creating the table of the ET data
#we use i to iterate through each row of the data
def tabETapp(i):
    res = (4*pi**2 / (3*H0**2))*Etab[i,0]**3 * Etab[i,3]**2
    return res

col1 = Etab[:,0] # assigning the first column of the data to the name col1
vals = range(len(Etab))#this makes the list of numbers to iterate through for i
col2 = np.array(list(map(tabETapp, vals)))
#map takes the values from vals and inputs them into the tabETapp function, the list() 
#formats the output from the function for each of these values
#as a list, and np.array() then turns them into an array. This is all assigned to col2
tabET = np.vstack((col1, col2)).T # this combines the two columns in a 2d array with 2 rows 
                                #and the .T transposes the array so that it instead has 2 columns
tabET2 = tabET[col2 < 10**(-5)] #this looks at the second column for values less than 10**-5
                                #and indexes the ET table so that it only containes these 
                                #rows where the 2nd column



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
#%%
#Now the BPLS for ET
#Now for BPL
elstep = (elmaxet-elminet)/itera
elET = np.arange(elminet, elmaxet, elstep)
def bplET(f, fstar, n1, n2):
    s = 10
    res = (f/fstar)**n1 * (1+(f/fstar)**s)**(-(n1-n2)/s)
    return res

def AbplminET(fs, n1, n2):
    integrand = lambda f, fs, n1, n2:(bplET(f, fs, n1, n2)/sigETapp(f))**2
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
AtabET = np.vstack((inputsET.T, AminET)).T.reshape(len(n1r),len(n1r),len(n1r),4)
#%%
i = range(len(fs))
j = range(len(n1r))
k = range(len(n2r))
m = range(len(fs))
coords = np.array(np.meshgrid(i, j, k, m)).T.reshape(-1,4)

#%%
def fbpltabET(i, j, k, m):
    bplres = bplET(fs[m], AtabET[i,j,k,0], AtabET[i,j,k,1], AtabET[i,j,k,2])
    return AtabET[i,j,k,3]*bplres

   
FtabET = np.array(list(map(lambda args: fbpltabET(*args), coords))).reshape(len(n1r),len(fs),len(n1r),len(n2r))
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
plt.savefig('ETBPLScurves.png')

#%%
###########################
#LISA
###########################
#Now can create the noise model using functions
#Sa is the acceleration noise and ss is the
#optical metrology noise.
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

og = np.array(list(map(lambda args: sigtab(*args), Rtab)))


#%%    
freqvals = np.logspace(elminL, elmaxL, itera)   
sigvals = np.array(list(map(SigmaLisaApproxnom, freqvals)))
elstep = (elmaxL-elminL)/itera
elLISA = np.arange(elminL, elmaxL+elstep, elstep)

#%%
#Now for BPL

def bpl(f, fstar, n1, n2):
    s = 10
    res = (f/fstar)**n1 * (1+(f/fstar)**s)**(-(n1-n2)/s)
    return res


def Abplmin(fs, n1, n2):
    integrand = lambda f, n1, n2, fs:(bpl(f, fs, n1, n2)/SigmaLisaApprox(f))**2
    I1 = quad(integrand, ffmin, 10**(-3), args=(n1, n2, fs))[0]
    I2 = quad(integrand, 10**(-3), 10**(-1), args=(n1, n2, fs))[0]
    res = snr5/np.sqrt(T*sum((I1,I2)))
    return res



elbpl = np.arange(elminL, elmaxL, elstep)
n1r = np.linspace(ntmin, ntmax, itera)
n2r = np.linspace(ntmin, ntmax, itera)
fs = 10**elbpl

inputs = np.array(np.meshgrid(fs, n2r, n1r)).T.reshape(-1,3)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputs[:,[0,1,2]] = inputs[:,[0,2,1]]

#%%

Amin2 = np.array(list(map(lambda args: Abplmin(*args), inputs)))
Atab2 = np.vstack((inputs.T, Amin2)).T.reshape(len(n1r),len(n1r),len(n1r),4)

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

Ftab2 = np.array(list(map(lambda args: fbpltab(*args), coords))).reshape(len(n1r),len(fs),len(n1r),len(n2r))

#%%
maxims = []
def maxbplvals(i):
    maxims = np.log(np.max(Ftab2[i]))
    return maxims

maxpos = range(len(Ftab2))

maxbpl = np.array(list(map(maxbplvals, maxpos)))
#%%
fbplo = np.vstack((np.log(fs), maxbpl)).T


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
plt.ylim(1e-14, 1e-4)
plt.savefig('LISAnomBPLS.png')
#%%
#######################
#Combining LISA and ET curves
#######################

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
 
fvalscomb = np.logspace(np.log10(ffmin), np.log10(ffmax),750)
combine = np.array(list(map(omegatog, fvalscomb)))
nom = np.array(list(map(nomtog, fvalscomb)))
otog = np.vstack((fvalscomb, combine)).T

#%%
#bpls
def combbpl(f, fstar, n1, n2):
    s = 10
    res = (f/fstar)**n1 * (1+(f/fstar)**s)**(-(n1-n2)/s)
    return res 


def Abplmincomb(fs, n1, n2):
    integrand = lambda f, fs, n1, n2:(combbpl(f, fs, n1, n2)/SigmaLisaApprox(f))**2
    I1 = quad(integrand, ffmin, 10**(-3), args=(fs, n1, n2))[0]
    I2 = quad(integrand, 10**(-3), 10**(-1), args=(fs, n1, n2))[0]
    integrand2 = lambda f, fs, n1, n2:(combbpl(f, fs, n1, n2)/sigETapp(f))**2
    I3 = quad(integrand2, 1.6, 10**(1), args=(fs, n1, n2))[0]
    I4 = quad(integrand2, 10**(1), 445, args=(fs, n1, n2))[0]
    res = snr5/np.sqrt(T*sum((I1,I2,I3,I4)))
    return res   

elmin = np.log10(ffmin)
elmax = np.log10(ffmax)
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

#%%
def fbpltabcomb(i, j, k, m):
    res = []
    bplres = combbpl(fsc[m], Atab4[i,j,k,0], Atab4[i,j,k,1], Atab4[i,j,k,2])
    res.append([Atab4[i,j,k,3]*bplres])
    return res



Ftab4 = np.array(list(map(lambda args: fbpltabcomb(*args), coordsc))).reshape(len(fsc),len(fsc),len(fsc),len(fsc))
   

#%%
maximsc = []
def combmaxbplvals(i):
    maximsc = np.log(np.max(Ftab4[i]))
    return maximsc

combmaxpos = range(len(Ftab4))
maxbplcomb = np.array(list(map(combmaxbplvals, combmaxpos)))
combfbplo = np.vstack((np.log(fsc), maxbplcomb)).T
#%%
plt.figure(figsize=(6, 9))
plt.loglog(otog[:,0], nom, label = "Nominal Curve", color = "indigo", linewidth=2.5)
plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]), label = "Combined BPLS", color = "lime", linewidth=2.5)
plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]),':', color = "teal", label = 'LISA BPLS', linewidth=2.5)
plt.loglog(np.exp(fbploET[:,0]), np.exp(fbploET[:,1]), ':',color = "black", label = " ET BPLS", linewidth=2.5)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.title("Nominal and BPLS Curve for LISA and ET", fontsize = 16)
plt.xlabel("f (Hz)", fontsize = 16)
plt.legend(loc = (0.08,0.8), fontsize = 14)
plt.yscale('log')
plt.xscale('log')
plt.xlim(ffmin, ffmax)
plt.grid(True)
plt.savefig('CombineNomBPLSwold.png')
 