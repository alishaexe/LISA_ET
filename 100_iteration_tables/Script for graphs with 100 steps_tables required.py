import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
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

fetstar = 20
fi = 0.4*10**(-3)
#For the LISA mission they have designed the
#arms to be length L = 2.5*10^(6)
T = 3*yr
snr5 = 5

ffmin = 10**(-6)
ffmax = 10**3

elmin = (np.log10(ffmin))
elmax = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/100
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


plt.loglog(tabET2[:,0], tabET2[:,1], ',', label = "Numerical", color = "darkviolet")
plt.title('Einstein telescope sensitivity curve data')
plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
plt.xlim(1, 10**3)
plt.yscale('log')
plt.xscale('log')
plt.ylim(5 * 10**(-10), 10**(-5))
plt.show()
#%%
def sigp(f):
    res = 1.3*((3*30*10**(-1)*f**(-30)+5*10**(-6)*f**(-4.5)+0.6*10**(-11)*f**(2.8))
               *(1/2-1/2*np.tanh(0.1*(f-42)))+(1/2*np.tanh(0.1*(f-42)))*(2*10**(-11)*f**2.25 
                                                                         +10**(-13)*f**3))
    return res

def sigETapp(f):#Sigma_Ohm approx
    if f <= 1.6 or f >= 450:
        return 10**(-5)        
    if f > 1.6 and f < 450:
        res = sigp(f)
        if res > 10**(-5):#some values of f (f=449) give a result that is  > 10**-5
            res = 10**(-5)#so since we can't measure there anyway have set these values to ==10**-5
        return res
#%%
fvalsET = np.logspace(np.log10(ffmin), np.log10(ffmax),2000)#frequency values
sigETvals = np.array(list(map(sigETapp, fvalsET)))#The Omega_gw values from the ET data


plt.loglog(fvalsET, sigETvals, color = "indigo")
plt.title("Nominal sensitivity curve")
plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
plt.ylim(10**(-10), 10**(-4))
plt.yscale('log')
plt.xscale('log')
plt.xlim(ffmin, ffmax)
plt.grid(True)
plt.show()
#%%
#Plots both curves overlayed each other

plt.loglog(tabET2[:,0], tabET2[:,1], '-', label = "Numerical", color = "darkviolet")
plt.title("Einstein sensitivity curves")
plt.loglog(fvalsET, sigETvals, label = "Approximate", color = "indigo" )
plt.legend()
plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
plt.ylim(10**(-10), 10**(-4))
plt.yscale('log')
plt.xscale('log')
plt.xlim(ffmin, ffmax)
plt.grid(True)
plt.show()
#%%
#Now we find the PLS for ET
#This is the function for calculating the A_min values
#I separate the integration into 4 to help with the accuracy and avoid warnings
def AETmin(nt):
    integrand = lambda f, nt:((f/fetstar)**(nt)/sigETapp(f))**2
    I1 = quad(integrand, ffmin, 10**(-4), args=(nt))[0]
    I2 = quad(integrand, 10**(-4), 10**(-1), args=(nt))[0]
    I3 = quad(integrand, 10**(-1), 10, args=(nt))[0]
    I4 = quad(integrand, 10, ffmax, args = (nt))[0]
    res = snr5/np.sqrt(T*(I1+I2+I3+I4))
    return res

ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/100
ntetvals = np.arange(ntmin, ntmax, step)

#this array/list/map function does the same as earlier by mapping the nt values across
#without having to iterate
aetvals = np.array(list(map(AETmin, ntetvals)))
AETtab = np.vstack((ntetvals, aetvals)).T

def FETtab(i, j):
    res = AETtab[j,1]*(10**elET[i]/fetstar)**AETtab[j,0]
    return res
#%%
elstep = (elmax-elmin)/100
elET = np.arange(elmin, elmax, elstep)
i = range(len(elET))
j = range(len(AETtab))
coordset = np.array(np.meshgrid(i,j)).T.reshape(-1,2)
Ftabetpls = np.array(list(map(lambda args: FETtab(*args), coordset))).reshape(len(elET), len(AETtab))

maxedET = []
def maxETpls(i):
    maxedET = np.log(np.max(Ftabetpls[i]))
    return maxedET

maxposet = range(len(Ftabetpls))
maxplsvals = np.array(list(map(maxETpls, maxposet)))
flogomET = np.vstack((np.log(10**elET), maxplsvals)).T
#%%
plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]), color = "orangered")
plt.title("PLS curve for Einstein Telescope")
plt.axhline(AETmin(0), color='r', linestyle='--')
plt.ylim(1e-14, 1e-5)
plt.xlabel('f (Hz)')
plt.ylabel('Omega_gw')
plt.grid(True)
plt.xscale('log')
plt.show()

plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]), color = "orangered", label = "PLS")
plt.loglog(fvalsET, sigETvals, color = "indigo", label = "Nominal")
plt.legend()
plt.title("Nominal and PLS curve for ET")
plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
plt.ylim(10**(-14), 10**(-4))
plt.yscale('log')
plt.xscale('log')
plt.xlim(ffmin, ffmax)
plt.grid(True)
plt.show()
#%%
#Now the BPLS for ET
#Now for BPL
n1r = np.linspace(ntmin, ntmax, 100)
n2r = np.linspace(ntmin, ntmax, 100)
fs = 10**elET

inputsET = np.array(np.meshgrid(fs, n2r, n1r)).T.reshape(-1,3)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputsET[:,[0,1,2]] = inputsET[:,[0,2,1]]
AminET = np.load("AminET.npy")
AtabET = np.load("AtabET.npy")

#%%
def maxETbplvals(i):
    maximsET = np.log(np.max(FtabET[i]))
    return maximsET

FtabET = np.load("FtabET.npy")

maximsET = []
maxposET = range(len(FtabET))
maxbplET = np.array(list(map(maxETbplvals, maxposET)))


#%%
fbploET = np.vstack((np.log(fs), maxbplET)).T

#plots the bpl curve
plt.loglog(np.exp(fbploET[:,0]), np.exp(fbploET[:,1]), color = "orangered")
plt.title("BPLS Curve ET")
plt.ylim(1e-14, 1e-1)
plt.xlim(ffmin, ffmax)
plt.xlabel('f (Hz)')
plt.ylabel('Omega_gw')
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()

#plots all 3 graphs on same plot
plt.loglog(np.exp(fbploET[:,0]), np.exp(fbploET[:,1]), color = "orangered", label = "PLS curve")
plt.loglog(fvalsET, sigETvals, color = "indigo", label = "Nominal curve")
plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]), color = "lime", label = "BPLS curve")
plt.title("Einstein Sensitivity Curves")
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-14, 1e-4)
plt.xlim(ffmin, ffmax)
plt.show()
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
    sig = np.sqrt(2)*20/3 * (SI(f)/(2*pi*f)**4 + SII)*(1+(f/f2)**2)
    return sig

def SigmaLisaApprox(f):#Sigma_Ohm approx
    const = ((4*pi**2/(3*H0**2)))
    if f <= 10**(-5) or f >= 10**(-1):
        return 10**(-5)        
    if f > 10**(-5) and f < 10**(-1):
        res = const * f**3 *sigI(f)
        if res > 10**(-5):
            res = 10**(-5)
        return res


L = 25/3
fLisa = 1/(2*pi*L)



def N1(f): #They call these c1 and c2 even though it is N's in the paper
    n1 = 4*Ss + 8*Sa(f)*(1+(np.cos(f/fLisa))**2)
    return n1

def N2(f):
    n2 = -((2*Ss+8*Sa(f))*np.cos(f/fLisa))#Look at eq 61
    return n2

#%%    
freqvals = np.logspace(np.log10(ffmin), np.log10(ffmax),100)   
sigvals = np.array(list(map(SigmaLisaApprox, freqvals)))


plt.loglog(freqvals, sigvals, color = "indigo")
plt.ylabel(r"$\Omega_{gw}$")
plt.grid(True)
plt.xlim(ffmin, ffmax)
plt.xlabel("f (Hz)")
plt.title("Nominal Sensitivity curve of LISA")
plt.show()
#%%
#Now finding the PLS for LISA
def Almin(nt):
    integrand = lambda f, nt:((f/fLisa)**(nt)/SigmaLisaApprox(f))**2
    I1 = quad(integrand, ffmin, 10**(-4), args=(nt))[0]
    I2 = quad(integrand, 10**(-4), 10**(-1), args=(nt))[0]
    I3 = quad(integrand, 10**(-1), 10, args=(nt))[0]
    I4 = quad(integrand, 10, ffmax, args = (nt))[0]
    res = snr5/np.sqrt(T*(I1+I2+I3+I4))
    return res

#determining the envelop of the functions that bound Omega_gw for all the values of nt

ntvals = np.arange(ntmin, ntmax, step)
Aminvals = np.array(list(map(Almin, ntvals)))
Atab = np.vstack((ntvals, Aminvals)).T

#%%
#Our Ftab is slightly different from the mathematica script as they have the table
def Ftab(i, j):
    res = Atab[j,1]*(10**elLISA[i]/fLisa)**Atab[j,0]
    return res




elstep = (elmax-elmin)/100
elLISA = np.arange(elmin, elmax, elstep)
i = range(len(elLISA))
j = range(len(Atab))
coordsl1 = np.array(np.meshgrid(i, j)).T.reshape(-1,2)

FtabLISA = np.array(list(map(lambda args: Ftab(*args), coordsl1))).reshape(len(elLISA), len(Atab))
#%%
maxed = []

def maxtablisa(i):
    maxed = np.log(np.max(FtabLISA[i]))
    return maxed

maxposlisa = range(len(FtabLISA))
maxpls = np.array(list(map(maxtablisa, maxposlisa)))

flogom = np.vstack((np.log(10**elLISA), maxpls)).T

#iterped = interp1d(flogom[:,0], flogom[:,1])

plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]), color = "orangered")
plt.axhline(Almin(0), color='r', linestyle='--')
plt.ylim(1e-14, 1e-4)
plt.xlim(ffmin, ffmax)
plt.xlabel('f (Hz)')
plt.ylabel('Omega_gw')
plt.title('PLS curve LISA')
plt.grid(True)
plt.xscale('log')
plt.show()
#%%
plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]), color = "orangered", label = "PLS")
plt.loglog(freqvals, sigvals, color = "indigo", label = "Nominal")
plt.ylabel(r"$\Omega_{gw}$")
plt.legend()
plt.grid(True)
plt.xlim(ffmin, ffmax)
plt.ylim(1e-14, 1e-4)
plt.xlabel("f (Hz)")
plt.title("Nominal and PLS curve LISA")
plt.show()
#%%
#Now for BPL
elbpl = np.arange(elmin, elmax, elstep)
n1r = np.linspace(ntmin, ntmax, 100)
n2r = np.linspace(ntmin, ntmax, 100)
fs = 10**elbpl

inputs = np.array(np.meshgrid(fs, n2r, n1r)).T.reshape(-1,3)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputs[:,[0,1,2]] = inputs[:,[0,2,1]]

Amin2 = np.load("Amin2.npy")
Atab2 = np.load("Atab2.npy")
#%%
i = np.array(range(len(fs)))#defining them as arrays here means that in
j = np.array(range(len(n1r)))#the meshgrid they'll stay in order
k = np.array(range(len(n2r)))#i.e 000, 001,002 etc
m = np.array(range(len(fs)))
coords = np.array(np.meshgrid(i,j,k,m)).T.reshape(-1,4)

Ftab2 = np.load("Ftab2.npy")

       

#%%
maxims = []
def maxbplvals(i):
    #maxims.append(np.log(np.max(Ftab2[i])))
    maxims = np.log(np.max(Ftab2[i]))
    return maxims
maxpos = range(len(Ftab2))
maxbpl = np.array(list(map(maxbplvals, maxpos)))

#%%
fbplo = np.vstack((np.log(fs), maxbpl)).T


plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]), color = "lime")
plt.title("LISA BPLS sensitivity curve")
plt.ylim(1e-14, 1e-5)
plt.xlabel('f (Hz)')
plt.ylabel('Omega_gw')
plt.grid(True)
plt.xscale('log')
plt.xlim(ffmin, ffmax)
plt.show()


plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]), label = "BPLS curve", color = "lime")
plt.loglog(freqvals, sigvals, label = "Nominal Curve", color = "indigo")
plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]), label = "PLS Curve", color = "orangered")
plt.grid(True)
plt.title("LISA sensitivity curves")
plt.legend()
plt.xlim(ffmin, ffmax)
plt.xlabel('f (Hz)')
plt.ylabel('Omega_gw')
plt.xscale('log')
plt.ylim(1e-14, 1e-4)
plt.show()
#%%
####Combining LISA and ET curves
def omegatog(f):
    if f <= 1:
        return SigmaLisaApprox(f)
    if f > 1:
        return sigETapp(f)


fvalscomb = np.logspace(np.log10(ffmin), np.log10(ffmax),500)
combine = np.array(list(map(omegatog, fvalscomb)))
 
otog = np.vstack((fvalscomb, combine)).T

plt.loglog(otog[:,0], otog[:,1], color = "indigo")
plt.title("Nominal curves for ET and LISA")
plt.grid(True) 
plt.xlim(ffmin, ffmax) 
plt.xlabel('f (Hz)')
plt.ylabel('Omega_gw')
plt.show()

#%%
#combine pls curves
fstar = 1.1
ntmin = -9/2
ntmax = 9/2
    
def Amincomb(nt):
    integrand = lambda f, nt:((f/fstar)**(nt)/omegatog(f))**2
    I1 = quad(integrand, ffmin, 10**(-4), args=(nt))[0]
    I2 = quad(integrand, 10**(-4), 10**(-1), args=(nt))[0]
    I3 = quad(integrand, 10**(-1), 10, args=(nt))[0]
    I4 = quad(integrand, 10, ffmax, args = (nt))[0]
    res = snr5/np.sqrt(T*(I1+I2+I3+I4))
    return res    

ntcombvals = np.linspace(ntmin, ntmax, 100)
Amin3 = np.array(list(map(Amincomb, ntcombvals)))


Atab3 = np.vstack((ntcombvals, Amin3)).T


def ftab3(i, j):
    res = Atab3[j,1]*(10**combel[i]/fstar)**Atab3[j,0]
    return res


combel = np.linspace(elmin, elmax, 100)
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


plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]), color = "orangered")
plt.xscale('log')
plt.ylim(1e-14, 1e-4)
plt.xlim(ffmin, ffmax) 
plt.xlabel('f (Hz)')
plt.ylabel('Omega_gw')
plt.title('Combined PLS curve for LISA and ET')
plt.grid(True)
plt.show()

plt.loglog(otog[:,0], otog[:,1] , color = "indigo", label = "Nominal")
plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]), color = "orangered", label = "PLS")
plt.title("Nominal and PLS curves ")
plt.legend()
plt.grid(True) 
plt.xlim(ffmin, ffmax) 
plt.xlabel('f (Hz)')
plt.ylabel('Omega_gw')
plt.show()
#%%
#bpls
elc = np.linspace(elmin, elmax, 100)
n1c = np.arange(ntmin, ntmax, step)
n2c = np.arange(ntmin, ntmax, step)
fsc = 10**elc



inputsc = np.array(np.meshgrid(fsc, n2c, n1c)).T.reshape(-1,3)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputsc[:,[0,1,2]] = inputsc[:,[0,2,1]]

Amin4 = np.load("Amin4.npy")
Atab4 = np.load("Atab4.npy")

#%%
ic = range(len(fsc))
jc = range(len(n1c))
kc = range(len(n2c))
mc = range(len(fsc))
coordsc = np.array(np.meshgrid(ic,jc,kc, mc)).T.reshape(-1,4)
   

#%%
maximsc = []
def combmaxbplvals(i):
    maximsc = np.log(np.max(Ftab4[i]))
    return maximsc
Ftab4 = np.load("Ftab4.npy")
combmaxpos = range(len(Ftab4))
maxbplcomb = np.array(list(map(combmaxbplvals, combmaxpos)))
combfbplo = np.vstack((np.log(fsc), maxbplcomb)).T



plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]), color = "lime")
plt.title("BPLS Curve for LISA and ET")
plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
plt.ylim(10**(-14), 10**(-4))
plt.yscale('log')
plt.xscale('log')
plt.xlim(ffmin, ffmax)
plt.grid(True)
plt.show()
#%%
plt.loglog(otog[:,0], otog[:,1], label = "Nominal Curve", color = "indigo")
plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]), label = "BPLS", color = "lime")
plt.ylabel(r"$\Omega_{gw}$")
plt.title("Nominal and BPLS Curve for LISA and ET")
plt.xlabel("f (Hz)")
plt.legend()
plt.ylim(10**(-14), 10**(-4))
plt.yscale('log')
plt.xscale('log')
plt.xlim(ffmin, ffmax)
plt.grid(True)
plt.show()


#%%
plt.loglog(otog[:,0], otog[:,1], label = "Nominal", color = "indigo")
plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]), label = "BPLS", color = "lime")
plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]), label = "PLS", color = "orangered")
plt.ylabel(r"$\Omega_{gw}$")
plt.title("Curves for LISA and ET")
plt.legend()
plt.xlabel("f (Hz)")
plt.ylim(10**(-14), 10**(-4))
plt.yscale('log')
plt.xscale('log')
plt.xlim(ffmin, ffmax)
plt.grid(True)
plt.show()