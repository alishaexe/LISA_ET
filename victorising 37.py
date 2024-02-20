import matplotlib.pyplot as plt
import numpy as np
#from scipy.interpolate import make_interp_spline, BSpline
#from scipy import integrate
from scipy.integrate import quad
#from scipy.signal import savgol_filter as sav
#import multiprocessing
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

fetstar = 20
fi = 0.4*10**(-3)
#For the LISA mission they have designed the
#arms to be length L = 2.5*10^(6)
T = 3*yr
snr5 = 5

ffmin = 5*10**(-5)
ffmax = 10**3

elmin = (np.log10(ffmin))
elmax = (np.log10(ffmax))
#%%
col1 = Etab[:,0]

def tabETapp(i):
    res = (4*pi**2 / (3*H0**2))*Etab[i,0]**3 * Etab[i,3]**2
    return res

vals = range(len(Etab))
col2 = np.array(list(map(tabETapp, vals)))
tabET = np.vstack((col1, col2)).T
tabET2 = tabET[col2 < 10**(-5)] #this sorts the tabET array so that it only assigns the rows where the 2nd column
                                #has values < 10**-5

#plt.figure(figsize=(8, 4))
for i in range(tabET2.shape[0]):
    plt.loglog(tabET2[i,0], tabET2[i,1], marker=',', color='k') #plots each point, idk why it won't do it without the for loop


plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
plt.xlim(1, 10**3)
plt.yscale('log')
plt.xscale('log')
plt.ylim(5 * 10**(-10), 10**(-5))
plt.show()
#%%

#plots the interpolated curve
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
            res = 10**(-5)#so since we can't/don't measure there anyway have set these values to ==10**-5
        return res
#%%
fvalsET = np.logspace(np.log10(ffmin), np.log10(ffmax),2000)
sigETvals = np.array(list(map(sigETapp, fvalsET)))


plt.loglog(fvalsET, sigETvals)
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
for i in range(tabET2.shape[0]):
    plt.loglog(tabET2[i,0], tabET2[i,1], marker=',', color='k')

plt.loglog(fvalsET, sigETvals)
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

aetvals = np.array(list(map(AETmin, ntetvals)))
AETtab = np.vstack((ntetvals, aetvals)).T

def FETtab(j, A, nt):
    res = A*(j/fetstar)**nt
    return res


#%%
maxedET = []
elstep = (elmax-elmin)/50
elET = np.arange(elmin, elmax, elstep)

for i in range(len(elET)):
    bemaxET = FETtab(10**elET[i],AETtab[:,1], AETtab[:,0])
    maxedET.append(np.log(np.max(bemaxET)))


flogomET = np.vstack((np.log(10**elET), maxedET)).T

# #iterped = interp1d(flogomET[:,0], flogomET[:,1])

plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]))
plt.axhline(AETmin(0), color='r', linestyle='--')
plt.ylim(1e-14, 1e-5)
plt.xlabel('f (Hz)')
plt.ylabel('Omega_gw')
plt.title('Log-Log Plot of Omega_gw (calculated max)')
plt.grid(True)
plt.xscale('log')
plt.show()

plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]))
plt.loglog(fvalsET, sigETvals)
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
def bplET(f, fstar, n1, n2):
    s = 10
    res = (f/fstar)**n1 * (1+(f/fstar)**s)**(-(n1-n2)/s)
    return res

def AbplminET(fs, n1, n2):
    integrand = lambda f, n1, n2, fs:(bplET(f, fs, n1, n2)/sigETapp(f))**2
    I1 = quad(integrand, ffmin, 10**(-4), args=(n1, n2, fs))[0]
    I2 = quad(integrand, 10**(-4), 10**(-1), args=(n1, n2, fs))[0]
    I3 = quad(integrand, 10**(-1), 10, args=(n1, n2, fs))[0]
    I4 = quad(integrand, 10, ffmax, args = (n1, n2, fs))[0]
    res = snr5/np.sqrt(T*(I1+I2+I3+I4))
    return res

n1r = np.linspace(ntmin, ntmax, 50)
n2r = np.linspace(ntmin, ntmax, 50)

fs = 10**elET


idek = []
inputsET = np.array(np.meshgrid(fs, n2r, n1r)).T.reshape(-1,3)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputsET[:,[0,1,2]] = inputsET[:,[0,2,1]]

AminET = []

AminET = np.array(list(map(lambda args: AbplminET(*args), inputsET)))


AtabET = np.vstack((inputsET.T, AminET)).T


#test = Atabvals.reshape(-1, len(inputs), 4)
AtabET = AtabET.reshape(len(n1r),len(n1r),len(n1r),4)
#%%
i = range(len(fs))
j = range(len(n1r))
k = range(len(n2r))
coords = np.array(np.meshgrid(i,j,k)).T.reshape(-1,3)
sort = np.argsort(coords[:,0])
coords = coords[sort]

#%%

#%%

test = np.linspace(elmin, elmax, 50)

fs2 = 10**test
def fbpltabET(i, j, k):
    res = []
    bplres = bplET(fs2[i], AtabET[i,j,k,0], AtabET[i,j,k,1], AtabET[i,j,k,2])
    res.append([AtabET[i,j,k,3]*bplres])
    return res


FtabET = []

FtabET = np.array(list(map(lambda args: fbpltabET(*args), coords))).reshape(-1,2500,1)
#%%

def maxETbplvals(i):
    #maxims.append(np.log(np.max(Ftab2[i])))
    maximsET = np.log(np.max(FtabET[i]))
    return maximsET
maxposET = range(len(FtabET))

maxbplET = np.array(list(map(maxETbplvals, maxposET)))
#%%
fbploET = np.vstack((np.log(fs), maxbplET)).T
#iterped2 = interp1d(fbplo[:,0], fbplo[:,1])


plt.loglog(np.exp(fbploET[:,0]), np.exp(fbploET[:,1]))
#plt.axhline(Almin(0), color='r', linestyle='--')
plt.ylim(1e-14, 1e-1)
plt.xlim(ffmin, ffmax)
plt.xlabel('f (Hz)')
plt.ylabel('Omega_gw')
#plt.title('Log-Log Plot of Omega_gw (calculated max)')
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()




plt.loglog(np.exp(fbploET[:,0]), np.exp(fbploET[:,1]))
plt.loglog(fvalsET, sigETvals)
plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]))
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-14, 1e-1)
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
    if f <= 10**(-4) or f >= 10**(-1):
        return 10**(-5)        
    if f > 10**(-4) and f < 10**(-1):
        return const * f**3 *sigI(f)

L = 25/3
fLisa = 1/(2*pi*L)
ffmin = 5*10**(-5)
ffmax = 10**3


def N1(f): #They call these c1 and c2 even though it is N's in the paper
    n1 = 4*Ss + 8*Sa(f)*(1+(np.cos(f/fLisa))**2)
    return n1

def N2(f):
    n2 = -((2*Ss+8*Sa(f))*np.cos(f/fLisa))#Look at eq 61
    return n2

#%%    


freqvals = np.logspace(np.log10(ffmin), np.log10(ffmax),37)

    
sigvals = np.array(list(map(SigmaLisaApprox, freqvals)))

    

plt.loglog(freqvals, sigvals)
plt.ylabel(r"$\Omega_{gw}$")
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
ntmin = -9/2
ntmax = 9/2


#In the mathematica notebook they generate the table for the values of A
#It will take us a couple lines of code to do the same thing.
#The step value is the size of the steps we take in our values of ntmin and ntmax
step = 0.25
#ntvals creates an array of these values between (AND including) ntmin and ntmax
#the ntmax+step ensures that we include our ntmax number as np.arange(1, n, 2) = 1, ..., n-2 

ntvals = np.arange(ntmin, ntmax+step, step)
#we can now formulate these into a table by mapping these values together
#We now want to map our values of nt and Amin
#we can do this using the list() and map() operations
#in python
#map computes the function Amin with the iterable values of ntvals
#list turns the mapped values into a list, and array turns these
#into an array - have to do it this way can't just do map -> array


Aminvals = np.array(list(map(Almin, ntvals)))


#We vertically stack the two arrays together and transpose them to get (41,2) dim array
#which is the table in mathematica.
Atab = np.vstack((ntvals, Aminvals)).T

#Our Ftab is slightly different from the mathematica script as they have the table
def Ftab(j, A, nt):
    res = A*(j/fLisa)**nt
    return res



xmin = np.log10(ffmin)
xmax = np.log10(ffmax)
xstep = 0.2
x = np.arange(xmin, xmax, xstep)
newf = np.array(np.log(10**x))


maxed = []

for i in range(len(x)):
    bemax = Ftab(10**x[i],Atab[:,1], Atab[:,0])
    maxed.append(np.log(np.max(bemax)))


flogom = np.vstack((newf, maxed)).T

#iterped = interp1d(flogom[:,0], flogom[:,1])

plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]))
plt.axhline(Almin(0), color='r', linestyle='--')
plt.ylim(1e-14, 1e-5)
plt.xlabel('f (Hz)')
plt.ylabel('Omega_gw')
plt.title('Log-Log Plot of Omega_gw (calculated max)')
plt.grid(True)
plt.xscale('log')
plt.show()
#%%

#Now for BPL
def bpl(f, fstar, n1, n2):
    s = 10
    res = (f/fstar)**n1 * (1+(f/fstar)**s)**(-(n1-n2)/s)
    return res


def Abplmin(fs, n1, n2):
    integrand = lambda f, n1, n2, fs:(bpl(f, fs, n1, n2)/SigmaLisaApprox(f))**2
    I1 = quad(integrand, ffmin, 10**(-4), args=(n1, n2, fs))[0]
    I2 = quad(integrand, 10**(-4), 10**(-1), args=(n1, n2, fs))[0]
    I3 = quad(integrand, 10**(-1), 10, args=(n1, n2, fs))[0]
    I4 = quad(integrand, 10, ffmax, args = (n1, n2, fs))[0]
    res = snr5/np.sqrt(T*(I1+I2+I3+I4))
    return res


elmin = np.log10(ffmin)
elmax = np.log10(ffmax)
elstep = 0.2
el = np.arange(elmin, elmax, elstep)



n1r = np.linspace(ntmin, ntmax, 37)
n2r = np.linspace(ntmin, ntmax, 37)

fs = 10**el


idek = []
inputs = np.array(np.meshgrid(fs, n2r, n1r)).T.reshape(-1,3)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputs[:,[0,1,2]] = inputs[:,[0,2,1]]

#%%
#meep = inputs.reshape(1369,37,3)
start = time.time()
Amin2 = []

Amin2 = np.array(list(map(lambda args: Abplmin(*args), inputs)))


Atab2 = np.vstack((inputs.T, Amin2)).T


#test = Atabvals.reshape(-1, len(inputs), 4)
Atab2 = Atab2.reshape(len(n1r),len(n1r),len(n1r),4)
end = time.time()
print(end-start)
#%%
i = range(len(fs))
j = range(len(n1r))
k = range(len(n2r))
coords = np.array(np.meshgrid(i,j,k)).T.reshape(-1,3)
sort = np.argsort(coords[:,0])
coords = coords[sort]

def fbpltab(i, j, k):
    res = []
    bplres = bpl(fs[i], Atab2[i,j,k,0], Atab2[i,j,k,1], Atab2[i,j,k,2])
    res.append([Atab2[i,j,k,3]*bplres])
    return res





Ftab2 = []
freqlist = []


Ftab2 = np.array(list(map(lambda args: fbpltab(*args), coords))).reshape(-1,1369,1)
        
            #i get correct values for Ftab2 but incorrect dims

#end = time.time()
#print(end-start)
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
#iterped2 = interp1d(fbplo[:,0], fbplo[:,1])


plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]))
#plt.axhline(Almin(0), color='r', linestyle='--')
plt.ylim(1e-14, 1e-5)
plt.xlabel('f (Hz)')
#plt.ylabel('Omega_gw')
#plt.title('Log-Log Plot of Omega_gw (calculated max)')
plt.grid(True)
plt.xscale('log')
plt.show()




plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]))
plt.loglog(freqvals, sigvals)
plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]))
plt.show()



#%%
plt.loglog(fvalsET, sigETvals)
plt.loglog(freqvals, sigvals)
plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
plt.ylim(10**(-14), 10**(-4))
plt.yscale('log')
plt.xscale('log')
plt.xlim(ffmin, ffmax)
plt.grid(True)
plt.show()

#%%
####Combining LISA and ET curves
def omegatog(f):
    if f <= 1:
        return SigmaLisaApprox(f)
    if f > 1:
        return sigETapp(f)

# combineET = np.vstack((fvalsET, sigETvals)).T
# combineLISA = np.vstack((freqvals,sigvals)).T
# filteredET = combineET[combineET[:, 0] > 1]
# filteredLISa = combineLISA[combineLISA[:,0] <= 1]
# otog = np.concatenate((filteredLISa, filteredET))
fvalscomb = np.logspace(np.log10(ffmin), np.log10(ffmax),500)
combine = np.array(list(map(omegatog, fvalscomb)))   
otog = np.vstack((fvalscomb, combine)).T
plt.loglog(otog[:,0], otog[:,1])
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

Amin3 = []
ntcombvals = np.linspace(ntmin, ntmax, 75)
Amin3 = np.array(list(map(Amincomb, ntcombvals)))


Atab3 = np.vstack((ntcombvals, Amin3)).T


#test = Atabvals.reshape(-1, len(inputs), 4)
#Atab3 = Atab3.reshape(len(n1r),len(n1r),len(n1r),4)
# combineetpls = np.vstack((np.exp(flogomET[:,0]), np.exp(flogomET[:,1]))).T
# combinelisapls = np.vstack((np.exp(flogom[:,0]), np.exp(flogom[:,1]))).T
# filteredETpls = combineetpls[combineetpls[:, 0] > 1]
# filteredLISapls = combinelisapls[combinelisapls[:,0] <= 1]
# otogpls = np.concatenate((filteredLISapls, filteredETpls))

def ftab3(j, A, nt):
    res = A*(j/fstar)**nt
    return res

combmin = np.log10(ffmin)
combmax = np.log10(ffmax)
combstep = 0.2
comb = np.arange(combmin, combmax, combstep)
newfcomb = np.array(np.log(10**comb))


combmaxed = []

for i in range(len(comb)):
    bemax = ftab3(10**comb[i],Atab3[:,1], Atab3[:,0])
    combmaxed.append(np.log(np.max(bemax)))


flogomcomb = np.vstack((newfcomb, combmaxed)).T
otogpls = np.vstack((np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1])) )

plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]))
plt.xscale('log')
plt.ylim(1e-14, 1e-4)
plt.xlabel('f (Hz)')
plt.ylabel('Omega_gw')
plt.title('Log-Log Plot of Omega_gw (calculated max)')
plt.grid(True)
plt.show()

plt.loglog(otog[:,0], otog[:,1])
plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]))
plt.show()
#%%
#bpls

def combbpl(f, fstar, n1, n2):
    s = 10
    res = (f/fstar)**n1 * (1+(f/fstar)**s)**(-(n1-n2)/s)
    return res 




def Abplmincomb(fs, n1, n2):
    integrand = lambda f, n1, n2, fs:(combbpl(f, fs, n1, n2)/omegatog(f))**2
    I1 = quad(integrand, ffmin, 10**(-4), args=(n1, n2, fs))[0]
    I2 = quad(integrand, 10**(-4), 10**(-1), args=(n1, n2, fs))[0]
    I3 = quad(integrand, 10**(-1), 10, args=(n1, n2, fs))[0]
    I4 = quad(integrand, 10, ffmax, args = (n1, n2, fs))[0]
    res = snr5/np.sqrt(T*(I1+I2+I3+I4))
    return res


elmin = np.log10(ffmin)
elmax = np.log10(ffmax)
elstepc = 0.2
#elc = np.linspace(elmin, elmax, 50)
elc = np.arange(elmin, elmax, 0.25)


n1c = np.arange(ntmin, ntmax, 0.25)
n2c = np.arange(ntmin, ntmax, 0.25)

fsc = 10**elc



inputsc = np.array(np.meshgrid(fsc, n2c, n1c)).T.reshape(-1,3)
#This makes it so n1r is in the second column
# so inputs(fs, n1r, n2r)
inputsc[:,[0,1,2]] = inputsc[:,[0,2,1]]

Amin4 = []

Amin4 = np.array(list(map(lambda args: Abplmincomb(*args), inputsc)))

Atab4 = np.vstack((inputsc.T, Amin4)).T
Atab4 = Atab4.reshape(len(fsc),len(n1c),len(n1c),4)
ic = range(len(fsc))
jc = range(len(n1c))
kc = range(len(n2c))
coordsc = np.array(np.meshgrid(ic,jc,kc)).T.reshape(-1,3)
sortc = np.argsort(coordsc[:,0])
coordsc = coordsc[sortc]
#%%
def fbpltabcomb(i, j, k):
    res = []
    bplres = combbpl(fsc[i], Atab4[i,j,k,0], Atab4[i,j,k,1], Atab4[i,j,k,2])
    res.append([Atab4[i,j,k,3]*bplres])
    return res





Ftab4 = []
freqlist = []


Ftab4 = np.array(list(map(lambda args: fbpltabcomb(*args), coordsc))).reshape(-1,len(fsc)**2,1)
        
            #i get correct values for Ftab2 but incorrect dims

maximsc = []
def combmaxbplvals(i):
    #maxims.append(np.log(np.max(Ftab2[i])))
    maximsc = np.log(np.max(Ftab4[i]))
    return maximsc
combmaxpos = range(len(Ftab4))

maxbplcomb = np.array(list(map(combmaxbplvals, combmaxpos)))
combfbplo = np.vstack((np.log(fsc), maxbplcomb)).T
#iterped2 = interp1d(fbplo[:,0], fbplo[:,1])


plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]))
plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
plt.ylim(10**(-14), 10**(-6))
plt.yscale('log')
plt.xscale('log')
plt.xlim(ffmin, ffmax)
plt.grid(True)
plt.show()
#test = Atabvals.reshape(-1, len(inputs), 4)
#Atab4 = Atab4.reshape(len(n1r),len(n1r),len(n1r),4)

# combineetbpls = np.vstack((np.exp(fbploET[:,0]), np.exp(fbploET[:,1]))).T
# combinelisabpls = np.vstack((np.exp(fbplo[:,0]), np.exp(fbplo[:,1]))).T

# filteredETbpls = combineetbpls[combineetbpls[:, 0] > 1]
# filteredLISabpls = combinelisabpls[combinelisabpls[:,0] <= 1]
# otogbpls = np.concatenate((filteredLISabpls, filteredETbpls))
    
#plt.loglog(otogbpls[:,0], otogbpls[:,1])
#%%
plt.loglog(otog[:,0], otog[:,1])
plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]))
plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
plt.ylim(10**(-14), 10**(-4))
plt.yscale('log')
plt.xscale('log')
plt.xlim(ffmin, ffmax)
plt.grid(True)
plt.show()






