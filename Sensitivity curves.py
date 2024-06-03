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
#function for creating the table of the ET data
#we use i to iterate through each row of the data
def tabETapp(i):
    res = (4*pi**2 *Etab[i,0]**3 * Etab[i,3]**2/ (3*H0**2))
    return res

col1 = Etab[:,0] # assigning the first column of the data to the name col1
vals = range(len(Etab))#this makes the list of numbers to iterate through for i
col2 = np.array(list(map(tabETapp, vals)))
#map takes the values from vals and inputs them into the tabETapp function, the list() 
#formats the output from the function for each of these values
#as a list, and np.array() then turns them into an array. This is all assigned to col2
tabET = np.vstack((col1, col2)).T # this combines the two columns in a 2d array with 2 rows 
                                #and the .T transposes the array so that it instead has 2 columns
# tabET2 = tabET[col2 < 10**(-5)] #this looks at the second column for values less than 10**-5
                                #and indexes the ET table so that it only containes these 
                                #rows where the 2nd column

#%%
# plt.figure(figsize=(6, 9)) 
# plt.loglog(tabET2[:,0], tabET2[:,1], ',', label = "Numerical", color = "darkviolet", linewidth=2.5)
# plt.title('Einstein telescope sensitivity curve data')
# plt.ylabel(r"$\Omega_{gw}$")
# plt.xlabel("f (Hz)")
# plt.xlim(1, ffmax)
# plt.yscale('log')
# plt.xscale('log')
# plt.grid(True)
# plt.show()
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
plt.xlim(10**0, 400)
plt.grid(True)
plt.show()
#%%
#Plots both curves overlayed each other

plt.figure(figsize=(6, 9)) 
plt.loglog(tabET[:,0], tabET[:,1], '--', label = "Numerical", color = "darkviolet", linewidth=2.5)
plt.title("Nominal Sensitivity curve of ET", fontsize = 16)
plt.loglog(fvalsET, sigETvals, '-',label = "Approximate", linewidth = 2.5,color = "indigo" )
plt.legend(fontsize = 16)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.xlabel("f (Hz)", fontsize = 16)
#plt.ylim(10**(-9), 10**(-5))
plt.yscale('log')
plt.xscale('log')
plt.xlim(10**0, 400)
plt.grid(True)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/ETnoms.png', bbox_inches='tight')
plt.show()

#%%
#Now we find the PLS for ET
#This is the function for calculating the A_min values
#I separate the integration into 4 to help with the accuracy and avoid warnings
def AETmin(nt):
    integrand = lambda f, nt:((f/fetstar)**(nt)/sigp(f))**2
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


#%%

elstep = (elmaxet-elminet)/2000
elET = np.arange(elminet, elmaxet, elstep)
i = range(len(elET))
j = range(len(AETtab))
coordset = np.array(np.meshgrid(i,j)).T.reshape(-1,2)
Ftabetpls = np.array(list(map(lambda args: FETtab(*args), coordset))).reshape(len(elET), 1,len(AETtab))


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


#%%
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
#Now the BPLS for ET
#Now for BPL
# def bplET(f, fstar, n1, n2):
#     s = 10
#     res = (f/fstar)**n1 * (1/2+(1/2)*(f/fstar)**s)**(-(n1-n2)/s)
#     return res

# def AbplminET(fs, n1, n2):
#     integrand = lambda f, fs, n1, n2:(bplET(f, fs, n1, n2)/sigp(f))**2
#     I1 = quad(integrand, 1.6, 100, args=(fs, n1, n2))[0]
#     I2 = quad(integrand, 100, 445, args=(fs, n1, n2))[0]
#     res = snr5/np.sqrt(2*T*sum((I1, I2)))
#     return res

# n1r = np.linspace(ntmin, ntmax, itera)
# n2r = np.linspace(ntmin, ntmax, itera)
# fs = 10**elET

# inputsET = np.array(np.meshgrid(fs, n2r, n1r)).T.reshape(-1,3)
# #This makes it so n1r is in the second column
# # so inputs(fs, n1r, n2r)
# inputsET[:,[0,1,2]] = inputsET[:,[0,2,1]]

# AminET = np.array(list(map(lambda args: AbplminET(*args), inputsET)))
# AtabET = np.vstack((inputsET.T, AminET)).T.reshape(len(n1r),len(n1r),len(n1r),4)
# #%%
# i = range(len(fs))
# j = range(len(n1r))
# k = range(len(n2r))
# m = range(len(fs))
# coords = np.array(np.meshgrid(i, j, k, m)).T.reshape(-1,4)

# #%%
# def fbpltabET(i, j, k, m):
#     bplres = bplET(fs[m], AtabET[i,j,k,0], AtabET[i,j,k,1], AtabET[i,j,k,2])
#     return AtabET[i,j,k,3]*bplres

   
# FtabET = np.array(list(map(lambda args: fbpltabET(*args), coords))).reshape(len(n1r),len(fs),len(n1r),len(n2r))
# #%%
# def maxETbplvals(i):
#     maximsET = np.log(np.max(FtabET[i]))
#     return maximsET

# maximsET = []
# maxposET = range(len(FtabET))
# maxbplvals = np.array(list(map(maxETbplvals, maxposET)))
# maxbplET = maxbplvals

# #%%
# fbploET = np.vstack((np.log(fs), maxbplET)).T
# #plots the bpl curve
# plt.figure(figsize=(6, 9)) 
# plt.loglog(np.exp(fbploET[:,0]), np.exp(fbploET[:,1]), '-',color = "lime", linewidth=2.5)
# plt.title("BPLS Curve ET")
# plt.xlabel('f (Hz)')
# plt.ylabel(r"$\Omega_{gw}$")
# plt.grid(True)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()


# #%%
# #plots all 3 graphs on same plot
# plt.figure(figsize=(6, 9)) 
# plt.loglog(fvalsET, sigETvals, color = "indigo",linewidth=2.5, label = "Nominal")
# plt.loglog(np.exp(fbploET[:,0]), np.exp(fbploET[:,1]), linewidth=2.5,color = "lime", label = "BPLS")
# plt.title('ET Nominal and BPL Curves')
# plt.ylabel(r"$\Omega_{gw}$")
# plt.legend(fontsize="10", loc = 'upper center')
# plt.grid(True)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/ETBPLScurves.png', bbox_inches='tight')
# plt.show()
#%%
###########################
#LISA
###########################

#Numerical curve from smith and caldwell
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

plt.figure(figsize=(6, 9)) 
plt.loglog(f, og,'--' ,color = "darkviolet", label = "Numerical", linewidth=2.5)
plt.loglog(freqvals, sigvals, color = "indigo", label = "Approximate", linewidth=2.5)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.grid(True)
plt.legend(fontsize = 16)
plt.xlabel("f (Hz)", fontsize = 16)
plt.title("Nominal Sensitivity curve of LISA", fontsize = 16)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/LISAnoms.png', bbox_inches='tight')
plt.show()
#%%
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
#Now for BPL

# def bpl(f, fstar, n1, n2):
#     s = 10
#     res = (f/fstar)**n1 * (1/2+1/2*(f/fstar)**s)**(-(n1-n2)/s)
#     return res


# def Abplmin(fs, n1, n2):
#     integrand = lambda f, n1, n2, fs:(bpl(f, fs, n1, n2)/Ohms(f))**2
#     I1 = quad(integrand, ffmin, 10**(-3), args=(n1, n2, fs))[0]
#     I2 = quad(integrand, 10**(-3), 10**(-1), args=(n1, n2, fs))[0]
#     res = snr5/np.sqrt(2*T*sum((I1,I2)))
#     return res



# elbpl = np.arange(elminL, elmaxL, elstep)
# n1r = np.linspace(ntmin, ntmax, itera)
# n2r = np.linspace(ntmin, ntmax, itera)
# fs = 10**elbpl

# inputs = np.array(np.meshgrid(fs, n2r, n1r)).T.reshape(-1,3)
# #This makes it so n1r is in the second column
# # so inputs(fs, n1r, n2r)
# inputs[:,[0,1,2]] = inputs[:,[0,2,1]]

# #%%

# Amin2 = np.array(list(map(lambda args: Abplmin(*args), inputs)))
# Atab2 = np.vstack((inputs.T, Amin2)).T.reshape(len(n1r),len(n1r),len(n1r),4)

# #%%
# i = np.array(range(len(fs)))#defining them as arrays here means that in
# j = np.array(range(len(n1r)))#the meshgrid they'll stay in order
# k = np.array(range(len(n2r)))#i.e 000, 001,002 etc
# m = np.array(range(len(fs)))
# coords = np.array(np.meshgrid(i,j,k,m)).T.reshape(-1,4)
# #here in meshgrid have done ikj purely because this may it will sort j
# #like it sorts i and lets k change; we then switch round the columns so that
# #it is i,j,k
# def fbpltab(i, j, k, m):
#     bplres = bpl(fs[m], Atab2[i,j,k,0], Atab2[i,j,k,1], Atab2[i,j,k,2])
#     return Atab2[i,j,k,3]*bplres

# Ftab2 = np.array(list(map(lambda args: fbpltab(*args), coords))).reshape(len(n1r),len(fs),len(n1r),len(n2r))

# #%%
# maxims = []
# def maxbplvals(i):
#     maxims = np.log(np.max(Ftab2[i]))
#     return maxims

# maxpos = range(len(Ftab2))

# maxbpl = np.array(list(map(maxbplvals, maxpos)))
# #%%
# fbplo = np.vstack((np.log(fs), maxbpl)).T

# plt.figure(figsize=(6, 9)) 
# plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]), color = "lime", linewidth=2.5)
# plt.title("LISA BPLS sensitivity curve")
# plt.ylim(1e-14, 1e-5)
# plt.xlabel('f (Hz)')
# plt.ylabel(r"$\Omega_{gw}$")
# plt.grid(True)
# plt.xscale('log')
# plt.show()


# plt.figure(figsize=(6, 9))
# plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]), label = "BPLS curve", color = "lime", linewidth=2.5)
# plt.loglog(freqvals, sigvals, label = "Nominal Curve", color = "indigo", linewidth=2.5)
# plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]), label = "PLS Curve", color = "orangered", linewidth=2.5)
# plt.grid(True)
# plt.title("LISA sensitivity curves")
# plt.legend()
# plt.xlabel('f (Hz)')
# plt.ylabel(r"$\Omega_{gw}$")
# plt.xscale('log')
# plt.ylim(1e-14, 1e-4)
# plt.show()
# #%%
# plt.figure(figsize=(6, 9))
# plt.loglog(freqvals, sigvals, label = "Nominal Curve", color = "indigo", linewidth=2.5)
# plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]), label = "BPLS curve", color = "lime", linewidth=2.5)
# plt.grid(True)
# plt.title("LISA BPLS Curves")
# plt.legend()
# plt.xlabel('f (Hz)')
# plt.ylabel(r'$\Omega_{gw}$', fontsize = 12)
# plt.xscale('log')
# plt.ylim(1e-14, 1e-4)
# # plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/LISAnomBPLS.png', bbox_inches='tight')
# plt.show()
#%%
######################
#LogNS curves
######################
#ET Logns

def logn(f, fstar, sig):
    res = np.exp(-1/(2*10**sig)*(np.log(f/fstar))**2)
    return res


def AlogETmin(fstar, sig):
    integrand = lambda f, fstar, sig :(logn(f, fstar, sig)/sigp(f))**2
    I1 = quad(integrand, 1.6, 100, args=(fstar, sig))[0]
    I2 = quad(integrand, 100, ffmax, args = (fstar, sig))[0]
    res = snr5/np.sqrt(2*T*sum((I1,I2)))
    return res


lf = np.linspace(elminet, elmaxet, itera)
flf = 10**lf
sigma = np.linspace(-1, 0, 5)




vals = np.array(np.meshgrid(flf, sigma)).T.reshape(-1,2)
aminvals = np.array(list(map(lambda args: AlogETmin(*args), vals)))

atab = np.vstack((vals.T, aminvals)).T.reshape(-1,len(sigma), 3)

def ftab(i, j, k):
    res = atab[i, j, 2]*logn(flf[k], atab[i, j, 0], atab[i, j, 1])
    return res

i = range(len(flf))
j = range(len(sigma))
k = range(len(flf))

coords = np.array(np.meshgrid(i, j, k)).T.reshape(-1,3)

Ftabetlog = np.array(list(map(lambda args: ftab(*args), coords))).reshape(len(flf),len(flf),len(sigma))

def maxlogvals(l):
    maximslog = np.log(np.max(Ftabetlog[l]))
    return maximslog


maxposlog = range(len(Ftabetlog))
maxlog = np.array(list(map(maxlogvals, maxposlog)))
flogplot = np.vstack((np.log(flf), maxlog)).T

#%%
plt.figure(figsize=(6, 9))
plt.loglog(np.exp(flogplot[:,0]), np.exp(flogplot[:,1]), color = "aqua", linewidth=2.5)
plt.title("LogNS curve ET")
plt.xlabel('f (Hz)')
plt.ylabel(r"$\Omega_{gw}$")
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()



plt.figure(figsize=(6, 9))
plt.loglog(fvalsET, sigETvals, color = "indigo", label = "Nominal", linewidth=2.5)
plt.loglog(np.exp(flogplot[:,0]), np.exp(flogplot[:,1]), color = "aqua", label = "LogNs", linewidth=2.5)
plt.legend()
plt.title("Nominal and LogNs curve for ET")
plt.ylabel(r"$\Omega_{gw}$")
plt.xlabel("f (Hz)")
plt.ylim(10**(-13),10**(-5))
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/ETnomlognslisa.png', bbox_inches='tight')
plt.show()
#%%
#LISA logns plots

    
def lognL(f, fstar, sig):
    res = np.exp(-1/(2*10**sig)*(np.log(f/fstar))**2)
    return res

def aminlogL(fstar, sig):
    integrand = lambda f, fstar, sig: (lognL(f, fstar, sig)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-3), args=(fstar, sig))[0]
    I2 = quad(integrand, 10**(-3), 10**(-1), args=(fstar, sig))[0]
    res = snr5/np.sqrt(2*T*sum((I1,I2)))
    return res

   

ls = np.linspace(elminL, elmaxL, itera)
fls = 10**ls
sigma = np.linspace(-1, 0, 5)


valslog = np.array(np.meshgrid(fls, sigma)).T.reshape(-1,2)
aminvals = np.array(list(map(lambda args: aminlogL(*args), valslog)))

atablog = np.vstack((valslog.T, aminvals)).T.reshape(-1,len(sigma), 3)

def fLlogtab(i, j, k):
    res = atablog[i, j, 2]*lognL(fls[k], atablog[i, j, 0], atablog[i, j, 1])
    return res


i = range(len(fls))
j = range(len(sigma))
k = range(len(fls))

coords = np.array(np.meshgrid(i, j, k)).T.reshape(-1,3)

Ftabloglisa = np.array(list(map(lambda args: fLlogtab(*args), coords))).reshape(len(fls),len(fls),len(sigma))

def maxlogvalsL(l):
    maximslog = np.log(np.max(Ftabloglisa[l]))
    return maximslog


maxposlog = range(len(Ftabloglisa))
maxlogvals = np.array(list(map(maxlogvalsL, maxposlog)))
maxlogL = maxlogvals

flogplotL = np.vstack((np.log(fls), maxlogL)).T


#%%
plt.figure(figsize=(6, 9))
plt.loglog(np.exp(np.log(fls)), np.exp(maxlogL), color = "aqua", linewidth=2.5)
plt.title("LogNS curve LISA")
plt.xlabel('f (Hz)')
plt.ylabel(r"$\Omega_{gw}$")
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()



plt.figure(figsize=(6, 9))
plt.loglog(freqvals, sigvals, color = "indigo", label = "Nominal", linewidth=2.5)
plt.loglog(np.exp(np.log(fls)), np.exp(maxlogL), color = "aqua", label = "LogNs", linewidth=2.5)
plt.title("Nominal and LogNs curve for LISA")
plt.xlabel('f (Hz)', fontsize = 16)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/LISAnomlognset.png', bbox_inches='tight')
plt.show()
#%%
# plt.loglog(np.exp(np.log(flf)), np.exp(maxlog), color = "aqua", label = "LogNs", linewidth=2.5)
# plt.loglog(np.exp(fbploET[:,0]), np.exp(fbploET[:,1]), color = "orangered", label = "PLS", linewidth=2.5)
# plt.loglog(fvalsET, sigETvals, color = "indigo", label = "Nominal", linewidth=2.5)
# plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]), color = "lime", label = "BPLS", linewidth=2.5)
# plt.legend()
# plt.ylabel(r"$\Omega_{gw}$")
# plt.xlabel('f (Hz)')
# plt.title("ET Sensitivity Curves")
# plt.grid(True)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()



# plt.loglog(np.exp(np.log(fls)), np.exp(maxlogL), color = "aqua", label = "LogNs", linewidth=2.5)
# plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]), label = "BPLS curve", color = "lime", linewidth=2.5)
# plt.loglog(freqvals, sigvals, label = "Nominal Curve", color = "indigo", linewidth=2.5)
# plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]), label = "PLS Curve", color = "orangered", linewidth=2.5)
# plt.grid(True)
# plt.title("LISA sensitivity curves")
# plt.legend()
# plt.xlabel('f (Hz)')
# plt.ylabel(r"$\Omega_{gw}$")
# plt.xscale('log')
# plt.show()
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
#combine pls curves
fstar = 1.1
ntmin = -9/2
ntmax = 9/2

def Amincomb(nt):
    integrand = lambda f, nt:((f/fstar)**(nt)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-4), args=(nt))[0]
    I2 = quad(integrand, 10**(-4), 10**(0), args=(nt))[0]
    I3 = quad(integrand, 10**(0), 10, args=(nt), limit = 1000)[0]
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


plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]), color = "orangered", linewidth=2.5)
plt.xscale('log')
plt.xlim(ffmin, ffmax) 
plt.xlabel('f (Hz)')
plt.ylabel(r"$\Omega_{gw}$")
plt.title('Combined PLS curve for LISA and ET')
plt.grid(True)
plt.show()
#%%

plt.loglog(otog[:,0], nom, color = "indigo", label = "Nominal", linewidth=2.5)
plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]), color = "orangered", label = "PLS", linewidth=2.5)
plt.title("Nominal and PLS curves ")
plt.legend(loc = (0.1,0.7))
plt.grid(True) 
plt.xlim(ffmin, ffmax) 
plt.xlabel('f (Hz)')
plt.ylabel(r'$\Omega_{gw}$')
plt.show()
#%%

plt.figure(figsize=(6, 9))
plt.loglog(otog[:,0], nom , color = "indigo", label = "Nominal", linewidth=2.5)
plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]), color = "orangered", label = "Combined PLS", linewidth=2.5)
plt.loglog(np.exp(flogom[:,0]), np.exp(flogom[:,1]), ':',color = "teal", label = 'LISA PLS', linewidth=2.5)
plt.loglog(np.exp(flogomET[:,0]), np.exp(flogomET[:,1]), ':',color = "black", label = 'ET PLS', linewidth=2.5)
plt.title("Nominal and PLS curves ", fontsize = 16)
plt.legend(loc = (1.05,0.5), fontsize = 14)
plt.grid(True) 
plt.xlim(ffmin, ffmax) 
plt.xlabel(r'$f$ (Hz)', fontsize = 16)
plt.ylabel(r'$\Omega_{gw}$', fontsize = 16)
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/CombineNomPLSwold.png', bbox_inches='tight')
plt.show()
#%%
#bpls
# def combbpl(f, fstar, n1, n2):
#     s = 10
#     res = (f/fstar)**n1 * (1/2+1/2*(f/fstar)**s)**(-(n1-n2)/s)
#     return res 


# def Abplmincomb(fs, n1, n2):
#     integrand = lambda f, fs, n1, n2:(combbpl(f, fs, n1, n2)/Ohms(f))**2
#     I1 = quad(integrand, ffmin, 10**(-3), args=(fs, n1, n2))[0]
#     I2 = quad(integrand, 10**(-3), 10**(-1), args=(fs, n1, n2))[0]
#     integrand2 = lambda f, fs, n1, n2:(combbpl(f, fs, n1, n2)/sigp(f))**2
#     I3 = quad(integrand2, 1.6, 10**(1), args=(fs, n1, n2))[0]
#     I4 = quad(integrand2, 10**(1), 445, args=(fs, n1, n2))[0]
#     res = snr5/np.sqrt(2*T*sum((I1,I2,I3,I4)))
#     return res   

# step = (ntmax-ntmin)/itera

# elc = np.linspace(elmin, elmax, itera)

# n1c = np.arange(ntmin, ntmax, step)
# n2c = np.arange(ntmin, ntmax, step)

# fsc = 10**elc



# inputsc = np.array(np.meshgrid(fsc, n2c, n1c)).T.reshape(-1,3)
# #This makes it so n1r is in the second column
# # so inputs(fs, n1r, n2r)
# inputsc[:,[0,1,2]] = inputsc[:,[0,2,1]]


# Amin4 = np.array(list(map(lambda args: Abplmincomb(*args), inputsc)))
# #%%
# Atab4 = np.vstack((inputsc.T, Amin4)).T
# Atab4 = Atab4.reshape(len(fsc),len(n1c),len(n1c),4)
# ic = range(len(fsc))
# jc = range(len(n1c))
# kc = range(len(n2c))
# mc = range(len(fsc))
# coordsc = np.array(np.meshgrid(ic,jc,kc, mc)).T.reshape(-1,4)

# #%%
# def fbpltabcomb(i, j, k, m):
#     res = []
#     bplres = combbpl(fsc[m], Atab4[i,j,k,0], Atab4[i,j,k,1], Atab4[i,j,k,2])
#     res.append([Atab4[i,j,k,3]*bplres])
#     return res



# Ftab4 = np.array(list(map(lambda args: fbpltabcomb(*args), coordsc))).reshape(len(fsc),len(fsc),len(fsc),len(fsc))
   

# #%%
# maximsc = []
# def combmaxbplvals(i):
#     maximsc = np.log(np.max(Ftab4[i]))
#     return maximsc

# combmaxpos = range(len(Ftab4))
# maxbplcomb = np.array(list(map(combmaxbplvals, combmaxpos)))
# combfbplo = np.vstack((np.log(fsc), maxbplcomb)).T



# plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]), color = "lime", linewidth=2.5)
# plt.title("BPLS Curve for LISA and ET")
# plt.ylabel(r"$\Omega_{gw}$")
# plt.xlabel("f (Hz)")
# plt.yscale('log')
# plt.xscale('log')
# plt.xlim(ffmin, ffmax)
# plt.grid(True)
# plt.show()

#%%
#combine log curves
def logncomb(f, fstar, sig):
    res = np.exp(-1/(2*10**sig)*(np.log(f/fstar))**2)
    return res

def aminlogcomb(fstar, sig):
    integrand = lambda f, fstar, sig: (logncomb(f, fstar, sig)/Ohms(f))**2
    I1 = quad(integrand, ffmin, 10**(-3), args=(fstar, sig))[0]
    I2 = quad(integrand, 10**(-3), 10**(-1), args=(fstar, sig))[0]
    I3 = quad(integrand, 10**(-1), 1e0, args=(fstar, sig))[0]
    I4 = quad(integrand, 1e0, 0.5e1, args=(fstar, sig))[0]
    I5 = quad(integrand, 0.5e1, ffmax, args=(fstar, sig))[0]
    integrand2 = lambda f, fstar, sig: (logncomb(f, fstar, sig)/sigp(f))**2
    I6 = quad(integrand2, ffmin, 1e-1, args=(fstar, sig))[0]
    I7 = quad(integrand2, 1e-1, ffmax, args=(fstar, sig))[0]
    res = snr5/np.sqrt(2*T*sum((I1, I2, I3, I4, I5, I6, I7)))
    return res 
itera = 200

lsc = np.linspace(elmin, elmax,itera)
flsc = 10**lsc
sigma = np.linspace(-1, 0, 5)


valslogcom = np.array(np.meshgrid(flsc, sigma)).T.reshape(-1,2)
aminvalscom = np.array(list(map(lambda args: aminlogcomb(*args), valslogcom)))

atablogcom = np.vstack((valslogcom.T, aminvalscom)).T.reshape(-1,len(sigma), 3)

def fLlogtabc(i, j, k):
    res = atablogcom[i, j, 2]*logncomb(flsc[k], atablogcom[i, j, 0], atablogcom[i, j, 1])
    return res

i = range(len(flsc))
j = range(len(sigma))
k = range(len(flsc))

coords = np.array(np.meshgrid(i, j, k)).T.reshape(-1,3)

Ftablogcomb = np.array(list(map(lambda args: fLlogtabc(*args), coords))).reshape(len(flsc),len(flsc),len(sigma))

def maxlogvalsc(l):
    maximslog = np.log(np.max(Ftablogcomb[l]))
    return maximslog


maxposlog = range(len(Ftablogcomb))
maxlogcom = np.array(list(map(maxlogvalsc, maxposlog)))
flogplotc = np.vstack((np.log(flsc), maxlogcom)).T
#%%
# plt.loglog(np.exp(flogplotc[:,0]), np.exp(flogplotc[:,1]), color = "aqua", label = "LogNs", linewidth=2.5)
# plt.title(" Combined LogNS curve for LISA and ET")
# plt.xlim(ffmin, ffmax)
# plt.xlabel('f (Hz)')
# plt.ylabel(r"$\Omega_{gw}$")
# plt.grid(True)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

plt.figure(figsize=(6, 9))
plt.loglog(otog[:,0], nom, label = "Nominal", color = "indigo", linewidth=2.5)
plt.loglog(np.exp(flogplotc[:,0]), np.exp(flogplotc[:,1]), color = "aqua", label = " Combined LogNs", linewidth=2.5)
plt.loglog(np.exp(flogplot[:,0]), np.exp(flogplot[:,1]), ':',color = "black", label = "ET LogNs", linewidth=2.5)
plt.loglog(np.exp(np.log(fls)), np.exp(maxlogL),':', color = "teal", label = "LISA LogNs", linewidth=2.5)
plt.title(" Combined LogNS curve for LISA and ET",fontsize = 14)
plt.legend(loc = (0.02,0.8),fontsize = 14)
plt.xlim(ffmin, ffmax)
plt.xlabel('f (Hz)',fontsize = 14)
plt.ylabel(r"$\Omega_{gw}$",fontsize = 16)
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/CombineNomlogNswold.png', bbox_inches='tight')
plt.show()
#%%

# plt.loglog(otog[:,0], nom, label = "Nominal Curve", color = "indigo", linewidth=2.5)
# plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]), label = "BPLS", color = "lime", linewidth=2.5)
# plt.ylabel(r"$\Omega_{gw}$")
# plt.title("Nominal and BPLS Curve for LISA and ET")
# plt.xlabel("f (Hz)")
# plt.legend(loc = (0.1,0.7))
# plt.yscale('log')
# plt.xscale('log')
# plt.xlim(ffmin, ffmax)
# plt.grid(True)
# plt.show()


#%%
# plt.loglog(otog[:,0], nom, label = "Nominal", color = "indigo", linewidth=2.5)
# plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]), label = "BPLS", color = "lime", linewidth=2.5)
# plt.loglog(np.exp(flogomcomb[:,0]), np.exp(flogomcomb[:,1]), label = "PLS", color = "orangered", linewidth=2.5)
# plt.loglog(np.exp(flogplotc[:,0]), np.exp(flogplotc[:,1]), color = "aqua", label = "LogNs", linewidth=2.5)
# plt.ylabel(r"$\Omega_{gw}$")
# plt.title("Curves for LISA and ET")
# plt.legend(loc = (0.1,0.7))
# plt.xlabel("f (Hz)")
# plt.yscale('log')
# plt.xscale('log')
# plt.xlim(ffmin, ffmax)
# plt.grid(True)
# plt.show()

#%%
# combfbplo = np.load("Ftabcomb.npy")
# fbploET = np.load("FtabET.npy")
# fbplo = np.load("FtabLISA.npy")

#%%
# plt.figure(figsize=(6, 9))
# plt.loglog(otog[:,0], nom, label = "Nominal Curve", color = "indigo", linewidth=2.5)
# plt.loglog(np.exp(combfbplo[:,0]), np.exp(combfbplo[:,1]), label = "Combined BPLS", color = "lime", linewidth=2.5)
# plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]),':', color = "teal", label = 'LISA BPLS', linewidth=2.5)
# plt.loglog(np.exp(fbploET[:,0]), np.exp(fbploET[:,1]), ':',color = "black", label = " ET BPLS", linewidth=2.5)
# plt.ylabel(r"$\Omega_{gw}$", fontsize = 16)
# plt.title("Nominal and BPLS Curve for LISA and ET", fontsize = 16)
# plt.xlabel("f (Hz)", fontsize = 16)
# plt.legend(loc = (1.05,0.5), fontsize = 14)
# plt.yscale('log')
# plt.xscale('log')
# plt.xlim(ffmin, ffmax)
# plt.grid(True)
# # plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/CombineNomBPLSwold.png', bbox_inches='tight')
# plt.show()
# plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/CombineNomBPLSwold.png')
# np.save("Ftab4.npy", Ftab4)
# np.save("Atab4.npy", Atab4)
# np.save("Amin4.npy", Amin4)

#%%





















