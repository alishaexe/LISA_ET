import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
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
RAnum = interp1d(f, Rae)
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

itera = 2000

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
    return 0.816**2 * res

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

#%%
fvalsET = np.logspace(0, 3,2000)#frequency values
sigETvals = np.array(list(map(etnomonly, fvalsET)))#The Omega_gw values from the ET data

# plt.figure(figsize=(6, 9)) 
# plt.loglog(fvalsET, sigETvals, color = "indigo", linewidth=2.5)
# plt.title("Nominal sensitivity curve ET")
# plt.ylabel(r"$\Omega_{gw}$")
# plt.xlabel("f (Hz)")
# #plt.ylim(10**(-9), 10**(-5))
# plt.yscale('log')
# plt.xscale('log')
# plt.xlim(10**0, 400)
# plt.grid(True)
# plt.show()
#%%
#Plots both curves overlayed each other

plt.figure(figsize=(6, 9)) 
plt.loglog(tabET2[:,0], tabET2[:,1], '--', label = "Numerical", color = "darkviolet", linewidth=2.5)
plt.title("Nominal Sensitivity curve of ET", fontsize = 16)
plt.loglog(fvalsET, sigETvals, '-',label = "Approximate", linewidth = 2.5,color = "indigo" )
plt.legend(fontsize = 16, loc=9)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 20)
plt.xlabel(r"$f$ (Hz)", fontsize = 20)
plt.tick_params(axis='both', which='major', labelsize=14) 

#plt.ylim(10**(-9), 10**(-5))
plt.yscale('log')
plt.xscale('log')
# plt.xlim(1, 445)
# plt.ylim(1e-9,1e-5)
plt.grid(True)
plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/ETnoms.png', bbox_inches='tight')
plt.show()

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

f2 = 4*fLisa/3


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
    return 1/np.sqrt(2)*res

og = np.array(list(map(lambda args: sigtab(*args), Rtab)),dtype = float)
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

def sigtab(f):
    const = 4*pi**2/(3*H0**2)
    res = 1/np.sqrt(2)*(N_AA(f)/(16*(np.sin(2*pi*f*L))**2  * (2*pi*f*L)**2*RAnum(f)))*const*f**3
    return res


freqvals = np.logspace(elminL, elmaxL, itera)   
sigvals = np.array(list(map(Ohms, freqvals)))
og = np.array(list(map(sigtab, freqvals)))

plt.figure(figsize=(6, 9)) 
plt.loglog(freqvals, og,'--' ,color = "darkviolet", label = "Numerical", linewidth=2.5)
plt.loglog(freqvals, sigvals, color = "indigo", label = r"Approximate", linewidth=2.5)
plt.ylabel(r"$\Omega_{gw}$", fontsize = 20)
plt.grid(True)
# plt.ylim(1e-11,1e-5)
# plt.xlim(1e-5,1.5e-1)
plt.legend(fontsize = 16)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.xlabel(r"$f$ (Hz)", fontsize = 20)
plt.title("Nominal sensitivity curve of LISA ", fontsize = 16)
plt.savefig('/Users/alisha/Documents/LISA_ET/Sensitivity Curves/LISAnoms.png', bbox_inches='tight')
plt.show()










