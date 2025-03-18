import matplotlib.pyplot as plt
import numpy as np


TabA = np.loadtxt('/Users/alisha/Documents/LISA_ET/TwoP/TabA.dat')
TabB = np.loadtxt('/Users/alisha/Documents/LISA_ET/TwoP/TabB.dat')
TabC = np.loadtxt('/Users/alisha/Documents/LISA_ET/TwoP/TabC.dat')
TabD = np.loadtxt('/Users/alisha/Documents/LISA_ET/TwoP/TabD.dat')

#%%

# Change the part in the " " to change the titles, filenames, and legends
# For each of the response functions

# Table names
taba = "TabA"
tabb = "TabB"
tabc = "TabC"
tabd = "TabD"
tabc1 = "TabC1"
tabd1 = "TabD1"

#For the combined graphs
tabcc1 = "TabC_C1_comb"
tabdd1 = "TabD_D1_comb"
comb = "MultiFunc"

#Graph locations
# Just delete filepath if dont want it
# IT MUST CONTAIN {nom} TO UPDATE THE FILENAME TO MATCH
# EACH RESPONSE FUNCTION
save = "/Users/alisha/Documents/LISA_ET/TwoP/TwoP{nom}.png"

#%%
plt.figure(figsize=(6, 9))
plt.loglog(TabA[:,0], TabA[:,1], color = "indigo", linewidth=2.5)
plt.grid(True)
plt.title("{nom} Response Function".format(nom = taba), fontsize = 16)
plt.xlabel('f (Hz)', fontsize = 16)
plt.ylabel(r'$\mathcal{R}$', fontsize = 20)
plt.savefig(save.format(nom = taba), bbox_inches='tight')


#%%
plt.figure(figsize=(6, 9))
plt.loglog(TabB[:,0], TabB[:,1], color = "indigo", linewidth=2.5)
plt.grid(True)
plt.title("{nom} Response Function".format(nom = tabb), fontsize = 16)
plt.xlabel('f (Hz)', fontsize = 16)
plt.ylabel(r'$\mathcal{R}$', fontsize = 20)
plt.savefig(save.format(nom = tabb), bbox_inches='tight')


#%%

plt.figure(figsize=(6, 9))
plt.loglog(TabC[:,0], TabC[:,1], color = "red", linewidth=2.5, label="{nom}".format(nom = tabc))
plt.grid(True)
plt.title("{nom} Response Function".format(nom = tabc), fontsize = 16)
plt.xlabel('f (Hz)', fontsize = 16)
plt.ylabel(r'$\mathcal{R}$', fontsize = 20)
plt.savefig(save.format(nom = tabc), bbox_inches='tight')

#%%
plt.figure(figsize=(6, 9))
plt.loglog(TabD[:,0], TabD[:,1], color = "blue", linewidth=2.5)
plt.ylim(1e-10, 1e0)
plt.xlim(0.05,40)
plt.grid(True)
plt.title("{nom} Response Function".format(nom = tabd), fontsize = 16)
plt.xlabel('f (Hz)', fontsize = 16)
plt.ylabel(r'$\mathcal{R}$', fontsize = 20)
plt.savefig(save.format(nom = tabd), bbox_inches='tight')


#%%
# Better to use the function than the datafile here
def c1(x, x1s, sig):
    res = (1 / 4**(1/sig)) * (9/20) * (1/2 + 1/2 * (x/x1s)**sig)**(-2/sig)
    return res


x = np.linspace(0.005,100,1000)
y = c1(x, 1.25, 3)

plt.figure(figsize=(6, 9))
plt.loglog(x, y,'--' ,color = "red", linewidth=2.5, label="{nom}".format(nom = tabc1))
plt.grid(True)
plt.title("{nom} Response Function".format(nom = tabc1), fontsize = 16)
plt.xlabel('f (Hz)', fontsize = 16)
plt.ylabel(r'$\mathcal{R}$', fontsize = 20)
plt.savefig(save.format(nom = tabc1), bbox_inches='tight')

#%%

# plt.figure(figsize=(6, 9))
plt.loglog(TabC[:,0], TabC[:,1],'--' , color = "red", linewidth=1.5, label=r"$\mathcal{R}_{A^\ell, E^\ell}$")
plt.loglog(x, y,color = "red", linewidth=1.5, label=r"$\mathcal{R}^{fit}_{A^\ell, E^\ell}$")
plt.grid(True)
plt.xlim(0.005,30)
plt.legend(fontsize = 16, loc=3)
# plt.title("{nom1} and {nom2} Response Function".format(nom1 = tabc, nom2 = tabc1), fontsize = 16)
plt.xlabel(r'$f/f_\star$', fontsize = 16)
plt.ylabel(r'$\mathcal{R}$', fontsize = 20)
plt.savefig('/Users/alisha/Documents/LISA_ET/TwoP/redrep.png', bbox_inches='tight')


#%%
def d1(x, x1s, sig):
    res = 2**(-(8/sig)) * (1/10) * (x/x1s)**6 * (1/2 + 1/2 * (x/x1s)**sig)**(-8/sig)
    return res

m = np.linspace(0.05,100,1000)
n = d1(m, 2.8, 6)

plt.figure(figsize=(6, 9))
plt.loglog(m, n, '--',color = "blue", linewidth=2.5, label="{nom}".format(nom = tabd1))
plt.grid(True)
plt.title("{nom} Response Function".format(nom = tabd1), fontsize = 16)
plt.xlabel(r'$f (Hz)$', fontsize = 16)
plt.ylabel(r'$\mathcal{R}$', fontsize = 20)
plt.savefig(save.format(nom = tabd1), bbox_inches='tight')

#%%

# plt.figure(figsize=(6, 9))
plt.loglog(TabD[:,0], TabD[:,1],'--', color = "blue", linewidth=1.5, label=r"$\mathcal{R}_{T^\ell}$")
plt.loglog(m, n, color = "blue", linewidth=1.5, label=r"$\mathcal{R}^{fit}_{T^\ell}$")
plt.grid(True)
plt.ylim(1e-14, 1e-1)
plt.xlim(0.05,40)

plt.legend(fontsize = 16, loc=4)
# plt.title("{nom1} and {nom2} Response Function".format(nom1 = tabd, nom2 = tabd1), fontsize = 16)
plt.xlabel(r'$f/f_\star$', fontsize = 16)
plt.ylabel(r'$\mathcal{R}$', fontsize = 20)
plt.savefig('/Users/alisha/Documents/LISA_ET/TwoP/bluerep.png', bbox_inches='tight')



#%%

plt.figure(figsize=(6, 9))
plt.loglog(m, n, '--',color = "blue", linewidth=2.5, label="{nom}".format(nom = tabd1))
plt.loglog(TabD[:,0], TabD[:,1], color = "blue", linewidth=2.5, label="{nom}".format(nom = tabd))
plt.loglog(TabC[:,0], TabC[:,1], color = "red", linewidth=2.5, label="{nom}".format(nom = tabc))
plt.loglog(x, y,'--' ,color = "red", linewidth=2.5, label="{nom}".format(nom = tabc1))
plt.ylim(1e-6, 1e0)
plt.xlim(0.05,40)

plt.grid(True)
plt.legend(fontsize = 16, loc=6)
plt.title("{nom1} Response Function".format(nom1 = comb), fontsize = 16)
plt.xlabel('f (Hz)', fontsize = 16)
plt.ylabel(r'$\mathcal{R}$', fontsize = 20)
plt.savefig(save.format(nom = comb), bbox_inches='tight')





