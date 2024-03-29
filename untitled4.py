#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:58:08 2024

@author: alisha
"""

#%%
covm = np.linalg.inv((FM2))
# covm[0,0]*= 1e-26
# covm[0,1]*= 1e-13
# covm[1,0]*= 1e-13
# covm[1,1]*= 1
#covb = np.linalg.inv((FMt))

means = np.array([5,2/3])
nsamp = int(1E6)
samps = np.random.multivariate_normal(means, covm, size=nsamp)
#samps2 = np.random.multivariate_normal(means, covb, size=nsamp)
names = [r'\Omega_*',r'nt']
labels =  [r'\Omega_*',r'nt']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'LISA Only')
#samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='LISA Only')

g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=14
g.settings.legend_fontsize = 16
g.settings.axes_labelsize = 16
g.triangle_plot([samples],# samples2],
                filled=True, markers={r'\Omega_*': means[0],'nt': means[1]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of LISA')
#%%
covm = np.linalg.inv((FM2))
# covm[0,0]*= (5*10**(-13))**(-2)
# covm[0,1]*=( 5*10**(-13))**(-1)
# covm[1,0]*= (5*10**(-13))**(-1)
# covm[1,1]*= 1
#covb = np.linalg.inv((abs(FM2)))
eigv, eigenve = np.linalg.eig(covm)
means = np.array([1,2/3])
nsamp = int(1E6)
samps = np.random.multivariate_normal(means, covm, size=nsamp)
#samps2 = np.random.multivariate_normal(means, covb, size=nsamp)
names = [r'\Omega_*',r'nt']
labels =  [r'\Omega_* \times 5 \times10^{-13}',r'nt']
samples = MCSamples(samples=samps,names = names, labels = labels, label = 'LISA')
#samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='LISA Only')

g = plots.get_subplot_plotter(subplot_size=5)
g.settings.axes_fontsize=14
g.settings.legend_fontsize = 16
g.settings.axes_labelsize = 16
g.triangle_plot([samples],
                filled=True, markers={r'\Omega_*': means[0],'nt': means[1]}, title_limit=1)
plt.suptitle(r'Fisher Analysis for SNR of LISA')