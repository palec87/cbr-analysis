# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:00:45 2020

@author: David Palecek

- Testing people's ability to fit some kind of TA data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import erf



## RHS used for the model
def rhs01(s, v, p0, p1, p2, p3): 
    return [-v[0]/p0 - v[0]/p1 + v[1]/p3,
            -v[1]/p2 + v[0]/p1 - v[1]/p3]

## control without the back transfer component
## Used by Victor
def rhs02(s, v, p0, p1, p2): 
    return [-v[0]/p0 - v[0]/p1,
            -v[1]/p2 + v[0]/p1]

# component 1, pentacene
spectra = np.loadtxt('TAS.txt')
wl = spectra[10:-30,0]
sp0 = spectra[10:-30,2]*1000

# component 2, p3ht fibers
sp1 = np.loadtxt('test_data.txt')[20,:]
wl1 = np.linspace(wl[0], wl[-1], len(sp1))
sp2 = np.interp(wl, wl1, sp1) 

t = np.linspace(-49.9,500, 166)
res0, res1 = np.zeros((len(t), )), np.zeros((len(t), ))

plt.plot(wl, sp0, label = 'comp 0')
plt.plot(wl, sp2, label = 'comp 1')
plt.title('Model spectral components')
plt.legend()
plt.show()

## solve ODE
par = (200,60,300,150)

res = solve_ivp(rhs01, (0.001,500),[1,0.3],args=par,t_eval=t[15:])
res_control = solve_ivp(rhs02, (0.001,500),[1,0.3],args=par[:-1],t_eval=t[15:])
res0[15:] = res.y[0]
res1[15:] = res.y[1]
res0conv = res0 * erf(t/5)
res1conv = res1 * erf(t/5)

## control plot, if the backtransfer is possible to get out
plt.plot(res.y[0], label = 'model')
plt.plot(res_control.y[0], label= 'no back transf.')
plt.title('Kin, comp 0')
plt.legend()
plt.show()

plt.plot(res.y[1], label = 'model')
plt.plot(res_control.y[1], label= 'no back transf.')
plt.title('Kin, comp 0')
plt.title('Kin, comp 1')
plt.legend()
plt.show()


###############
plt.plot(t, res0, ':', color = 'C0', label = 'kin 0')
plt.plot(t, res1, ':', color = 'C1', label = 'kin 1')
plt.plot(t, res0conv, color = 'C0', label = 'kin 0 + IRF')
plt.plot(t, res1conv, color = 'C1', label = 'kin 1 + IRF')
plt.xlim([-20,50])
plt.title('IRF convolution')
plt.legend()
plt.show()

comp0 = np.outer(sp0, res0conv)
comp1 = np.outer(sp2, res1conv)
noise = np.random.normal(0, 0.01, len(wl)*len(t)).reshape(comp0.shape)
full = comp0 + comp1 + noise

plt.contourf(t,wl,comp0, 31)
plt.colorbar()
plt.show()

plt.contourf(t,wl,comp1, 31)
plt.colorbar()
plt.show()

plt.contourf(t,wl,full, 51)
plt.colorbar()
plt.show()

# resulting and separate spectra
plt.plot(wl, sp0, label = 'sol1')
plt.plot(wl, sp2, label = 'sol2')
plt.plot(wl, full[:,17], label= 'early')
plt.plot(wl, full[:,-10], label= 'late')
plt.legend()
plt.show()

np.savetxt('dtt2fit.txt', full, fmt='%10.5f')
np.savetxt('wl.txt', wl, fmt='%10.5f')
np.savetxt('t.txt', t, fmt='%10.5f')

# ----------------------------------
#### test victor components 
# ------------------------------------
comp = np.loadtxt('components_victor.csv',
                  delimiter = ',')
print(comp.shape)
res_victor0, res_victor1 = np.zeros((len(t), )), np.zeros((len(t), ))
par_in = (84,62,264)
res_victor = solve_ivp(rhs02, (0.001,500),
                        [1,0.0],args=par_in,
                        t_eval=t[15:])

res_victor0[15:] = res_victor.y[0]
res_victor1[15:] = res_victor.y[1]
res_victor0conv = res_victor0 * erf(t/5)
res_victor1conv = res_victor1 * erf(t/5)

## control plot, if the backtransfer is possible to get out
plt.plot(res0conv, label = 'kin0 model')
plt.plot(res_victor0conv, label = 'kin0 Victor')
plt.title('Kin0 comparison')
plt.legend()
plt.savefig('kin0.png')
plt.show()

plt.plot(res1conv, label = 'kin1 model')
plt.plot(res_victor1conv, label = 'kin1 Victor')
plt.title('Kin1 comparison')
plt.legend()
plt.savefig('kin1.png')
plt.show()


## spectra ##
plt.plot(wl, sp0, label = 'sol1')
plt.plot(wl, comp[:,1], label = 'Victor 1')
plt.title('Comp 1 comparison')
plt.legend()
plt.savefig('spe0.png')
plt.show()

plt.plot(wl, sp2, label = 'sol2')
plt.plot(wl, comp[:,2], label = 'Victor 2')
plt.title('Comp 2 comparison')
plt.legend()
plt.savefig('spe1.png')
plt.show()


## Victor whole map ##
full_victor = (np.outer(comp[:,1], res_victor0conv) +
               np.outer(comp[:,2], res_victor1conv))

plt.contourf(t,wl,full_victor, 51)
plt.colorbar()
plt.title('Full TA, Victor')
plt.show()

plt.contourf(t,wl,full - full_victor, 51)
plt.colorbar()
plt.title('Residuals, Model-Victor')
plt.savefig('residuals.png')
plt.show()












