# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:27:14 2020
Takes care of various fitting procedures, which do
similar, but not exactly the same things.
@author: David Palecek
"""

import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt

# ------------------------------------------------------ #
# ---- fitting few kinetics, all nonlinear params ------ #
# ------------------------------------------------------ #


def fit_kinetics(x_data, y_data,
                 n_exp=1, init_par=None, const=None,
                 **kwargs):
    '''
    fits one to few kinetics to sum
    of exponentials
    - no models involved.

    Keyword arguments:
    bounds -- two tuple array
    '''
    bounds = kwargs.get('bounds', (-np.inf, np.inf))
    if init_par is None:
        p0 = []
        for i in range(n_exp):
            p0.extend([3*(np.random.random()-0.5),
                      100*np.random.random()])
    else:
        p0 = init_par

    if const is not None:
        p0.append(const)
    fit = optim.least_squares(res_kin, p0,
                              args=(y_data, x_data, n_exp),
                              bounds=bounds)
    return fit


def exp_model(par, x, n):
    amp = par[::2]
    tau = par[1::2]
    func = np.sum([amp[k]*np.exp(-x/tau[k])
                  for k in range(n)],
                  axis=0)
    try:  # adding constant to fit
        func = func + np.ones(len(x))*amp[n]
    except IndexError:
        pass
    return func


def res_kin(p, *args):
    y, *params = args
    return (y - exp_model(p, *params))**2


if __name__ == "__main__":
    x = np.linspace(0, 99, 100)
    par = (1, 5, 1, 50, -1, 200, 1)
    data2fit = exp_model(par, x, n=3)
    data2fit += np.random.normal(0, 0.07, len(x))

    fit = fit_kinetics(x, data2fit, n_exp=2)

    fit_result = exp_model(fit.x, x, 2)
    fig0 = plt.figure()
    plt.plot(data2fit, label='data')
    plt.plot(fit_result, label='fit')
    plt.legend()
    plt.show()

    print(fit)
