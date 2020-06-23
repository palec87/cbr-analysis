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
from ..helpers import support as sup


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


# ------------------------------------------------------ #
# ---- fitting few kinetics, global -------------------- #
# ------------------------------------------------------ #
def fit_kinetics_global(x_data, y_data, gl_par,
                        n_exp=1, init_par=None, const=None,
                        **kwargs):
    '''
    fits one or few kinetics globally
    to sum of exponentials
    lifetimes are global params (Basically DAS)
    '''
    # basic checks of inputs
    bounds = kwargs.get('bounds', (-np.inf, np.inf))
    ndat = len(x_data)
    if init_par is None:
        p0 = []
        for i in range(n_exp):
            if gl_par[2*i] == 1:
                p0.extend([3 * (np.random.random()-0.5)])
            else:
                p0.extend([3 * (np.random.random()-0.5)
                           for k in range(ndat)])
            if gl_par[2*i+1] == 1:
                p0.extend([100 * np.random.random()])
            else:
                p0.extend([100 * np.random.random()
                           for k in range(ndat)])
    else:
        p0 = init_par

    if const is not None:
        if gl_par[2*n_exp] == 0:
            p0.extend([const] * ndat)
        else:
            p0.extend([const])
    # NL fitting
    fit = optim.least_squares(res_kin_gl, p0,
                              args=(y_data, gl_par, x_data, n_exp),
                              bounds=bounds)
    return fit


def group_par(params, bool_gl, n, size):
    idx_count = [size if item == 0 else item
                 for item in bool_gl]
    amp_count = idx_count[::2]
    tau_count = idx_count[1::2]
    amp, tau = [], []
    par = list(params.copy())
    for i in range(n):
        # amplitudes
        if amp_count[i] == size:
            amp.extend(par[:size])
            par = par[size:]
        else:
            amp.extend([par[0]]*size)
            par = par[1:]
        # lifetimes
        if tau_count[i] == size:
            tau.extend(par[:size])
            par = par[size:]
        else:
            tau.extend([par[0]]*size)
            par = par[1:]

    par_total = []
    for i in range(size):
        a = [amp[size*k + i] for k in range(n)]
        t = [tau[size*k + i] for k in range(n)]
        par_total.append((a, t))
    return par_total


def exp_model_gl(params, bool_gl, x, n):
    func_total = []
    ndat = len(x)
    # order parameters to list of tuples
    par = group_par(params, bool_gl, n, ndat)
    for i in range(ndat):
        func = np.sum([par[i][0][k] * np.exp(-x[i] / par[i][1][k])
                       for k in range(n)],
                      axis=0)
        # adding constant to the fit
        try:
            bool_gl[2*n]
        except IndexError:
            pass
        else:
            if bool_gl[2*n] == 1:
                func += np.ones(len(x[i])) * params[-1]
            else:
                func += np.ones(len(x[i])) * params[-ndat+i]
        func_total.append(func)
    return func_total


def res_kin_gl(p, *args):
    y, *params = args
    model = exp_model_gl(p, *params)
    y_flat = np.array([item
                       for sublist in y
                       for item in sublist])
    model_flat = np.array([item
                           for sublist in model
                           for item in sublist])
    return (y_flat - model_flat)**2


# ------------------------------------------------------ #
# ------------- helper functions ----------------------- #
# ------------------------------------------------------ #
def x_limits(x_data: tuple, y_data: tuple, t_lims: tuple):
    n_dat = len(y_data)
    # no limits, take all positive
    if t_lims is None:
        x = [x_data[x_data > 0] for trace in y_data]
        data = [trace[x_data > 0] for trace in y_data]
    # x limits global for all kinetics
    elif len(t_lims) == 2:
        beg, end = sup.get_idx(*t_lims, axis=x_data)
        x = [x_data[beg:end+1]] * n_dat
        data = [trace[beg:end+1] for trace in y_data]
    # x limits for each kinetic specified
    elif len(t_lims) == n_dat*2:
        x = []
        for i in range(n_dat):
            beg, end = sup.get_idx(*t_lims[2*i:2*i+2], axis=x_data)
            x.append(x_data[beg:end+1])
            data[i] = data[i][beg:end+1]
    else:
        print('Wrong shape of t_lims.')
        return
    return x, data
# ------------------------------------------------------ #
# ---- fitting few kinetics, ODE approach -------------- #
# ------------------------------------------------------ #


if __name__ == "__main__":
    x = np.linspace(0, 99, 100)
    par = (1, 5, 1, 50, -1, 200, 1)
    data2fit = exp_model(par, x, n=3)
    data2fit += np.random.normal(0, 0.07, len(x))

    # fit = fit_kinetics(x, data2fit, n_exp=2)

    # fit_result = exp_model(fit.x, x, 2)
    # fig0 = plt.figure()
    # plt.plot(data2fit, label='data')
    # plt.plot(fit_result, label='fit')
    # plt.legend()
    # plt.show()

    # print(fit)
    par = [1, 5, 1, 50, -1, 200, 0.1]
    par2 = par[:2] + [0.1]
    par3 = par[:4] + [0.1]
    print(par, par2, par3)
    x = [np.linspace(0, 99, 100),
         np.linspace(0, 49, 100),
         np.linspace(0, 199, 200)]
    data2fit = [exp_model(par, x[0], n=3),
                exp_model(par2, x[1], n=1),
                exp_model(par3, x[2], n=2)]

    glob = [1, 1, 1, 1, 0]
    fit = fit_kinetics_global(x, data2fit, gl_par=glob, n_exp=2, const=0.01)
    fit_result = exp_model_gl(fit.x, bool_gl=glob, x=x, n=2)
    print(fit.x)
    for i in range(len(x)):
        # for glob = [1,0]
        # single_fit = exp_model([fit.x[0], fit.x[i+1]], x[i], 1)

        # for glob = [1,0,1,0]
        # single_fit = exp_model([fit.x[0], fit.x[i+1],
        #                         fit.x[4], fit.x[4+i+1]], x[i], 2)

        # for glob = [0,1,0,1]
        # single_fit = exp_model([fit.x[i], fit.x[3],
        #                         fit.x[4+i], fit.x[-1]], x[i], 2)

        # for glob = [1,1,1,1]
        # single_fit = exp_model([fit.x[0], fit.x[1],
        #                         fit.x[2], fit.x[3]], x[i], 2)

        # for glob = [1,1,1,1,1]
        # single_fit = exp_model([fit.x[0], fit.x[1],
        #                         fit.x[2], fit.x[3], fit.x[-1]], x[i], 2)

        # for glob = [1,1,1,1,0]
        single_fit = exp_model([fit.x[0], fit.x[1],
                                fit.x[2], fit.x[3], fit.x[-len(x)+i]], x[i], 2)

        plt.plot(x[i], data2fit[i], 'o', label=i)
        plt.plot(x[i], fit_result[i], 'k-')
        plt.plot(x[i], single_fit, 'k:')
    plt.legend()
    plt.show()
