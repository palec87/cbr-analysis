# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:27:14 2020
Takes care of various fitting procedures, which do
similar, but not exactly the same things.
@author: David Palecek
"""

import numpy as np
import scipy.optimize as optim
from inspect import signature
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from analysis.helpers import support as sup
from time import perf_counter


# ------------------------------------------------------ #
# ---- fitting few kinetics, all nonlinear params ------ #
# ------------------------------------------------------ #
def fit_kinetics(x_data, y_data,
                 n_exp=1, init_par=None, const=None,
                 **kwargs):
    """fits one or few kinetics to sum of exponentials.
    No models involved.

    Args:
        x_data (list/tuple): x axis data
        y_data (list/tuple): y axis data
        n_exp (int, optional): Number of exponentials. Defaults to 1.
        init_par ([type], optional): Set of initial fitting parametres.
            Defaults to None.
        const ([type], optional): Selects if const is added as a free
            parameter. Defaults to None.

    Kwargs:
        bounds: set of bounds for parameters.
            Defaults to (-np.inf, np.inf).

    Returns:
        list: list of least_square fit outputs.
    """
    bounds = kwargs.get('bounds', (-np.inf, np.inf))
    fit = []
    for j in range(len(x_data)):
        if init_par is None:
            p0 = []
            for i in range(n_exp):
                p0.extend([3*(np.random.random()-0.5),
                          100*np.random.random()])
        else:
            p0 = list(init_par)

        if const is not None:
            if isinstance(const, (list, tuple)):
                p0.append(const[j])
            else:
                p0.append(const)
        fit.append(optim.least_squares(res_kin, p0,
                                       args=(y_data[j],
                                             x_data[j],
                                             n_exp),
                                       bounds=bounds))
    return fit


def exp_model(par, x, n):
    amp = par[::2]
    tau = par[1::2]
    # print(x, tau)
    func = np.sum([amp[k]*np.exp(-x/tau[k])
                  for k in range(n)],
                  axis=0, dtype=np.float64)
    try:  # adding constant to fit
        func = func + np.ones(len(x))*amp[n]
    except IndexError:
        pass
    return func


def res_kin(p, *args):
    y, *params = args
    return np.sqrt((y - exp_model(p, *params))**2)


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
        print('Using default init_par')
        p0 = []
        for i in range(n_exp):
            if gl_par[2*i] == 1:  # global parameter
                p0.extend([3 * (np.random.random()-0.5)])
            else:  # not global
                p0.extend([3 * (np.random.random()-0.5)
                           for k in range(ndat)])
            if gl_par[2*i+1] == 1:  # global
                p0.extend([100 * np.random.random()])
            else:  # not global
                p0.extend([100 * np.random.random()
                           for k in range(ndat)])
    else:
        p0 = list(init_par)

    if const is not None:
        if gl_par[2*n_exp] == 0:
            if isinstance(const, (tuple, list)) and len(const) == ndat:
                p0.extend(list(const))
            elif isinstance(const, (tuple, list)) and len(const) == 1:
                p0.extend(list(const) * ndat)
            elif isinstance(const, (float, int)):
                p0.extend([const] * ndat)
            else:
                raise AttributeError('Wrong length of const')
        else:
            if isinstance(const, (float, int)):
                p0.append(const)
            elif isinstance(const, (tuple, list)) and len(const) == 1:
                p0.extend(list(const))
            else:
                raise AttributeError('Wrong length of const')
    # checking correct length of params
    if len(p0) != int(len(gl_par) + (len(gl_par)-sum(gl_par))*(ndat-1)):
        print(len(p0), int(len(gl_par) + (len(gl_par)-sum(gl_par))*(ndat-1)))
        raise AttributeError('Wrong length of init_par')
        return

    # NL fitting
    fit = optim.least_squares(res_kin_gl, p0,
                              args=(y_data, gl_par, x_data, n_exp),
                              bounds=bounds)
    return fit


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
# ---- fitting few kinetics, ODE approach -------------- #
# ------------------------------------------------------ #
def fit_ode(x_data, y_data,
            model, p0_amp, p0_ode, const=None,
            **kwargs):
    tol = kwargs.get('tol', 1e-2)
    fit_total = []
    for i in range(len(x_data)):
        j = 0
        n_par = len(p0_ode+p0_amp)
        par_before = np.array(p0_ode + p0_amp)
        rel_change = np.array([1] * n_par)
        par_ode_in, par_amp_in = p0_ode, p0_amp
        lims = (x_data[i][0], x_data[i][-1])
        while any(i > tol for i in rel_change) and j < 50:
            # kinetic
            kin = solve_ivp(model_dict[model][0],
                            lims, par_amp_in,
                            args=par_ode_in,
                            t_eval=x_data[i]).y
            #  fit amplitude(s)
            if len(kin) == 1:
                par_amp_out, _ = optim.curve_fit(func, kin[0],
                                                 y_data[i])
            else:
                A = np.vstack(tuple(kin[:]))
                par_amp_out = optim.lsq_linear(A.T, y_data[i]).x
            # fit ODE(s)
            par_amp_out = tuple(par_amp_out*par_amp_in)
            t0 = perf_counter()
            fit = optim.least_squares(res_ode, par_ode_in,
                                      args=(y_data[i], model,
                                            lims, par_amp_out,
                                            x_data[i]),
                                      bounds=(0, 1e8),
                                      max_nfev=4)
            t1 = perf_counter()
            par_after = np.array(tuple(fit.x) + tuple(par_amp_out))
            rel_change = abs((par_before-par_after)/par_before)
            par_before = par_after
            par_ode_in = tuple(fit.x)
            par_amp_in = par_amp_out
            j += 1
            # print(f'ODE took: {t1-t0} secs.')
            if t1-t0 > 5:
                print('takes too long, try again.')
                break
            # print(f'params after fit: {par_after}')
            # print(rel_change, rel_change.any() > tol)
        fit_total.append((par_ode_in, par_amp_in))
    return fit_total


def res_ode(p, *args):
    y, model, lims, par_amp, x_range = args
    model_ode = solve_ivp(model_dict[model][0],
                          lims, par_amp,
                          args=p,
                          t_eval=x_range).y
    all_comp_sum = np.sum(model_ode, axis=0)
    res = (y - all_comp_sum)**2
    return res


# ------------------------------------ #
# -------ODE 2D approach ------------- #
# ------------------------------------ #
def fit_ode_2d(x_data, y_data,
               model, p0_amp, p0_ode, const=None,
               **kwargs):
    # tol = kwargs.get('tol', 1e-3)
    par_ode_in, par_amp_in = p0_ode, p0_amp
    lims = (x_data[0], x_data[-1])
    j = 0
    while j < 10:
        # kinetic
        kin = solve_ivp(model_dict[model][0],
                        lims, par_amp_in,
                        args=par_ode_in,
                        t_eval=x_data).y
        # fit amplitudes
        if len(kin) == 1:
            par_amp_out = []
            for i in range(y_data.shape[1]):
                _par, _ = optim.curve_fit(func, kin[0],
                                          y_data[:, i])
                par_amp_out.append(_par)
        else:
            par_amp_out = np.zeros((y_data.shape[1],
                                   model_dict[model][1]))
            for i in range(y_data.shape[1]):
                A = np.vstack(tuple(kin[:]))
                par_amp_out[i, :] = optim.lsq_linear(A.T,
                                                     y_data[:, i]).x
        # fit ODE(s)
        fit = optim.least_squares(res_ode_2d, par_ode_in,
                                  args=(y_data, model,
                                        lims, par_amp_out,
                                        x_data),
                                  bounds=(0, 1e8),
                                  max_nfev=4)
        j += 1
        print(j)
    return fit.x, par_amp_out


def res_ode_2d(p, *args):
    y, model, lims, par_amp, x_range = args
    model_ode = solve_ivp(model_dict[model][0],
                          lims, tuple([1] * model_dict[model][1]),
                          args=p,
                          t_eval=x_range).y
    data_sol = np.zeros(y.shape)
    for i in range(model_dict[model][1]):
        data_sol += np.outer(model_ode[i], par_amp[:, i])
    res = np.sum((y - data_sol)**2, axis=1)
    return res


def func(kin, m):
    return m*kin


def ode_solution(p, *args):
    model, lims, par_amp, x_range = args
    sol = solve_ivp(model_dict[model][0], lims, par_amp,
                    args=p,
                    t_eval=x_range)
    return sol


def rhs00(t, states, t0):
    '''single state decay'''
    s0 = states
    return -s0/t0


def rhs01(t, states, t0, t1):
    ''' Two states, no transfer'''
    s0, s1 = states
    return [-s0/t0,
            -s1/t1]


def rhs02(t, states, t0, t1, t2):
    ''' Two states, transfer from 1 ->2'''
    s0, s1 = states
    return [-s0/t0 - s0/t1,
            -s1/t2 + s0/t1]


# ------------------------------------------------------ #
# ---------- fitting for SVD script -------------------- #
# ------------------------------------------------------ #
def rotation(var, k, pos, T, P, DTT, C0, t, function):
    global C, R, V, calc, res
    k[pos] = var
    C = odeint(function, C0, t, args=(k,))
    R = T.T @ np.linalg.pinv(C.T)
    V = P @ R
    calc = V @ C.T
    res = (DTT-calc)
    if any(x < 0 for x in k):
        penalty = 1e6
    else:
        penalty = 0
    error = np.linalg.norm(DTT-calc, 'fro') + penalty
    return error


# ------------------------------------------------------ #
# ------------- helper functions ----------------------- #
# ------------------------------------------------------ #
def nest_data(data):
    """Nesting of the data for functions which handle both single or multiple
    data datasets in a form of list/tuple of arrays.

    Args:
        data (array/list/tuple): x/y data of a single line data, ie. a kinetic

    Returns:
        list/tuple/array: always nested output.
    """
    if any(isinstance(i, (list, tuple, np.ndarray))
           for i in data):
        pass   # nested data, do nothing
    else:  # nest data
        data = (data,)
    return data


def duplicate_nesting(x, y):
    """Ensures the same level of nesting for x and y data. Works for the case
    when single x axis used for multiple y data, ie. kinetics.

    Args:
        x (list/tuple/ndarray): x data, nested or not.
        y (list/tuple/ndarray): y data, nested or not.

    Raises:
        ValueError: If more x axes than y datasets.

    Returns:
        tuple/list/ndarray: nested x axes to match y datasets.
    """
    x = nest_data(x)
    if len(x) == len(y):
        pass
    elif len(x) > len(y):
        raise ValueError('more x axes than datasets')
    else:  # only case I need duplicate x axis
        x = tuple([x[0]]*len(y))
    return x


def cut_limits(x_data: tuple, y_data: tuple, x_lims: tuple):
    """Selecting range of x/y data based on x_lims.

    Args:
        x_data (tuple): x axis data
        y_data (tuple): y axis values
        x_lims (tuple): limits to select range from x and y

    Raises:
        ValueError: If x_lims have wrong shape

    Returns:
        list: limited x data, limited y data
    """
    # # extending x axis to match number of datasets.
    y_data = nest_data(y_data)
    n_dat = len(y_data)
    x_data = duplicate_nesting(x_data, y_data)
    print(f'number of datasets: {n_dat}.')
    # no limits, take all positive x
    x = []
    data = []
    if x_lims is None:
        for i in range(n_dat):
            x.append(x_data[i][x_data[i] > 0])
            data.append(y_data[i][x_data[i] > 0])
    # x limits global for all kinetics
    elif len(x_lims) == 2:
        for i in range(n_dat):
            beg, end = sup.get_idx(*x_lims, axis=x_data[i])
            x.append(x_data[i][beg:end+1])
            data.append(y_data[i][beg:end+1])
    # x limits for each kinetic specified
    elif len(x_lims) == n_dat*2:
        for i in range(n_dat):
            beg, end = sup.get_idx(*x_lims[2*i:2*i+2], axis=x_data[i])
            x.append(x_data[i][beg:end+1])
            data.append(y_data[i][beg:end+1])
    else:
        raise ValueError('Wrong shape of t_lims.')
        return
    return x, data


def group_par(params, bool_gl, n, size):
    idx_count = [size if item == 0 else item
                 for item in bool_gl]
    amp_count = idx_count[::2]
    tau_count = idx_count[1::2]
    amp, tau = [], []
    for i in range(n):
        par = list(params.copy())
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


def get_params_ode(model, par):
    sig = signature(model_dict[model][0])
    n_ode_params = len(sig.parameters)-2
    n_amp_params = model_dict[model][1]
    if par is None:
        par_out = (tuple([10000*np.random.random()
                          for i in range(n_ode_params)]),
                   tuple([2*np.random.random()-1
                          for i in range(n_amp_params)])
                   )
    else:
        # lengths are correct
        if (len(par[0]) == n_ode_params and
           len(par[1]) == n_amp_params):
            par_out = par
        else:
            print('Wrong number of input params: generate random ones for you')
            par_out = (tuple([10000*np.random.random()
                              for i in range(n_ode_params)]),
                       tuple([2*np.random.random()-1
                              for i in range(n_amp_params)])
                       )
    return par_out


def check_fit_params(obj):
    """Check if fit parameters exist and should be rewritten

    Args:
        obj (class Trs): time-resolved experiment
    """
    if obj._fitParams is None:
        obj._fitParams = []
        obj._fitData = []
    else:
        a = input('Rewrite old fits [y/n]?')
        if a == 'y':
            obj._fitParams = []
            obj._fitData = []
        else:
            print('I will append fit parametres to existing field')


# first is function, second is number of states
model_dict = {
              'one_state': (rhs00, 1),
              'two_states': (rhs01, 2),
              'two_states_transfer': (rhs02, 2)
            }

if __name__ == "__main__":
    # x = np.linspace(0, 99, 100)
    # par = (1, 5, 1, 50, -1, 200, 1)
    # data2fit = exp_model(par, x, n=3)
    # data2fit += np.random.normal(0, 0.07, len(x))

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

    x = [np.linspace(0, 99, 100),
         np.linspace(0, 49, 100),
         np.linspace(0, 199, 200)]
    data2fit = [exp_model(par, x[0], n=3),
                exp_model(par2, x[1], n=1),
                exp_model(par3, x[2], n=2)]

    # glob = [1, 1, 1, 1, 0]
    # fit = fit_kinetics_global(x, data2fit, gl_par=glob, n_exp=2, const=0.01)
    # fit_result = exp_model_gl(fit.x, bool_gl=glob, x=x, n=2)
    # print(fit.x)
    # for i in range(len(x)):
    #     # for glob = [1,0]
    #     # single_fit = exp_model([fit.x[0], fit.x[i+1]], x[i], 1)

    #     # for glob = [1,0,1,0]
    #     # single_fit = exp_model([fit.x[0], fit.x[i+1],
    #     #                         fit.x[4], fit.x[4+i+1]], x[i], 2)

    #     # for glob = [0,1,0,1]
    #     # single_fit = exp_model([fit.x[i], fit.x[3],
    #     #                         fit.x[4+i], fit.x[-1]], x[i], 2)

    #     # for glob = [1,1,1,1]
    #     # single_fit = exp_model([fit.x[0], fit.x[1],
    #     #                         fit.x[2], fit.x[3]], x[i], 2)

    #     # for glob = [1,1,1,1,1]
    #     # single_fit = exp_model([fit.x[0], fit.x[1],
    #     #                         fit.x[2], fit.x[3], fit.x[-1]], x[i], 2)

    #     # for glob = [1,1,1,1,0]
    #     single_fit = exp_model([fit.x[0], fit.x[1],
    #                             fit.x[2], fit.x[3],
    #                             fit.x[-len(x)+i]], x[i], 2)

    #     plt.plot(x[i], data2fit[i], 'o', label=i)
    #     plt.plot(x[i], fit_result[i], 'k-')
    # plt.legend()
    # plt.show()

    fit = fit_ode(x, data2fit,
                  'one_state',
                  p0_amp=(1,), p0_ode=(100,),
                  const=None)
    plt.plot(x[0], data2fit[0], 'o')
    plt.plot(x[0][1:], fit.y[0], 'k-')
    plt.show()
