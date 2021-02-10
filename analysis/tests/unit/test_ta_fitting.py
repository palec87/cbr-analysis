""" testing fitting module
    - single fits
    - single fits, global
    - ODE fit
    - TODO: ODE fit, global
"""
import pytest
import numpy as np
# import matplotlib.pyplot as plt
# from analysis.experiments.ta import Ta
from analysis.modules import fitting as ft


@pytest.mark.parametrize('x_axis, data, tlim', [
    ([np.linspace(-5, 199, 205)], [np.ones(205)], (10, 20)),  # single datasets
    (np.linspace(-5, 199, 205), np.ones(205), (9, 19)),
    (np.linspace(-5, 11, 17), np.ones(17), None),

    # multiple datasets (list and tuples)
    ([np.linspace(-5, 11, 17), np.linspace(-4, 11, 16)],
     [np.ones(17), np.ones(16)], None),
    ((np.linspace(-5, 11, 17), np.linspace(-4, 11, 16)),
     (np.ones(17), np.ones(16)), None),
    ((np.linspace(-5, 11, 17), np.linspace(-4, 11, 16)),
     (np.ones(17), np.ones(16)), (1, 11)),
])
def test_cut_limits(x_axis, data, tlim):
    """Test for cutting x_axis and y data
    according to tlim

    Args:
        x_axis (list/tuple): x_axis for data
        data (list/tuple): data values
        tlim (two tuple or ): desired limits of x_axis
    """
    new_x, new_data = ft.cut_limits(x_axis, data, tlim)
    assert len(new_x) == len(new_data)
    assert [len(i) for i in new_x] == [11] * len(new_x)
    assert [len(i) for i in new_data] == [11] * len(new_data)


# testing limits, and parameter types
@pytest.mark.parametrize('x_axis, par, n_exp', [
    (np.linspace(-5, 199, 205), (1, 5), 1),
    (np.linspace(-5, 199, 205), [1, 5], 1),
    (np.linspace(-50, 1_000, 200), (-1, 250), 1),
    (np.linspace(-50, 2_000, 200), (-2, 250, 5, 700), 2),
    (np.linspace(-5, 199, 205), (1, 5), 1)
    ])
def test_single_kin(x_axis, par, n_exp):
    """test one and 2-exponential fitting

    Args:
        x_axis (array of single x): array of x
        par (tuple): amplitudes and taus
        n_exp (int): number of exponentials
    """
    data0 = ft.exp_model(par, x_axis, n=n_exp)
    fit = ft.fit_kinetics((x_axis,), (data0,),
                          n_exp=n_exp)
    print(fit[0].success)
    assert fit[0].success == 1
    # getting exaclty right params is not ensured
    # assert sorted(tuple(np.round(fit[0].x))) == sorted(par)


@pytest.mark.parametrize('x_axis, par, n_exp, const', [
    (np.linspace(-5, 199, 205), (1, 20), 1, (1,)),
    (np.linspace(-5, 199, 205), (1, 50), 1, 1),
    ])
def test_single_kin02(x_axis, par, n_exp, const):
    """testing adding constant to the exp fits

    Args:
        x_axis (array): x axis
        par (tuple): amplitudes and taus
        n_exp (int): number of exponentials
        const (float/tuple of floats): const term
    """
    try:
        par = par + const
    except TypeError:
        par = par + (const,)
    data0 = ft.exp_model(par, x_axis, n=n_exp)
    fit = ft.fit_kinetics((x_axis,), (data0,),
                          n_exp=n_exp, const=const)
    assert fit[0].success == 1
    # getting exaclty right params is not ensured
    # assert sorted(tuple(np.round(fit[0].x))) == sorted(par)


@pytest.mark.parametrize('x_axis, n_exp, const', [
    ((np.linspace(-5, 199, 205), np.linspace(-4, 199, 204)), 1, (1, 1)),
    ((np.linspace(-5, 199, 205), np.linspace(-4, 199, 204)), 1, 1),
    ([np.linspace(-5, 199, 205)], 1, 1),
    ((np.linspace(-5, 199, 205), np.linspace(-4, 199, 204)), 1, 1),
    ((np.linspace(-5, 199, 205), np.linspace(-4, 199, 204)), 1, (1, 0)),
    ])
def test_single_kin03(x_axis, n_exp, const):
    """Fitting multiple datasets

    Args:
        x_axis (tuple of arrays): x axes
        n_exp (int): number of exponentials
        const (float/tuple of floats): constant added to the fit
    """
    data = []
    for axis in x_axis:
        data.append(ft.exp_model((1, 50, -1), axis, n=n_exp))
    fit = ft.fit_kinetics(x_axis, data,
                          n_exp=n_exp, const=const)
    for i, axis in enumerate(x_axis):
        assert fit[i].success == 1
        # assert sorted(np.round(fit[i].x)) == [-1, 1, 50]


@pytest.mark.parametrize('x_axis, n_exp, const, init_par', [
    ((np.linspace(-5, 199, 205), np.linspace(-4, 199, 204)),
     1, (1, 1), (1, 100)),
    ((np.linspace(-5, 199, 205), np.linspace(-4, 199, 204)),
     1, 1, (1, 100)),
    ([np.linspace(-5, 199, 205)], 1, 1, (1, 100)),
    ((np.linspace(-5, 199, 205), np.linspace(-4, 199, 204)),
     1, 1, (-1, 100)),
    ((np.linspace(-5, 199, 205), np.linspace(-4, 199, 204)),
     1, (1, 0), (-5, 100)),
    ])
def test_single_kin04(x_axis, n_exp, const, init_par):
    """test initial parameters

    Args:
        x_axis (tuple/list of arrays): x axes
        n_exp (int): number of exponentials
        const (float/ tuple of floats): constant to the fit
        init_par (tuple): amplitudes and taus
    """
    data = []
    for axis in x_axis:
        data.append(ft.exp_model((1, 50, -1), axis, n=n_exp))
    fit = ft.fit_kinetics(x_axis, data,
                          n_exp=n_exp, init_par=init_par, const=const)
    for i, axis in enumerate(x_axis):
        assert fit[i].success == 1
        # assert sorted(np.round(fit[i].x)) == [-1, 1, 50]


def test_global01():
    """test single kin, global, 1-exp"""
    x_axis = (np.linspace(-5, 199, 205), np.linspace(-4, 199, 204))
    data = []
    for axis in x_axis:
        data.append(ft.exp_model((1, 50), axis, n=1))
    fit = ft.fit_kinetics_global(x_axis, data, gl_par=(1, 1), n_exp=1)
    assert sorted(np.round(fit.x)) == [1, 50]


def test_global02():
    """test single kin fitting, global, 2-exp"""
    x_axis = (np.linspace(-5, 199, 205), np.linspace(-4, 199, 204))
    data = []
    for axis in x_axis:
        data.append(ft.exp_model((1, 50), axis, n=1))
    fit = ft.fit_kinetics_global(x_axis, data, gl_par=(0, 0), n_exp=1)
    assert fit.success == 1
    # assert sorted(np.round(fit.x)) == [1, 1, 50, 50]


@pytest.mark.parametrize('init_par', [
    (None),
    ((5, 5, 100)),
    (-1, -5, 20),
])
def test_global03(init_par):
    """global fitting, test initial parameter input

    Args:
        init_par (tuple): amplitudes and taus
    """
    x_axis = (np.linspace(-5, 199, 205), np.linspace(-4, 199, 204))
    data = []
    for axis in x_axis:
        data.append(ft.exp_model((1, 50), axis, n=1))
    fit = ft.fit_kinetics_global(x_axis, data, gl_par=(0, 1),
                                 n_exp=1,
                                 init_par=init_par)
    assert fit.success == 1
    # assert sorted(np.round(fit.x)) == [1, 1, 50]


# testing global with constant
@pytest.mark.parametrize('const, init, glob, solution', [
    (1, (-1, 5, 20), (0, 1, 1), [-1, 1, 1, 50]),
    ((1,), (-1, 5, 20), (0, 1, 1), [-1, 1, 1, 50]),
    ((5, 5), (-1, 5, 20), (0, 1, 0), [-1, -1, 1, 1, 50]),
    ((-1, -5), (-1, 5, 20), (0, 1, 0), [-1, -1, 1, 1, 50]),
    ([-1, -5], [-1, 5, 20], [0, 1, 0], [-1, -1, 1, 1, 50]),
])
def test_global04(const, init, glob, solution):
    """testing global fit with constant added

    Args:
        const (float/tuple of floats): constant term
        init (tuple/list): initial params
        glob (tuple/list): booleans wheter parameter global or not
        solution (list): expected assert
    """
    x_axis = (np.linspace(-5, 199, 205), np.linspace(-4, 199, 204))
    data = []
    for axis in x_axis:
        data.append(ft.exp_model((1, 50, -1), axis, n=1))
    fit = ft.fit_kinetics_global(x_axis, data, gl_par=glob,
                                 n_exp=1,
                                 init_par=init,
                                 const=const)
    assert fit.success == 1
    # assert sorted(np.round(fit.x)) == solution


# testing bounds
@pytest.mark.parametrize('const, init, glob, bounds, solution', [
    (1, (-1, 5, 20), (0, 1, 1), (-10, 100), [-1, 1, 1, 50]),
    ((1,), (-1, 5, 20), (0, 1, 1), (-10, 100), [-1, 1, 1, 50]),
    ((5, 5), (-1, 5, 20), (0, 1, 0), (-10, 100), [-1, -1, 1, 1, 50]),
    ((-1, -5), (-1, 5, 20), (0, 1, 0), (-10, 100), [-1, -1, 1, 1, 50]),
    ([-1, -5], [-1, 5, 20], [0, 1, 0], (-10, 100), [-1, -1, 1, 1, 50]),
    ([-1, -5], [-1, 5, 20], [0, 1, 0],
     ((-6, -6, 0, -6, -6), (6, 6, 1000, 6, 6)),
     [-1, -1, 1, 1, 50])])
def test_global05(const, init, glob, solution, bounds):
    """testing bound on global fit

    Args:
        const (float/tuple/list of floats): constant term
        init (tuple/list): initial amplitudes and taus
        glob (tuple/list): booleans wheter par global or not
        solution (list): assert
        bounds (tuple/two-tuple): following notation of optimize.least_squares
    """
    x_axis = (np.linspace(-5, 199, 205), np.linspace(-4, 199, 204))
    data = []
    for axis in x_axis:
        data.append(ft.exp_model((1, 50, -1), axis, n=1))
    fit = ft.fit_kinetics_global(x_axis, data, gl_par=glob,
                                 n_exp=1,
                                 init_par=init,
                                 const=const,
                                 bounds=bounds)
    assert fit.success == 1
    assert sorted(np.round(fit.x)) == solution


@pytest.mark.parametrize('model, par, expect', [
    ('one_state', None, (1, 1)),
    ('two_states', None, (2, 2)),
    ('two_states_transfer', None, (3, 2)),
])
def test_get_params(model, par, expect):
    """test of getting right number of parameters if not provided
    by user.

    Args:
        model (string): ODE model (has to listed in dict in fitting module)
        par (two-tuple/None): ode params(lifetimes) and amplitudea
        expect (tuple): assert
    """
    par_in = ft.get_params_ode(model, par)
    assert len(par_in[0]) == expect[0]  # number of taus
    assert len(par_in[1]) == expect[1]  # number of amps
    assert len(par_in) == 2


@pytest.mark.parametrize('model, expect', [
    ('one_state', (1, 1)),
    ('two_states', (2, 2)),
    ('two_states_transfer', (3, 2)),
])
def test_ode01(model, expect):
    """testing ODE, non-global

    Args:
        model (string): ODE model from fitting module
        expect (tuple): assert
    """
    x_axis = (np.linspace(-5, 199, 205), np.linspace(-4, 199, 204))
    data = []
    for axis in x_axis:
        data.append(ft.exp_model((1, 50), axis, n=1))
    par_in = ft.get_params_ode(model, None)
    fit = ft.fit_ode(x_axis, data, model,
                     p0_amp=par_in[1],
                     p0_ode=par_in[0])
    for i, axis in enumerate(x_axis):
        assert len(fit[i][0]) == expect[0]
        assert len(fit[i][1]) == expect[1]
        # fit_result = np.sum(ft.ode_solution(fit[i][0], model, (0, 1e6),
        #                                     fit[i][1], axis[axis > 0]).y,
        #                     axis=0)
        # res = np.sum((data[i][axis > 0]-fit_result)**2)
        # assert res < 1
