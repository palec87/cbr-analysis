import pathlib as p
import numpy as np
# import cbr_analysis_v0.analysis.data as data
# from ...data import data
from analysis import data
from analysis.experiments.ta import Ta


path_fastlab_data = p.PurePath(data.__file__
                               ).parent.joinpath(
    '1_PbS_TETCA_high_tol_200um_40uW_532nm.hdf5')
path_uberfast_data = p.PurePath(data.__file__
                                ).parent.joinpath(
    'UF_data',
    'sol-f889_ma_540nm_5uW_MA_POL_Pump_H_POL_Probe.wtf')


def test_fastlab_import():
    data = Ta(path_fastlab_data)
    assert data.data.shape == (126, 1024)
    assert len(data.probe) == 22
    assert len(data.reference) == 22
    assert len(data.sweeps) == 22
    assert data.sweeps[0].shape == (126, 1024)
    assert data.n_sweeps == 22
    assert data.wl_unit == 'nm'
    assert data.wl_high == 740
    assert data.px_low == 258
    assert data.t_unit == 'ns'
    assert data.n_shots == 1000
    assert data.inc_sweeps == [1]*22


def test_uberfast_import():
    data = Ta(path_uberfast_data)
    assert data.data.shape == (178, 510)
    assert data.t_unit == 'ps'
    assert data.wl_unit == 'nm'
    assert data._t.shape == (178,)
    assert data.wl.shape == (510,)
    assert data.inc_sweeps == [1]*20
    assert data.n_sweeps == 20
    assert len(data.sweeps) == 20
    assert data.sweeps[0].shape == (178, 510)
    assert data.kin is None


def test_t0():
    obj = Ta()
    obj._t = np.linspace(0, 10, 11)
    obj.t0 = 0
    obj.set_t0(1)
    assert (obj._t - np.linspace(-1, 9, 11)).all() == 0
    assert obj.t0 == 1


def test_t02():
    obj = Ta()
    obj._t = np.linspace(-1, 9, 11)
    obj.t0 = 1
    obj.set_t0(-5)
    assert (obj._t - np.linspace(4, 14, 11)).all() == 0
    assert obj.t0 == -4


def test_rem_bg():
    obj = Ta()
    obj.data = np.ones((10, 10)).reshape(10, 10)
    obj._t = np.linspace(0, 9, 10)
    obj.rem_bg(5)
    assert np.sum(obj.data, axis=(0, 1)) == 0


def test_rem_region():
    obj = Ta()
    obj.data = np.ones((10, 10)).reshape(10, 10)
    obj.wl = np.linspace(0, 9, 10)
    obj.rem_region(2, 7)
    assert np.sum(obj.data, axis=(0, 1)) == 50


def test_cut_wl():
    obj = Ta()
    obj.data = np.ones((10, 10))
    obj.wl = np.linspace(0, 9, 10)
    obj.cut_wl(2, 7)
    assert list(obj.wl) == [2, 3, 4, 5, 6, 7]


def test_cut_wl_sweeps():
    pass


def test_cut_t():
    obj = Ta()
    obj.data = np.ones((10, 10))
    obj._t = np.linspace(0, 9, 10)
    Ta.cut_t(obj, 2, 7)
    assert list(obj._t) == [2, 3, 4, 5, 6, 7]


def test_calc_spe():
    obj = Ta()
    obj.data = np.ones((5, 5))
    obj._t = np.linspace(0, 4, 5)
    obj.calc_spe([2, 3.5])
    assert list(obj.spe[0]) == [1]*len(obj._t)
    assert obj.spe_rng == [2, 3.5]


def test_calc_kin():
    obj = Ta()
    obj.data = np.ones((5, 5))
    obj.wl = np.linspace(0, 4, 5)
    obj.calc_kin([2, 3.5])
    assert list(obj.kin[0]) == [1]*len(obj.wl)
    assert obj.kin_rng == [2, 3.5]


def test_recalc():
    obj = Ta()
    obj.data = np.ones((5, 5))
    obj.wl = np.linspace(0, 4, 5)
    obj._t = np.linspace(0, 4, 5)
    obj.kin = np.zeros(5)
    obj.kin_rng = [2, 3.5]
    obj.recalc()
    assert obj.spe is None
    assert list(obj.kin[0]) == [1]*len(obj.wl)
    assert obj.spe_rng is None


def test_new_average():
    obj = Ta()
    obj.data = np.ones((5, 5))
    obj.sweeps = [np.ones((5, 5))*k
                  for k in range(3)]
    obj.n_sweeps = len(obj.sweeps)
    obj.new_average([1, 1, 1])
    assert obj.inc_sweeps == [1, 1, 1]
    assert list(obj.data.ravel()) == [1]*25
    obj.new_average([1, 1, 0])
    assert obj.inc_sweeps == [1, 1, 0]
    assert list(obj.data.ravel()) == [0.5]*25


def test_invert_sweeps():
    obj = Ta()
    obj.sweeps = [np.ones((5, 5))*k
                  for k in range(3)]
    obj.n_sweeps = len(obj.sweeps)
    obj.new_average([1, 1, 1])
    obj.invert_sweeps([1, 1, 0])
    assert list(obj.data.ravel()) == [1/3]*25
    obj.inc_sweeps = [1, 0, 1]
    obj.invert_sweeps([1, 1, 0])
    assert list(obj.data.ravel()) == [1]*25
    obj.invert_sweeps([1, 1, 1])
    assert list(obj.data.ravel()) == [-1]*25
