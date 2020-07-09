"""Testing basic operations on TA datasets"""
import pathlib as p
import numpy as np

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
    """test for fastlab TA data import."""
    dat = Ta(path_fastlab_data)
    assert dat.data.shape == (126, 1024)
    assert len(dat.probe) == 22
    assert len(dat.reference) == 22
    assert len(dat.sweeps) == 22
    assert dat.sweeps[0].shape == (126, 1024)
    assert dat.n_sweeps == 22
    assert dat.wl_unit == 'nm'
    assert dat.wl_high == 740
    assert dat.px_low == 258
    assert dat.t_unit == 'ns'
    assert dat.n_shots == 1000
    assert dat.inc_sweeps == [1]*22


def test_uberfast_import():
    """test for uberfast TA data import"""
    dat = Ta(path_uberfast_data)
    assert dat.data.shape == (178, 510)
    assert dat.t_unit == 'ps'
    assert dat.wl_unit == 'nm'
    assert dat._t.shape == (178,)
    assert dat.wl.shape == (510,)
    assert dat.inc_sweeps == [1]*20
    assert dat.n_sweeps == 20
    assert len(dat.sweeps) == 20
    assert dat.sweeps[0].shape == (178, 510)
    assert dat.kin is None


def test_t0():
    """setting time zero"""
    obj = Ta()
    obj._t = np.linspace(0, 10, 11)
    obj.t0 = 0
    obj.set_t0(1)
    assert (obj._t - np.linspace(-1, 9, 11)).all() == 0
    assert obj.t0 == 1


def test_t02():
    """setting time zero in sequence"""
    obj = Ta()
    obj._t = np.linspace(-1, 9, 11)
    obj.t0 = 1
    obj.set_t0(-5)
    assert (obj._t - np.linspace(4, 14, 11)).all() == 0
    assert obj.t0 == -4


def test_rem_bg():
    """remove background"""
    obj = Ta()
    obj.data = np.ones((10, 10)).reshape(10, 10)
    obj._t = np.linspace(0, 9, 10)
    obj.rem_bg(5)
    assert np.sum(obj.data, axis=(0, 1)) == 0


def test_rem_region():
    """remove spectral region"""
    obj = Ta()
    obj.data = np.ones((10, 10)).reshape(10, 10)
    obj.wl = np.linspace(0, 9, 10)
    obj.rem_region(2, 7)
    assert np.sum(obj.data, axis=(0, 1)) == 50


def test_cut_wl():
    """cut wavelength axis outside of specified region"""
    obj = Ta()
    obj.data = np.ones((10, 10))
    obj.wl = np.linspace(0, 9, 10)
    obj.cut_wl(2, 7)
    assert list(obj.wl) == [2, 3, 4, 5, 6, 7]


def test_cut_wl_sweeps():
    """cut wavelength axis outside of region for sweeps"""
    pass


def test_cut_t():
    """cut time axis outside of the specified region"""
    obj = Ta()
    obj.data = np.ones((10, 10))
    obj._t = np.linspace(0, 9, 10)
    Ta.cut_t(obj, 2, 7)
    assert list(obj._t) == [2, 3, 4, 5, 6, 7]


def test_calc_spe():
    """ calculate spectra in ranges."""
    obj = Ta()
    obj.data = np.ones((5, 5))
    obj._t = np.linspace(0, 4, 5)
    obj.calc_spe([2, 3.5])
    assert list(obj.spe[0]) == [1]*len(obj._t)
    assert obj.spe_rng == [2, 3.5]


def test_calc_kin():
    """ calculate kinetics"""
    obj = Ta()
    obj.data = np.ones((5, 5))
    obj.wl = np.linspace(0, 4, 5)
    obj.calc_kin([2, 3.5])
    assert list(obj.kin[0]) == [1]*len(obj.wl)
    assert obj.kin_rng == [2, 3.5]


def test_recalc():
    """recalculate data if data changed"""
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
    """Calculate new average of data from specified sweeps"""
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
    """ invert selected sweeps"""
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
