import pathlib as p
import numpy as np
# import cbr_analysis_v0.analysis.data as data
# from ...data import data
from analysis import data
from analysis.experiments.ta import Ta



class Test:
    pass


path = p.PurePath(data.__file__).parent.joinpath(
        '1_PbS_TETCA_high_tol_200um_40uW_532nm.hdf5')
print(path)
data = Ta(path)
obj = Test()


def test_ta_fastlab_load():
    assert data.data.shape == (126, 1024)
    assert len(data.probe) == 22
    assert len(data.reference) == 22
    assert data.n_sweeps == 22
    assert data.wl_units == 'nm'
    assert data.wl_high == 740
    assert data.px_low == 258
    assert data.t_units == 'ns'
    assert data.n_shots == 1000


def test_t0():
    obj._t = np.linspace(0, 10, 11)
    obj.t0 = 0
    Ta.set_t0(obj, 1)
    assert (obj._t - np.linspace(-1, 9, 11)).all() == 0
    assert obj.t0 == 1


def test_t02():
    obj._t = np.linspace(-1, 9, 11)
    obj.t0 = 1
    Ta.set_t0(obj, -5)
    assert (obj._t - np.linspace(4, 14, 11)).all() == 0
    assert obj.t0 == -4


def test_rem_bg():
    obj.data = np.ones((10, 10)).reshape(10, 10)
    obj._t = np.linspace(0, 9, 10)
    Ta.rem_bg(obj, 5)
    assert np.sum(obj.data, axis=(0, 1)) == 0


def test_rem_region():
    obj.data = np.ones((10, 10)).reshape(10, 10)
    obj.wl = np.linspace(0, 9, 10)
    Ta.rem_region(obj, 2, 7)
    assert np.sum(obj.data, axis=(0, 1)) == 50


# def test_cut_wl():
#     obj.wl = np.linspace(0, 9, 10)
#     Ta.cut_wl(obj, 2, 7)
#     assert list(obj.wl) == [2, 3, 4, 5, 6, 7]


# def test_cut_wl_sweeps():
#     pass


# def test_cut_t():
#     obj._t = np.linspace(0, 9, 10)
#     Ta.cut_t(obj, 2, 7)
#     assert list(obj._t) == [2, 3, 4, 5, 6, 7]
