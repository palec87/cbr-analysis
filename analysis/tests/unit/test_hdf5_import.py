import pathlib as p
import numpy as np
from analysis import data
from analysis.experiments.ta import Ta


path = p.PurePath(data.__file__).parent.joinpath(
        '1_PbS_TETCA_high_tol_200um_40uW_532nm.hdf5')
data = Ta(path)
obj = Test()
obj._t = np.linspace(0, 10, 10)

class Test():
    pass


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
    Ta.set_t0(obj,1)
    assert obj._t == np.linspace(-1, 9, 10)
    assert obj.t0 == 1
