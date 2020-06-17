import pathlib as p
from analysis import data
from analysis.experiments.ta import Ta


path = p.PurePath(data.__file__).parent.joinpath(
        '1_PbS_TETCA_high_tol_200um_40uW_532nm.hdf5')

def test_ta_fastlab_load():
    data = Ta(path)
    print(data.n_sweeps)
    assert data.n_sweeps == 22