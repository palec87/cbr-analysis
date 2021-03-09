"""Testing basic operations on iCCD datasets"""
import pathlib as p

from analysis import data
from analysis.experiments.iccd import Iccd

path_fastlab_iccd_data = p.PurePath(
    data.__file__
    ).parent.joinpath(
        'iCCD_data',
        'f887_sol_0p2s_30uw_100accu_sig.asc',
        )


def test_fastlab_iccd_import():
    """test for fastlab iCCD data import."""
    data = Iccd(path_fastlab_iccd_data)
    assert data.data.shape == (4, 2047)
    assert len(data.t) == 4
    assert len(data.wl) == 2047
    assert data.step == float(10)
    assert data.exposure == float(0.2)
    assert data.n_accum == float(100)
    assert data.wl_unit == 'nm'
    assert data.t_unit == 'ns'


def test_load_bg():
    """test load_bg"""
    obj = Iccd()
    obj.path = path_fastlab_iccd_data
    obj.dir_path = obj.path.parent
    obj.load_bg()
    assert obj.bg.shape == (2047,)


def test_read_footer():
    """ test read_footer"""
    obj = Iccd()
    obj.path = path_fastlab_iccd_data
    obj.read_footer()
    ls_keys = ['Date and Time',
               'Model',
               'Gain level',
               'Input Side Slit Width (um)',
               ]
    ls_values = ['Thu Jan 23 11:55:35 2020',
                 'DH740_18mm',
                 '0',
                 '100',
                 ]
    for i in range(len(ls_keys)):
        assert obj.info[ls_keys[i]] == ls_values[i]
