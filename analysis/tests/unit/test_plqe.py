import pathlib as p
import numpy as np
import os
# import cbr_analysis_v0.analysis.data as data
# from ...data import data
from analysis import data
import analysis.experiments.plqe as an



path_3mes_data = p.PurePath(data.__file__
                                ).parent.joinpath(
    '12',
    '12_DPA')

path_3mes_data = p.PurePath(data.__file__
                                ).parent.joinpath(
    '1.3',
    'TIPS_TACA')


def test_Plqe_import():
    #test loading 3 meas
    FileType = '.asc' # file ending
    Delimiter = ','
    Header = 0
    Footer = 35
    data = an.Plqe(path_3mes_data,setup='CPT',filetype=FileType,delimiter = Delimiter,header=Header,footer=Footer)
    assert data.data.shape == (1018, 3)
    assert data.detector == 'Si'
    assert data.accum == ([20,20,20])
    assert data.exposure == ([0.5, 0.5, 0.5])
    assert data.center_wl == (600,)
    data.calibrate()
    data.calc_plqe(exc_wl=[390,435],pl_wl=[450,740],xlim=(350,750), yscale='log')
    assert round(data.plqe*100,2) == 48.5

    #test loading 5 meas
    FileType = '.asc' # file ending
    Delimiter = '\t'
    Header = 0
    Footer = 35

    data = an.Plqe(path_5mes_data,setup='CPT',filetype=FileType,delimiter = Delimiter,header=Header,footer=Footer,combine_wl=860)
    assert data.data.shape == (776, 3)
    assert data.detector == 'InGaAs'
    assert data.accum == ([5,5,5,5,5])
    assert data.exposure == ([5.0, 5.0, 5.0, 5.0, 5.0])
    data.calibrate(center_wl=[800,1200])
    data.calc_plqe(exc_wl=[640,680],pl_wl=[880,1350],xlim=(600,1500), yscale='log')
    assert round(data.plqe*100,2) == 34.02

