# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:01:57 2020

@author: David Palecek
"""
import os
import pathlib as p
import numpy as np

from .trs import Trs

print('running ta init')
__all__ = ['Ta']



class Ta(Trs):
    '''
    TA experimental class
    Child class of TRS (time-resolve spectroscopy)
    Handels Uberfast ps/fs and Fastlab TA files.
    '''
    def __init__(self, full_path, dir_save=None):
        super().__init__(dir_save)
        self.info = 'TA experimental data'
        self.path = p.PurePath(full_path)
        self.dir_path = self.path.parent

        self.load_data()
        self.save_path = self.create_save_path()


    def load_data(self):
        '''
        Calls loading function based on file suffix.
        Returns
        -------
        None.
        '''
        if self.path.suffix == '.hdf5':
            self.fastlab_import()
        elif self.path.suffix == '.wtf':
            self.uberfast_import()
        else:
            print('Unknown suffix')


    def fastlab_import(self):
        '''
        Importing .hdf5 files from Fastlab.
        '''
        pass


    def uberfast_import(self):
        '''
        Importing .wtf files from Uberfast fs and ps setups.
        '''
        data = np.loadtxt(self.path)
        wl_last = -1
        if max(data[:, 1]) > 0.1:
            print('ignoring first timeslice when importing ')
            ignore_first_spec = True
            data = np.delete(data, 1, axis=1)

        if not data[256:, 0].any():  # all zeros
            print('IR part empty, ps data')
            wl_last = 256
        self.wl = data[1:wl_last, 0]
        self.data = data[1:wl_last, 1:].transpose()*1000
        self._t = data[0, 1:]/1000

        self.t_unit = 'ps'
        self.wl_unit = 'nm'

        ### import sweeps ###
        try:
            sweep_files = [k for k in os.listdir(self.dir_path.joinpath('meas')) if 'meas' in k]
        except:
            print('No sweeps to load')
        else:
            self.n_sweeps = len(sweep_files)
            self.inc_sweeps = [1]*self.n_sweeps
            self.sweeps = (np.loadtxt(self.dir_path.joinpath('meas',
                                                           k))[1:, 1:].transpose()[:, :wl_last]*1000
                           for k in sweep_files)
            if ignore_first_spec:
                self.sweeps = [np.delete(sweep, 0, axis=0) for sweep in self.sweeps]
        