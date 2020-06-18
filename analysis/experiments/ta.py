# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:01:57 2020

@author: David Palecek
"""
import os
import h5py
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
    def __init__(self, full_path=None, dir_save=None):
        super().__init__(dir_save)
        self.info = 'TA experimental data'
        self.probe = []
        self.reference = []
        # case of providing path to data
        if full_path is not None:
            self.path = p.PurePath(full_path)
            self.dir_path = self.path.parent
            self.save_path = self.create_save_path()
            self.load_data()
        else:  # empty TA object
            self.path = None
            self.dir_path = None
            self.save_path = None
        print('correct version of analysis.')

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
        print('loading fastlab TA data')
        # os.chdir(p.PurePath(self.dir_path))
        f = h5py.File(p.PurePath(self.path), 'r')
        avg = np.array(f['Average'])
        self.data, self.data_raw = avg[1:, 1:]*1000, avg[1:, 1:]*1000
        self.wl = avg[0, 1:]  # array loads transposed compared to Matlab
        self.wl_raw = self.wl
        self._t = avg[1:, 0]
        self.t_raw = self._t

        metaD = f['Average'].attrs['time zero']
        if metaD:  # check for empty list
            # Set wavelength units / not stored in HDF5 file
            self.wl_unit = 'nm'
            delay = f['/Average'].attrs['delay type']
            self.delay_type = str(delay)

            if 'Long' in str(delay):
                self.t_unit = 'ns'
            elif 'UltraShort' in str(delay):
                self.t_unit = 'fs'
            elif 'Short' in str(delay):
                self.t_unit = 'ps'
            else:
                print('No delayType imported')
                print(str(delay))

            self.n_sweeps = len(f['Sweeps'].keys())
            self.inc_sweeps = [1]*self.n_sweeps
            self.n_shots = float(f['Average'].attrs['num shots'])
            self.px_low = float(f['Average'].attrs['calib pixel low'])
            self.wl_low = float(f['Average'].attrs['calib wave low'])
            self.px_high = float(f['Average'].attrs['calib pixel high'])
            self.wl_high = float(f['Average'].attrs['calib wave high'])

            # loading probe/reference spectra
            for i in list(f['Spectra']):
                if 'Error' in i:
                    self.error.append(np.array(f['Spectra'][i]))
                elif 'Probe' in i:
                    self.probe.append(np.array(f['Spectra'][i]))
                elif 'Reference' in i:
                    self.reference.append(np.array(f['Spectra'][i]))
                else:
                    print('Unknown specra to load..')

            self.ref_spe_init = self.reference[0]
            self.ref_spe_end = self.reference[-1]
            self.probe_spe_init = self.probe[0]
            self.probe_spe_end = self.probe[-1]

            self.sweeps = []
            for i in list(f['Sweeps']):
                self.sweeps.append(np.array(f['Sweeps'][i][1:, 1:] * 1000))
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

        #  import sweeps
        try:
            sweep_files = [k
                           for k in os.listdir(self.dir_path.joinpath('meas'))
                           if 'meas' in k]
        except NameError:
            print('No sweeps to load')
        else:
            self.n_sweeps = len(sweep_files)
            self.inc_sweeps = [1]*self.n_sweeps
            self.sweeps = (np.loadtxt(
                self.dir_path.joinpath('meas', k)
                                      )[1:, 1:].transpose()[:, :wl_last]*1000
                           for k in sweep_files)
            if ignore_first_spec:
                self.sweeps = [np.delete(sweep, 0, axis=0)
                               for sweep in self.sweeps]
