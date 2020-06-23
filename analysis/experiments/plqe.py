# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:20:57 2020

@author: David Palecek
"""
import os
import re
from .static import Static
import matplotlib.pyplot as plt
import pathlib as p
import numpy as np
import pandas as pd
from ..modules import plotting as plot

from .. import data
__all__ = ['Plqe']


class Plqe(Static):
    def __init__(self, full_path, setup, dir_save=None, **kwargs):
        super().__init__(dir_save)
        self.info = 'PLQE experimental data'
        self.data_raw = None
        self.wl = None
        self.wl_raw = None
        self.n_meas = None
        self.exposure = kwargs.get('exposure', None)
        self.accum = kwargs.get('accumulations', None)
        self.power = None
        self.center_wl = kwargs.get('center_wl', (600,))
        self.laser_wl = None
        self.combine_wl = kwargs.get('combine_wl', None)
        # PLQE
        self.exc_wl = None
        self.pl_wl = None
        self.plqe = None
        # files
        self.path = p.PurePath(full_path)
        self.dir_path = self.path.parent
        self.file_type = kwargs.get('filetype', '.asc')
        self.delimiter = kwargs.get('delimiter', ',')
        self.header = kwargs.get('header', 1)
        self.footer = kwargs.get('footer', 0)
        self.data_list = None
        # calibration
        self.setup = setup
        # operations
        self.is_combined = False
        self.is_cal = False
        self.is_filter = False
        self.is_bg_sub = False

        self.load_data()
        print(self.info)

    def reset_plqe(self):
        self.info = None
        self.calibration = None
        self.is_cal = False
        self.filter = None
        self.is_filter = False
        self.plqe_value = None
        self._calcAbs = None
        self.exc_wl = None
        self.pl_wl = None
        self.plqe = None
        self.is_combined = False
        self.is_bg_sub = False

        self.data = self.data_raw.copy()
        self.wl = self.wl_raw.copy()

    def load_data(self):
        '''
        load three or five dataset of PLQE data
        - Requires files to be named after convention ON/OFF/NO/BLANK
        - Determine 3 or 5 files to read
        author VG, last change: 14/5/20
        Returns
        -------
        None.
        '''
        print('Loading some data here and here')
        self.data_list = [k for k in os.listdir(self.dir_path)
                          if ('ON' in k or 'OFF' in k or 'NO' in k)
                          and self.file_type in k]
        self.n_meas = len(self.data_list)
        if self.n_meas != 3 and self.n_meas != 5:
            raise RuntimeWarning(f'Only works with 3 or 5 files, \
                                 but {self.n_meas} given.')

        # for 3 measurement:  order files on/off/blank or
        # for 5 measurements:  order on_las/on_pl/off_las/off_pl/blank
        key_3files = ['ON', 'OFF', 'NO']
        key_5files = ['ONLAS', 'OFFLAS', 'ONPL', 'OFFPL', 'NO']
        self.key = key_3files if self.n_meas == 3 else key_5files
        # indices of files as they should be ordered in key
        idx_sorted = [idx for name in self.data_list
                      for idx, value in enumerate(self.key)
                      if value in name]

        # sorting according to the selected key
        _, self.data_list = zip(*sorted(zip(idx_sorted, self.data_list)))

        # identify detector
        if 'ingaas' in self.data_list[0].lower():
            self.detector = 'InGaAs'
        else:
            self.detector = 'Si'

        power, laser, exposure,  accum, cw = [], [], [], [], []
        data, wl = [], []
        # extracts pump power from file names
        for i, name in enumerate(self.data_list):
            # extract pump wl
            str_pump = re.search(r'_\d+nm_', name.lower())
            if str_pump is not None:
                laser.append(float(str_pump.group()[1:-3]))

            # extract power
            str_power = re.search(r'_\d+p?\d*?[mu]w_', name.lower())
            if str_power is not None:
                power.append(float(str_power.group().replace('p', '.')[1:-3]))

            # extract center WL (for calibration)
            str_cw = re.search(r'_(cw)?\d+(cw)?_', name.lower())
            if str_cw is not None:
                cw.append(int(re.search(r'\d+', str_cw.group()).group()))

            # accumulations and exposure
            str_to_parse = re.search(r'\d+x\d+[p.s0-9]+', name.lower())
            if str_to_parse is None:
                print('Wrong format of accumulations/exposure string')
                expos = 1
                acc = 1
            else:
                expos = float(str_to_parse.group().split('x')[1][:-1].replace('p', '.'))
                acc = int(str_to_parse.group().split('x')[0])
                exposure.append(expos)
                accum.append(acc)

            # load data
            print(f'LOADING: {name} as {self.key[i]}')
            data_temp = np.genfromtxt(self.dir_path.joinpath(name),
                                      delimiter=self.delimiter,
                                      skip_header=self.header,
                                      skip_footer=self.footer)
            data.append(data_temp[:, 1] / (expos * acc))
            wl.append(data_temp[:, 0])

        self.power = tuple(power)  # po[0]
        # only save the unique center wavelengths, discard duplicates
        self.laser_wl = tuple({*laser})
        if cw != []:
            self.center_wl = tuple(cw)
        if self.exposure is None:
            self.exposure = exposure
            self.accum = accum
        self.data = np.vstack(data).T
        self.data_raw = self.data.copy()



        if self.n_meas == 5:
            self.wl = np.vstack((wl[0],
                                 wl[2])).T
            self.wl_raw = self.wl.copy()
            self.combine_plqe(combineWL=self.combine_wl)
        else:
            self.wl = wl[0]   # wl for all 3 measurements should be the same
            self.wl_raw = self.wl.copy()

    def calibrate(self, **kwargs):
        '''Author VG last edited 14/5/20'''
        '''calibrates PLQE data with input calibration file(s)' '''
        if self.is_cal is True:
            print('Calibration Already Performed, ignoring calibration request')
            return
        # calibration file path
        self.calib_file = p.PurePath(data.__file__).parent.joinpath(
            'Combined_Calibration_Uni_1908_CPT_190707_Fluorimeter.csv')
        self.center_wl = kwargs.get('center_wl', self.center_wl)

        if self.detector is None:
            self.detector = kwargs.get('detector', None)
        else:
            print(f'detector already defined: {self.detector}')

        files = []
        print(f'center WL is {self.center_wl}')
        for val in self.center_wl:
            files.append(str(self.setup)+'_'+str(self.detector)+'_c'+str(val)+'_wl')
            files.append(str(self.setup)+'_'+str(self.detector)+'_c'+str(val))

        CalibrationAll = np.array(pd.read_csv(self.calib_file,
                                              delimiter=';',
                                              usecols=files))
        Cal0 = CalibrationAll[:, :2]

        if len(CalibrationAll[0, :]) > 2:
            Cal1 = CalibrationAll[:, 2:4]
            self.calibration = np.vstack(((Cal0[Cal0[:, 0] < self.combine_wl, :]),
                                          (Cal1[Cal1[:, 0] >= self.combine_wl, :])))
        else:  # otherwise assume only one file and store it
            self.calibration = Cal0

        if self.is_filter is True:
            print('Calibrating current data with applied Filter:')
        else:
            print('Calibrating current RawData:')

        # Interpolate to self.wl and calculate self._datCal
        CalibIp = np.interp(self.wl,
                            self.calibration[:, 0],
                            self.calibration[:, 1])
        self.data = (self.data/CalibIp[:, None])  # Calibration
        self.is_cal = True

    def combine_plqe(self,combineWL=None):
        if type(combineWL) != int:
            self.combine_wl = min(self.wl[:, 1])+50
        else:
            self._combineWL = combineWL
        #  Combine the LAS and PL measurements. Create an intermediate array
        #  (inter) with all 5 columns and later remove the PL columns
        inter = self.data_raw[self.wl_raw[:, 0] <
                              self.combine_wl, :]  # pick LAS measurement until combineWL
        inter2 = self.data_raw[self.wl_raw[:, 1] >=
                               self.combine_wl, :]  # pick PL measurement after combineWL
        interLAS = inter[:, (0, 1, 4)]  # remove columns with with PL experiments
        interPL = inter2[:, (2, 3, 4)]  # remove columns with LAS experiments

        # combine WL arrays
        self.wl = np.concatenate(((self.wl[self.wl[:, 0] <
                                           self.combine_wl, 0]),
                                  (self.wl[self.wl[:, 1] >=
                                   self.combine_wl, 1])))

        com = np.vstack((interLAS, interPL))
        # Set the NO measurement PL region to 0 for a 5 measurement experiment.
        # Can add later a 6 experiment if a NOPL measurement is also performed
        com[self.wl >= self.combine_wl, -1] = 0
        self.data = com
        self.is_combined = True
        print('Measurements combined')

        # raw data update
        self.data_raw = self.data.copy()
        self.wl_raw = self.wl.copy()

    def rem_bg(self, bg_wl):
        '''averages the value at WL bgWL+-10nm and subtracts
        this from the data to remove a background drift.
        author VG last edited 01/05/20'''
        # single number for each spectrum (ON/OFF/NO)
        background = np.mean(self.data_raw[(self.wl_raw > bg_wl-10) &
                                           (self.wl_raw < bg_wl+10), :], axis=0)
        self.data = self.data_raw - background
        self.is_bg_sub = True
        print('Background Subtracted')
        if self.is_cal is True:
            print('recalibrating data...')
            CalibIp = np.interp(self.wl,
                                self.calibration[:, 0],
                                self.calibration[:, 1])
            self.data = (self.data / CalibIp[:, None])

    def calc_plqe(self, exc_wl=None, pl_wl=None, **kwargs):
        '''Calculates PLQE for the current data integrating
        over Ex and PL wls as defined by ExWL and PLWL
        author VG last edited 18/05/20'''
        self.exc_wl = exc_wl
        self.pl_wl = pl_wl

        if (self.is_cal is False):
            print('Data Not Calibrated, calculating PLQE from Raw data,\n\
                  PLQE value not accurate - perform calibration!')

        fig = self.plot_plqe(**kwargs)

        fig.axes[0].axvline(self.exc_wl[0], color='red')
        fig.axes[0].axvline(self.exc_wl[1], color='red')
        fig.axes[0].axvline(self.pl_wl[0], color='black')
        fig.axes[0].axvline(self.pl_wl[1], color='black')

        # Absorption = 1-sum(DataON/DataOFF)
        absorption = 1 - (np.sum(self.data[(self.wl > exc_wl[0]) &
                                           (self.wl < exc_wl[1]), 0]) /
                          np.sum(self.data[(self.wl > exc_wl[0]) &
                                           (self.wl < exc_wl[1]), 1]))

        pump_no = np.sum(self.data[(self.wl > exc_wl[0]) &
                                   (self.wl < exc_wl[1]), -1])
        print(f'The absorption between {exc_wl} nm is: {absorption}')

        # PLQE = (PLtotalOn-((1-Absorption)*PLtotalOff))/(Absorption*PumpBlank)
        # Background counts for NO sample
        bgd = np.sum(self.data[(self.wl > pl_wl[0]) &
                               (self.wl < pl_wl[1]), -1])
        # Integrate over PL ON
        pl_on = np.sum(self.data[(self.wl > pl_wl[0]) &
                                 (self.wl < pl_wl[1]), 0]) - bgd
        # Integrate over PL OFF
        pl_off = np.sum(self.data[(self.wl > pl_wl[0]) &
                                  (self.wl < pl_wl[1]), 1]) - bgd

        plqe = (pl_on - ((1-absorption) * pl_off)) / (absorption * pump_no)
        print(f'The PLQE between {pl_wl} nm is: {round(plqe*100,2)} %')
        self.plqe = plqe

    @plot.xlims
    @plot.ylims
    @plot.log_yscale
    def plot_plqe(self, **kwargs):
        if self.is_cal is False:
            print('No calibrated performed, plotting raw data.')
            fig = self.plot_plqe_raw(**kwargs)
        else:
            print(f'Calibration: {self.is_cal}\n',
                  f'Bgd subtracted: {self.is_bg_sub}\n',
                  f'Data combined: {self.is_combined}\n',
                  f'Filter: {self.is_filter}')
            fig = plt.figure()
            for i in range(self.data.shape[1]):
                plt.plot(self.wl, self.data[:, i], label=self.key[i])
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Photons')
            plt.title('Calibrated Data')
        self.figure = fig
        return fig

    @plot.xlims
    @plot.ylims
    @plot.log_yscale
    def plot_plqe_raw(self, **kwargs):
        fig = plt.figure()
        for i in range(self.data_raw.shape[1]):
            plt.plot(self.wl_raw, self.data_raw[:, i], label=self.key[i])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Raw counts')
        plt.title('Raw Data')
        self.figure = fig
        return fig
