# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:02:16 2020

@author: David Palecek
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# TODO, load fitting module
from ..helpers import support as sup
from ..helpers.support import refresh_vals
from ..modules import plotting as plot

from .exp import Exp

__all__ = ['Trs']


class Trs(Exp):
    '''
    Time-resolved-spectroscopy class
    TODO: Fitting here
    '''
    def __init__(self, dir_save=None):
        super().__init__(dir_save)
        self.info = f'Class instance of {self.__class__}'
        self.path = None
        self._t = None
        self.t0 = 0
        self.t_unit = None
        self.wl = None
        self.wl_unit = None
        # kinetics, spectra
        self.kin = None
        self.kin_rng = None
        self.spe = None
        self.spe_rng = None
        self.tmax_id = None
        self.tmin_id = None
        self.wlmax_id = None
        self.wlmin_id = None
        # sweeps attributes
        self.inc_sweeps = None
        self.sweeps = None
        self.n_sweeps = None

    @property
    def t(self):
        '''
        return private attribute of time axis.
        Returns
        -------
        TYPE
            DESCRIPTION.
        '''
        return self._t

    @t.setter
    def t(self):
        '''
        Parameters
        ----------
        value : TYPE
            DESCRIPTION.
        Returns
        -------
        None.

        '''
        print('not allowed to change like this')

    def set_t0(self, val):
        '''setting t0 by given value
        If more datasets loaded, It changes
        t0 for idx's dataset
        author DP, last change 28/04/20'''
        if sup.is_num(val):
            self._t = self._t - val
            self.t0 += val
        else:
            raise ValueError('Value has to be numeric, not a string.')

    def rem_bg(self, val):
        ''' remove background, where the background is calculated as the
            time-averaged spectra of all points before 'tPos'
            author DP, last change 28/04/20'''
        if sup.is_num(val):
            idx = self._t < val
            if sum(idx) == 0:
                idx[0] = True
                print('Warning: all time points after tPos')
            bg = np.mean(self.data[idx, :], axis=0)
            self.data = self.data - bg*np.ones(self.data.shape)
        else:
            raise ValueError('Value has to be numeric, not a string.')

    def rem_region(self, wl_min, wl_max):
        '''set data to 0 for spectral region of 2D data
         - on the half-open interval [wl_min, wl_max)
        author DP, last change 28/04/20'''
        if sup.is_num(wl_min) and sup.is_num(wl_max):
            i_min, i_max = sup.get_idx(wl_min, wl_max, axis=self.wl)
            print(self.wl[i_min], self.wl[i_max], self.wl)
            self.data[:, i_min:i_max] = 0
        else:
            raise ValueError('Value has to be numeric, not a string.')

    @refresh_vals
    def cut_wl(self, wlmin, wlmax):
        '''select wl range between wlMin and wlMax
        - returns closed interval [wlmin, wlmax]
        author DP, last change 28/04/20'''
        if sup.is_num(wlmin) and sup.is_num(wlmax):
            imn, imx = sup.get_idx(wlmin, wlmax, axis=self.wl)
            self.wl = self.wl[imn:imx+1]
            self.data = self.data[:, imn:imx+1]
            try:
                self.sweeps = [self.sweeps[i][:, imn:imx+1]
                               for i in range(self.n_sweeps)]
            except:
                print('No sweeps')
            self.wlmax_id = imx + 1
            self.wlmin_id = imn
        else:
            raise ValueError('Value has to be numeric, not a string.')

    @refresh_vals
    def cut_t(self, tmin, tmax):
        '''removes timepoints between 'tMin' and 'tMax'
        author DP, last change 28/04/20'''
        if sup.is_num(tmin) and sup.is_num(tmax):
            imn, imx = sup.get_idx(tmin, tmax, axis=self._t)
            self._t = self._t[imn:imx+1]
            self.data = self.data[imn:imx+1, :]
            try:
                self.sweeps = [self.sweeps[i][imn:imx+1, :]
                               for i in range(self.n_sweeps)]
            except:
                print('No sweeps')
            self.tmax_id = imx + 1
            self.tmin_id = imn
        else:
            raise ValueError('Value has to be numeric, not a string.')

    def calc_spe(self, rng):
        ''' calculates time-averaged spectra, with timepoints defined as:
        rng = [t1min, t1max, t2min, t2max, ... txmin, txmax]
        output is stored in obj.spe, using the wavelength axis
        obj.wl
        author DP, last change 28/04/20'''
        self.spe = []
        self.spe_rng = rng
        zipped_tuple = tuple(zip(rng[::2], rng[1::2]))
        for i in zipped_tuple:
            beg, end = sup.get_idx(*i, axis=self.t)
            self.spe.append(np.mean(self.data[beg:end, :],
                                    axis=0))

    def calc_kin(self, rng):
        '''
        calculates time-averaged spectra, with timepoints defined as:
        rng = [wl1 min, wl1 max, wl2 min, wl2 max, ... wlx min, wlx max]
        the output is stored in obj.kin, using the time axis
        self.t
        author DP, last change 28/04/20
        TODO: recalculation should delete fitParams I think
        '''
        self.kin = []
        self.kin_rng = rng
        zipped_tuple = tuple(zip(rng[::2], rng[1::2]))

        for i in zipped_tuple:
            beg, end = sup.get_idx(*i, axis=self.wl)
            self.kin.append(np.mean(self.data[:, beg:end],
                                    axis=1))

    @refresh_vals
    def new_average(self, include):
        '''include sweeps from binary list:
        include = [0,1,1,0...0,1]
        author DP, last change 28/04/20
        TODO: this should recalculate data (kin/spe/fits)'''
        if len(include) != self.n_sweeps:
            raise ValueError(f'list has to be lenght = {self.n_sweeps}')

        self.inc_sweeps = include
        newav = sum(self.sweeps[i]
                    for i in range(len(include))
                    if include[i] == 1)
        self.data = newav / sum(include)

    def recalc(self):
        '''
        Reculculates all generated data if they exist
        TODO: what if I need to pass more attributes to the method.
        Returns
        -------
        None.
        '''
        print('running recalc.')
        lookup_attr = (('kin', 'calc_kin', 'kin_rng'),
                       ('spe', 'calc_spe', 'spe_rng'),
                       ('fit_params', 'fit_kin', 'par_in'))

        for attr, method, to_pass in lookup_attr:
            if attr in self.__dict__ and self.__dict__[attr] is not None:
                print(f'Calling {Trs.__dict__[method]} because data changed')
                Trs.__dict__[method](self, self.__dict__[to_pass])

    def comp_sweep_kin(self, rng):
        '''compare kinetics from different sweeps within rng of WL
        rng = [wl1 min, wl1 max, wl2 min, wl2 max, ... wlx min, wlx max]
        author DP, last change 28/04/20'''
        idx = sup.get_idx(*rng, axis=self.wl)
        _, ax1 = plt.subplots()
        for j in range(self.n_sweeps):
            cmap = cm.gist_heat((j) / self.n_sweeps, 1)
            for i in range(int(len(rng)/2)):
                kin = np.mean(self.sweeps[j][:, idx[2*i]:idx[2*i+1]],
                              axis=1)
            if self.inc_sweeps[j]:
                ax1.plot(self._t, kin,
                         label=j,
                         color=cmap)
            else:
                ax1.plot(self._t, kin,
                         '--', linewidth=1,
                         label=f'{j} not in av',
                         color=cmap)
        # TODO works only for single rng.
        kin_av = np.mean(self.data[:, idx[2*i]:idx[2*i+1]],
                         axis=1)
        plt.plot(self._t, kin_av, linewidth=3, label='av kin')
        plt.xscale('Log')
        plt.legend()
        plt.show()

    @refresh_vals
    def invert_sweeps(self, invert):
        '''invert sweeps from binary list
        invert = [0,1,1,0...0,1]
        author DP, last change 09/06/20
        TODO: This calls decorator twice because of self.new_average'''
        if len(invert) != self.n_sweeps:
            raise ValueError('list has to be lenght = {self.n_sweeps}')

        for i, j in enumerate(invert):
            if j:
                self.sweeps[i] = -self.sweeps[i]
        # recalculating Average data
        self.new_average(self.inc_sweeps)

    @plot.title_plot
    @plot.log_xscale
    @plot.normalize_plot
    def plot_kin(self, **kwargs):
        fig_kin = plt.figure()
        for i, line in enumerate(self.kin):
            plt.plot(self._t, line, label=i)
        self.figure = fig_kin
        return fig_kin

    @plot.title_plot
    @plot.log_xscale
    @plot.normalize_plot
    def plot_spe(self, **kwargs):
        fig_spe = plt.figure()
        for i, line in enumerate(self.spe):
            plt.plot(self.wl, line, label=i)
        self.figure = fig_spe
        return fig_spe

    # def reset_def_vals(self):
    #     '''
    #     sets all values to initial default state after loading.
    #     '''
    #     return

    # def exp_fit(self, kin, tlim):
    #     '''
    #     Fitting of kinetics

    #     Returns
    #     -------
    #     None.
    #     '''
    #     return
