# -*- coding: utf-8 -*-
from .trs import Trs
import pathlib as p
import numpy as np

__all__ = ['Iccd']


class Iccd(Trs):
    """iCCD class, child of Trs

    Args:
        Trs (class): parent class
    """
    def __init__(self, full_path=None, dir_save=None):
        super().__init__(dir_save)
        self.info = 'iCCD experimental data'
        # case of providing path to data
        if full_path is not None:
            self.path = p.PurePath(full_path)
            self.dir_path = self.path.parent
            self.save_path = self.create_save_path()
            self.load_data()
        else:  # empty iCCD object
            self.path = None
            self.dir_path = None
            self.save_path = None

    def load_data(self):
        """load iccd data from ascci exported files for signal (_sig.asc) and
         the background (_bgd.asc) files. It reads the footer and tries to
         extract the experimnetal information.
        """
        print(f'loading data: {self.path}')
        self.read_footer()
        if isinstance(self.info, dict):
            self.info_to_attr()
        d = np.genfromtxt(self.path, skip_footer=35)
        try:
            t = np.linspace(0, self.step*(d.shape[1]-2), d.shape[1]-1)
        except AttributeError:
            step = float(input('What is a time step in ns:'))
            t = np.linspace(0, self.step*(d.shape[1]-2), d.shape[1]-1)
            self.step = step
        self.wl = d[:, 0]
        self._t = t
        self.tcorr = t
        self.data = d[:, 1:].transpose()
        self.wl_unit = 'nm'
        self.t_unit = 'ns'
        # Try to load background
        try:
            self.load_bg()
            self.info_to_attr()
        except:
            pass

    def read_footer(self):
        """Extract the experimental information from the ascii files.
        converts the footer into a dictionary.
        """
        dict_info = {}
        with open(self.path, "r") as f:
            line = f.readline()
            while not line.startswith('Date and Time'):
                line = f.readline()
            while line:
                split = line.split(':', 1)
                try:
                    dict_info[split[0]] = split[1].strip('\n').lstrip()
                except IndexError:
                    pass
                line = f.readline()
        if dict_info != {}:
            self.info = dict_info

    def info_to_attr(self):
        """Essential items from the self.info dictionary
        converted into the attributes. More can be added if
        needed in the future.
        """
        self.step = float(self.info['Gate Delay Step (nsecs)'])
        self.exposure = float(self.info['Exposure Time (secs)'])
        self.n_accum = float(self.info['Number of Accumulations'])

    def load_bg(self):
        """loading bgd file into the self.bg attribute
        """
        name = self.path.stem.replace('sig', 'bgd.asc')
        self.path_bg = self.dir_path.joinpath(name)

        bg = np.genfromtxt(self.path_bg, skip_footer=35)
        self.bg = bg[:, 1]

    def rem_bg(self):
        """correct the data with self.bg attribute if it
        exists.
        """
        try:
            self.data = self.data - self.bg*np.ones(self.data.shape)
        except AttributeError('No bg loaded, use load_bg first.'):
            pass
