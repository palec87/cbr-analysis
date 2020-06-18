# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:26:06 2020

@author: David Palecek
"""
import pickle
import os
from ..helpers import support as sup

__all__ = ['Exp']


class Exp():
    '''
    Class of all the experiments which takes care of ini/save...
    '''
    def __init__(self, dir_save=None):
        self.info = f'Class instance of {self.__class__}'
        self.data = None
        self.error = []
        self.path = None
        self.save_path = dir_save
        # figures
        self.figure = None

    def create_save_path(self):
        if self.save_path is None:
            if os.path.exists(self.dir_path.joinpath('Figs')):
                print(f'folder {self.dir_path.joinpath("Figs")} exists.')
            else:
                os.makedirs(self.dir_path.joinpath('Figs'))
                print(f'Created folder {self.dir_path.joinpath("Figs")}')
            save_path = self.dir_path.joinpath('Figs')
        else:
            print(f'save path is set to {self.save_path}.')
            save_path = None
        return save_path

    def reset_def_vals(self):
        from ..experiments.plqe import Plqe
        from analysis.experiments.ta import Ta
        if isinstance(self, Plqe):
            self.reset_plqe()
            print('Data reset to raw data.')
        elif isinstance(self, Ta):
            print(NotImplemented)
        else:
            print(NotImplemented)

    def save_project(self):
        '''
        saving all attributes of the project
        into pickle file
        TODO: add keyed hashing with 'hmac'
        '''
        dict_to_save = sup.dict_from_class(self)
        with open(self.d_path.joinpath('project.pkl'), 'wb') as outfile:
            pickle.dump(dict_to_save, outfile)

    def save_fig(self, **kwargs):
        '''Saves current figure in self._figure created during plotting.
        Author VG last edited 18/05/2020'''
        save_path = kwargs.get('path', self.save_path)
        filetype = kwargs.get('type', 'png')
        name = kwargs.get('name', 'Plot')
        resolution = kwargs.get('dpi', 600)

        # Save as figure
        fname = sup.gen_timed_path(save_path, name, f'.{filetype}')
        self.figure.savefig(fname, dpi=resolution, format=filetype)

        # Save as pickle for future editing possibilities
        fname = sup.gen_timed_path(save_path, name, '.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(self.figure, f)
