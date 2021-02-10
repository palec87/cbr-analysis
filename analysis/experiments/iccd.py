# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:19:59 2020

@author: David Palecek
"""
from .trs import Trs
import pathlib as p
print('running iccd init')

__all__ = ['Iccd']


class Iccd(Trs):
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
        print(f'loading data {self.path}')
