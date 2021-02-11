# -*- coding: utf-8 -*-
from .trs import Trs
import pathlib as p

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
        raise NotImplementedError
        # print(f'loading data {self.path}')
