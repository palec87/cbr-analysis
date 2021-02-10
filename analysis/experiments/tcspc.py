# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:19:44 2020

@author: David Palecek
"""
from .trs import Trs

__all__ = ['Tcspc']


class Tcspc(Trs):
    def __init__(self, dir_save):
        super().__init__(dir_save)
        self.info = f'Class instance of {self.__class__}'
        self.path = None
        print(self.info)
