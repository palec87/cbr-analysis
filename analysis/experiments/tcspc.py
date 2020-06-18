# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:19:44 2020

@author: David Palecek
"""
from .trs import Trs

__all__ = ['Tcspc']


class Tcspc(Trs):
    def mod_info():
        print(f'mod info from {object.__name__}')
