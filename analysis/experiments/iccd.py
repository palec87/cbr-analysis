# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:19:59 2020

@author: David Palecek
"""
from .trs import Trs
print('running iccd init')

__all__ = ['Iccd', 'mod_info']

class Iccd():
    def mod_info():
        print(f'mod info from {object.__name__}')
        
def helper_func():
    print('not important func')
    
def mod_info():
    print('INFO: iccd module')
