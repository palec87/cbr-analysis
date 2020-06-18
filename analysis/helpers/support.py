# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:14:45 2020

@author: David Palecek
"""
import functools
import numpy as np
import os
import time

print('running support.py')


def dict_from_class(cls):
    class S:
        pass
    _excluded_keys = set(S.__dict__.keys())
    return dict(
        (key, value)
        for (key, value) in cls.__dict__.items()
        if key not in _excluded_keys
        )


def get_idx(*vals, axis):
    '''returns closest index of the 'val'ue along the axis.
    - axis is time/wavelenght values to be converted to indexes
    - sets idx to min/max if out of bounds
    author DP, last change: 28/4/20'''
    idx = [np.argmin(abs(axis-val)) for val in vals]
    return idx


def is_num(obj):
    ''' checking if obj is int/float,
    if not, return False
    author DP, last change 28/04/20'''
    try:
        _ = int(obj)
    except ValueError:
        try:
            _ = float(obj)
        except ValueError:
            return False
        else:
            return True
    else:
        return True


def gen_timed_path(folder, name, suffix):
    '''
    return path to saveFolder which contains timeString
    author DP, last change: 28/4/20
    Parameters
    ----------
    folder : pathlib path
        save folder path.
    name : STR
        user's identifier.
    suffix : TYPE str
        what kind of file is being saved.
    Returns
    -------
    path : TYPE
        DESCRIPTION.
    '''
    timestr = time.strftime("%Y%m%d-%H%M")
    basename = timestr + name
    name = timestr + name
    path = os.path.join(folder, basename+suffix)

    count = 0
    while os.path.isfile(path):
        path = os.path.join(folder, basename + '_' + str(count) + suffix)
        count += 1
    return path


# -----------------------
# ----- Decorators ------
# -----------------------
def refresh_vals(func):
    @functools.wraps(func)
    def wrapper_refresh_vals(*args, **kwargs):
        func(*args, **kwargs)
        args[0].recalc()
    return wrapper_refresh_vals
