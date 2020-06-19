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
    idx = [np.argmin(abs(np.array(axis)-val)) for val in vals]
    return idx


def mean_subarray(array: np.ndarray,
                  axis: int,
                  rng=None,
                  ax_data=None) -> np.float or np.ndarray:
    '''
    return mean of subarray along 'axis'
    within the 'rng' of values in 'ax_data'.
    Returned array 1 less dimension then 'array'

    Positional arguments:
    array -- numpy array of dimension
    axis -- int: along which axis the mean is taken
    rng -- range of ax_data to be averaged
        if None, then taken as whole range
    ax_data -- the rng is applied to this axis
        has to have len of axis

    Return:
        array of dimension - 1 compared to array
            float in case of 1D input
            subarray on closed interval of rng values
            [rng1, rng2]
    '''
    if axis >= len(array.shape):
        print(f'axis with value:{axis} out of array dim:\
              {len(array.shape)}')
        return
    if ax_data is None:
        ax_data = np.linspace(0,
                              array.shape[axis]-1,
                              array.shape[axis]
                              )
    if len(ax_data) != array.shape[axis]:
        print('Provided axis is wrong length.')
        return
    if rng is not None:
        beg, end = get_idx(*rng, axis=ax_data)
    else:
        beg, end = 0, len(ax_data)-1

    subarray = np.mean(np.take(array,
                               indices=range(beg, end+1),
                               axis=axis),
                       axis=axis)
    return subarray


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
