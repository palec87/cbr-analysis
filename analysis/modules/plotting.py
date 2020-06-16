# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:51:20 2020

@author: David Palecek
"""
import functools


def log_xscale(graph):
    @functools.wraps(graph)
    def wrapper_log_xscale(*args, **kwargs):
        fig = graph(*args, **kwargs)
        try:
            fig.axes[0].set_xscale(kwargs['xscale'])
        except KeyError:
            pass
        return fig
    return wrapper_log_xscale


def log_yscale(graph):
    @functools.wraps(graph)
    def wrapper_log_yscale(*args, **kwargs):
        fig = graph(*args, **kwargs)
        try:
            fig.axes[0].set_yscale(kwargs['yscale'])
        except KeyError:
            pass
        return fig
    return wrapper_log_yscale


def ylims(graph):
    @functools.wraps(graph)
    def wrapper_ylims(*args, **kwargs):
        fig = graph(*args, **kwargs)
        try:
            fig.axes[0].set_ylim(kwargs['ylim'])
        except KeyError:
            pass
        return fig
    return wrapper_ylims


def xlims(graph):
    @functools.wraps(graph)
    def wrapper_xlims(*args, **kwargs):
        fig = graph(*args, **kwargs)
        try:
            fig.axes[0].set_xlim(kwargs['xlim'])
        except KeyError:
            pass
        return fig
    return wrapper_xlims


def title_plot(graph):
    @functools.wraps(graph)
    def wrapper_title(*args, **kwargs):
        fig = graph(*args, **kwargs)
        try:
            fig.axes[0].set_title(kwargs['title'])
        except KeyError:
            pass
        return fig
    return wrapper_title


def normalize_plot(graph):
    @functools.wraps(graph)
    def wrapper_normalize(*args, **kwargs):
        fig = graph(*args, **kwargs)
        if 'norm' in kwargs:
            data = [line.get_data()
                    for line in fig.axes[0].lines]
            data_norm = normalize_data(data,
                                       method=kwargs['norm'])
            for i in range(len(data)):
                fig.axes[0].lines[i].set_data(data_norm[i])
            fig.axes[0].relim()
        return fig
    return wrapper_normalize


def normalize_data(data: tuple, method: str) -> tuple:
    if method == 'max':
        data_norm = tuple((kin[0], kin[1] / max(kin[1]))
                          for kin in data)
    elif method == 'min':
        data_norm = tuple((kin[0], kin[1] / min(kin[1]))
                          for kin in data)
    elif method == 'abs':
        data_norm = tuple((kin[0], kin[1] / max(abs(kin[1])))
                          for kin in data)
    else:
        print('Unknown normalization method')
        data_norm = data
    return data_norm
