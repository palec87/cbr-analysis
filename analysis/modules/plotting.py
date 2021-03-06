# -*- coding: utf-8 -*-
"""
Plotting related classes and functions.
"""
import functools
import matplotlib.cm as cm
import numpy as np


class LineBuilder:
    """Based on mouse clicks on the plot, [x,y] coordinates are taken
    into a list to generate a line.
    """
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes != self.line.axes:
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()


def log_xscale(graph):
    """Wrapper to set xscale of the plots

    Args:
        graph: matplotlib.figure

    Returns:
        [type]: matplotlib.figure
    """
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
    """Wrapper to set yscale of the plots

    Args:
        graph: matplotlib.figure

    Returns:
        [type]: matplotlib.figure
    """
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
    """Wrapper to set ylim [min,max] of the plots

    Args:
        graph: matplotlib.figure

    Returns:
        [type]: matplotlib.figure
    """
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
    """Wrapper to set xlim [min,max] of the plots

    Args:
        graph: matplotlib.figure

    Returns:
        [type]: matplotlib.figure
    """
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
    """Wrapper to set title of the plot

    Args:
        graph: matplotlib.figure

    Returns:
        [type]: matplotlib.figure
    """
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
    """Wrapper to normalize lines in the lineplot. Methods so far
    are: 'max/min/abs/area'

    Args:
        graph: matplotlib.figure

    Returns:
        [type]: matplotlib.figure
    """
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


def set_cmap(graph):
    """Wrapper to change matplotlib.cmap of the plot

    Args:
        graph: matplotlib.figure

    Returns:
        [type]: matplotlib.figure
    """
    @functools.wraps(graph)
    def wrapper_set_cmap(*args, **kwargs):
        fig = graph(*args, **kwargs)
        if 'cmap' in kwargs:
            data = [line.get_data()
                    for line in fig.axes[0].lines]
            color = cm.get_cmap(kwargs['cmap'])(
                        np.linspace(0, 1, len(data)))  # define cmap
            for i in range(len(data)):
                fig.axes[0].lines[i].set_color(color[i])
        return fig
    return wrapper_set_cmap


def normalize_data(data: tuple, method: str) -> tuple:
    """Normalize lines in the lineplot

    Args:
        data (tuple): tuple of x and y values
        method (str): normalization method.

    Returns:
        tuple: Normalized data
    """
    if method == 'max':
        data_norm = tuple((kin[0], kin[1] / max(kin[1]))
                          for kin in data)
    elif method == 'min':
        data_norm = tuple((kin[0], kin[1] / min(kin[1]))
                          for kin in data)
    elif method == 'abs':
        data_norm = tuple((kin[0], kin[1] / max(abs(kin[1])))
                          for kin in data)
    elif method == 'area':
        data_norm = tuple((kin[0], kin[1] / np.trapz(abs(kin[1])))
                          for kin in data)
    else:
        print('Unknown normalization method')
        data_norm = data
    return data_norm
