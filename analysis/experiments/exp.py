# -*- coding: utf-8 -*-
import pickle
import os
from ..helpers import support as sup
import pathlib as p

__all__ = ['Exp']


class Exp():
    '''
    Class of all the experiments. Parent class to all the other.
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
        """Generate save_path if it was not specified in loading procedure.

        Returns:
            pathlib path: path to save any output.
        """
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
        """Reset dataset to default (after loading)

        Raises:
            NotImplementedError: So far only for PLQE and Ta
        """
        from ..experiments.plqe import Plqe
        from analysis.experiments.ta import Ta
        if isinstance(self, Plqe):
            self.reset_plqe()
            print('Data reset to raw data.')
        elif isinstance(self, Ta):
            self.reset_ta()
        else:
            raise NotImplementedError

    def save_project(self):
        """saving all attributes of the project
        into pickle file

        TODO: add keyed hashing with 'hmac'
        """
        dict_to_save = sup.dict_from_class(self)
        with open(self.save_path.joinpath('project.pkl'), 'wb') as outfile:
            pickle.dump(dict_to_save, outfile)

    def load_project(self, name='project.pkl', path=None):
        """Load project from a pickle. If not in the 'save_path', then name of the file
        and path specifies the file.

        Args:
            name (str, optional): filename. Defaults to 'project.pkl'.
            path (pathlib path, optional): path to folder of the pickle.
                Defaults to None.
        """
        load_path = self._check_path_argument(name, path)
        proj = pickle.load(open(load_path, "rb"))
        self.__dict__.update(proj)
        print(f'project loaded from: {load_path}')

    def save_fig(self, **kwargs):
        '''Saves current figure in self._figure created during plotting.
        Author VG last'''
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

    def _check_path_argument(self, name, path):
        """Check if the path of the user input is valid.

        Args:
            name (str): filename
            path (pathlib): folder path

        Returns:
            PurePath: absolute path
        """
        if path is None:
            try:
                return self.save_path.joinpath(name)
            except AttributeError:
                print('No save_path, use "path" input')
                raise
        else:
            return p.PurePath(path).joinpath(name)
