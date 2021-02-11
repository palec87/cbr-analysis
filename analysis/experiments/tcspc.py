# -*- coding: utf-8 -*-
from .trs import Trs

__all__ = ['Tcspc']


class Tcspc(Trs):
    """Tcspx class, child of Trs.

    Args:
        Trs (class): parent class
    """
    def __init__(self, dir_save):
        super().__init__(dir_save)
        self.info = f'Class instance of {self.__class__}'
        self.path = None
        print(self.info)

    def load_data(self):
        raise NotImplementedError
