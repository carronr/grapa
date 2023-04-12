# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:21:57 2018

@author: Romain
"""
from copy import copy
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import matplotlib.pyplot as plt

from grapa.curve_inset import Curve_Inset
from grapa.curve_subplot import Curve_Subplot



class ParserAxlines:
    # Possible inputs:
    # - 3
    # - [3, {'color':'r', 'xmin': 0.5]
    # - [1, 2, 4, {'color':'b'}]
    # - [[1, {'color':'c'}], [2, {'color':'m'}]]
    def __init__(self, data):
        self._list = []
        if not isinstance(data, list):  # handles "3"
            self.__init__([data])
            return
        # from now on data assumed to be a list
        kwdef = {'color': 'k'}
        if isinstance(data[-1], dict):  # handles [..., {'color':'m'}]
            kwdef.update(data[-1])
            data = data[:-1]
        for el in data:
            kw = copy(kwdef)
            if not isinstance(el, list):  # handles [1,2,3...]
                self._list.append({'args': [el], 'kwargs': kw})
            else:
                el_ = el
                # handles [[1,2, {'color':'y'], [3, {'color': 'm'}]]
                if isinstance(el[-1], dict):
                    kw.update(el[-1])
                    el_ = el[:-1]
                for e in el_:
                    self._list.append({'args': [e], 'kwargs': kw})

    def __str__(self):
        return 'self\t' + '\n\t'.join([str(a) for a in self._list])

    def plot(self, method):
        for el in self._list:
            method(*el['args'], **el['kwargs'])


class ParserAxhline(ParserAxlines):
    def plot(self, method, curvedummy, alter, type_plot):
        for el in self._list:
            pos = curvedummy.y(alter=alter[1], xyValue=[1, el['args'][0]],
                               errorIfxyMix=True, neutral=True)
            if pos != 0 or type_plot not in ['semilogy', 'loglog']:
                try:
                    method(pos, **el['kwargs'])
                except Exception as e:
                    print('Keyword axhline, exception', type(e),
                          'in ParserAxhline.plot', e)
        return True


class ParserAxvline(ParserAxlines):
    def plot(self, method, curvedummy, alter, type_plot):
        for el in self._list:
            pos = curvedummy.x(alter=alter[0], xyValue=[el['args'][0], 1],
                               errorIfxyMix=True, neutral=True)
            if pos != 0 or type_plot not in ['semilogx', 'loglog']:
                try:
                    method(pos, **el['kwargs'])
                except Exception as e:
                    print('Keyword axvline, exception', type(e),
                          'in ParserAxhline.plot', e)
        return True

