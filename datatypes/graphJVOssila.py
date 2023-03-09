# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

import numpy as np

from grapa.graph import Graph
from grapa.datatypes.curveJV import CurveJV


class GraphJVOssila(Graph):
    """
	Reads JV .csv files exported from Ossila 
    """

    FILEIO_GRAPHTYPE = 'J-V curve'

    AXISLABELS = [['XRF detector channel', '', ' '],
                  ['Counts', '', 'cts']]

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        if fileExt == '.csv' and line1.startswith('Pixel ') and line1.endswith(',') and line2.startswith('V (V),J (A/cm^2)'):
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        # read 2 header lines
        with open(self.filename, 'rb') as file:
            labels = [s.strip() for s in file.readline().decode('ascii').split(',')]
            units  = [s.strip() for s in file.readline().decode('ascii').split(',')]
        # read data
        data = np.transpose(np.genfromtxt(self.filename, skip_header=2, delimiter=',', invalid_raise=False))
        col = 0
        while col + 2 <= data.shape[0]:
            datax = data[col, :]
            datay = data[col+1, :]
            if 'J (A/cm^2)' in units[col+1]:
                units[col+1] = units[col+1].replace('J (A/cm^2)', 'J (mA/cm^2)')
                datay *= 1000
                print('GraphJVOssila converted A/cm2 into mA/cm2')
            if len(datax) > 1 and datax[1] < datax[0]:
                # sort data in ascending order, because JV Curve wants that way
                datax = datax[::-1]
                datay = datay[::-1]
                print('GraphJVOssila reversed order of data (', labels[col],')')
            self.append(CurveJV([datax, datay], attributes))
            self[-1].update({'label': labels[col],
                             'units': units[col] + ', ' + units[col+1]})
            col += 2
        # graph cosmetics
        self.update({'xlabel': self.formatAxisLabel(['Bias voltage', 'V', 'V']),
                     'ylabel': self.formatAxisLabel(['Current density', 'J', 'mA cm$^{-2}$'])})
        self.update({'axhline': [0, {'linewidth': 0.5}], 'axvline': [0, {'linewidth': 0.5}]})
