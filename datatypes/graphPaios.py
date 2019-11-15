# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:25:49 2019

@author: Romain Carron
Copyright (c) 2019, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np
import os

from grapa.graph import Graph
from grapa.curve import Curve

class GraphPaios(Graph):
    """
    reads CSV exports of the PAIOS system
    """

    FILEIO_GRAPHTYPE = 'PAIOS data CSV export'
    
    AXISLABELS = [['Time', '', 's'], ['Current', '', 'mA']]

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', **kwargs):
        if fileExt == '.csv' and line1.startswith('"Step number","Step name","Elapsed'):
            if 'Constant Current ' in fileName:
                return True
            elif 'Cyclic Voltammetry Start ' in fileName:
                return True
            elif 'Potentiostatic EIS ' in fileName:
                return True
            elif 'Open Circuit Potential ' in fileName:
                return True
        return False
        
    
        
    def readDataFromFile(self, attributes, **kwargs):
        # no headers to parse (!?!)
        basename = os.path.basename(self.filename)
        usecols, labels = None, None
        # retieve instructions for openning file
        if 'Constant Current ' in basename:
            usecols = (7,3)
            #attr.update({'label': attributes['label'].replace('Constant Current ','')})
        elif 'Cyclic Voltammetry Start ' in basename:
            usecols = (3,5)
        elif 'Potentiostatic EIS ' in basename:
            usecols = (9,8)
        elif 'Open Circuit Potential ' in basename:
            usecols = (2,3)
        # open file
        if usecols is None:
            print('GraphPaios: actually, cannot read file!?')
            return False
        try:
            data = np.genfromtxt(self.filename, skip_header=1, delimiter=',',
                                 usecols=usecols, invalid_raise=False)
            self.append(Curve(np.transpose(data), attributes))
        except Exception as e:
                print('WARNING GraphPaios cannot read file', self.filename)
                print(type(e), e)
                return False
        if labels is None:
            labels = [kwargs['line1'].split(',')[c][1:-2].replace('""',"''") for c in usecols]
            labels = [[l.split(' (')[0], '', l.split(' (')[-1]] for l in labels]
        self.update({'xlabel': self.formatAxisLabel(labels[0]),
                     'ylabel': self.formatAxisLabel(labels[1])})
                    
