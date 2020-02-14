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

class GraphSquidAdmiral(Graph):
    """
    reads CSV exports of the Squid system
    """

    FILEIO_GRAPHTYPE = 'Admiral Squid data CSV export'
    
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
            elif 'Galvanostatic EIS ' in fileName:
                return True
            elif 'Charge_Discharge ' in fileName:
                return True
        return False
        
    
        
    def readDataFromFile(self, attributes, **kwargs):
        # no headers to parse
        basename = os.path.basename(self.filename)
        usecols, labels, data = None, None, None
        
        # retieve instructions for openning file
        if 'Cyclic Voltammetry Start ' in basename:
            usecols = (3,5)
        elif 'Potentiostatic EIS ' in basename:
            usecols = (8,9)
        elif 'Galvanostatic EIS ' in basename:
            usecols = (8,9)
        elif 'Open Circuit Potential ' in basename:
            usecols = (2,3)
        elif 'Constant Current ' in basename:
            try:
                data = np.genfromtxt(self.filename, skip_header=1,
                                     delimiter=',', invalid_raise=False)
            except Exception as e:
                print('GraphSquidAdmiral Constant Current: actually, cannot read file!?')
                print(e)
                return False
            x, tm = [], data[0,2]
            y = data[:,4]
            for line in data: # some quite dirty and innefficient data parsing
                x.append((x[-1] if len(x)>0 else 0) + (line[2]-tm) * line[5])
                tm = line[2]
            if len(x) > 0:
                self.append(Curve([np.abs(np.array(x))/3600,y], attributes))
            labels = [['Charge', '', 'mAh'], ['Working Electrode', '', 'V']]
            data, usecols = 1, 1 # bypass default data opening, ensure algo knows everything ok
        elif 'Charge_Discharge ' in basename:
            try:
                data = np.genfromtxt(self.filename, skip_header=1,
                                     delimiter=',', invalid_raise=False)
            except Exception as e:
                print('GraphSquidAdmiral Charge_Discharge: actually, cannot read file!?')
                print(e)
                return False
            def createCurve(x, y, sign):
                self.append(Curve([np.abs(np.array(x))/3600,y], attributes))
                if sign < 0:
                    self.curve(-1).update({'label': self.curve(-1).getAttribute('label').replace('Charge ', '')})
                else:
                    self.curve(-1).update({'label': self.curve(-1).getAttribute('label').replace('Discharge ', '')})
            x, y, tm, previous = [], [], data[0,2], np.sign(data[0,5])
            for line in data: # some quite dirty and innefficient data parsing
                if np.sign(line[5]) != previous: # if change of sign of current -> from charge to discharge or opposite
                    createCurve(x, y, previous)
                    x, y = [], []
                    previous = np.sign(line[5])
                x.append((x[-1] if len(x)>0 else 0) + (line[2]-tm) * line[5])
                y.append(line[4])
                tm = line[2]
            if len(x) > 0:
                createCurve(x, y, previous)
            labels = [['Charge', '', 'mAh'], ['Working Electrode', '', 'V']]
            data, usecols = 1, 1 # bypass default data opening, ensure algo knows everything ok
        # open file
        if usecols is None:
            print('GraphSquidAdmiral: actually, cannot read file!?')
            return False
        # default data file reading
        if data is None:
            try:
                data = np.genfromtxt(self.filename, skip_header=1, delimiter=',',
                                     usecols=usecols, invalid_raise=False)
            except Exception as e:
                    print('WARNING GraphSquidAdmiral cannot read file', self.filename)
                    print(type(e), e)
                    return False
            self.append(Curve(np.transpose(data), attributes))
        if labels is None:
            labels = [kwargs['line1'].split(',')[c][1:-2].replace('""',"''") for c in usecols]
            labels = [[l.split(' (')[0], '', l.split(' (')[-1]] for l in labels]
        self.update({'xlabel': self.formatAxisLabel(labels[0]),
                     'ylabel': self.formatAxisLabel(labels[1])})
                    
