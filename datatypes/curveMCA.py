# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:44:58 2016

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np
from re import findall as refindall

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.mathModule import is_number


class GraphMCA(Graph):
    
    FILEIO_GRAPHTYPE = 'XRF MCA raw data'
    
    AXISLABELS = [['XRF detector channel', '', ' '], ['Counts per s', '', 'cps']]

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        if fileExt == '.mca' and line1 == '<<PMCA SPECTRUM>>':
            return True
        return False
        
    def readDataFromFile(self, attributes, **kwargs):
        # parse file
        y = np.array([])
        f = open(self.filename, 'r')
        line = ''
        while line != '<<DATA>>':
            line = f.readline().strip(' \r\n\t')
            split = refindall('([^-]+) - ([^-]+)', line)
            if len(split) > 0 and len(split[0]) > 1:
                self.sampleInfo[split[0][0]] = split[0][1]
        flag = True
        while flag:
            line = f.readline().strip(' \r\n\t')
            if is_number(line):
                y = np.append(y, float(line))
            else:
                flag = False
        while 1:
            line = f.readline().strip(' \r\n\t')
            if not line:
                break
            split = refindall('([^0-9]+): ([-+]?[0-9]*\.?[0-9]+) *([^0-9]*)', line)
            if len(split) > 0 and len(split[0]) > 2:
                tmp = split[0][2]
                if len(split[0][2]) > 0:
                    tmp = ' ['+split[0][2]+']'
                self.sampleInfo[split[0][0]+tmp] = float(split[0][1])
        f.close()
        # format data
        if 'Preset [sec]' not in self.sampleInfo and 'Preset [min]' in self.sampleInfo:
            self.sampleInfo['Preset [sec]'] = self.sampleInfo['Preset [min]'] * 60
        if 'Preset [sec]' in self.sampleInfo:
            attributes.update({'Preset [sec]': self.sampleInfo['Preset [sec]']})
        self.data.append(CurveMCA(np.append(np.arange(y.size), y).reshape((2, y.size)), attributes))
        self.update({'muloffset': "1/"+str(self.getAttribute('Preset [sec]'))})
        # important info, want to store in the curve itself
        self.headers.update({'collabels': ['Channel [ ]', 'Counts [ ]']})
        self.graphInfo.update({'xlabel': self.formatAxisLabel(GraphMCA.AXISLABELS[0]),
                               'ylabel': self.formatAxisLabel(GraphMCA.AXISLABELS[1])})
        if y.size == 1024:
            self.graphInfo.update({'xlim': [0, 1024]})
        print('Opened MCA raw data file. To know composition, open .html file instead.')



        
class CurveMCA(Curve):
    
    CURVE = 'Curve MCA'
    
    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        self.update({'Curve': CurveMCA.CURVE})
        self.update({'_MCA_CtokeV_offset': -3.35, '_MCA_CtokeV_mult': 0.031490109515627})

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        at = ['_MCA_CtokeV_offset', '_MCA_CtokeV_mult']
        out.append([self.updateValuesDictkeys, 'Save',
                    ['keV = (channel +', ') * '],
                    [self.getAttribute(a) for a in at], {'keys': at}])
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
#out.append([self.dataModifySwapChannelkeV, 'change data Channel<->keV', [], []]) # one line per function
        return out
    
    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        out += [['Channel <-> keV', ['MCAkeV', ''], '']]
        return out

