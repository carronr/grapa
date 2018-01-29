# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:18:06 2017

@author: Romain
Copyright (c) 2018, Empa, Romain Carron
"""

import numpy as np
from grapa.graph import Graph
from grapa.curve import Curve

class GraphMydata(Graph):
    # will be used by grapa code
    FILEIO_GRAPHTYPE = 'MyCurve'
    # 3-element list, default x and y axis labels: [text, symbol, unit]
    AXISLABELS = [['theta', '\theta', 'unit'], ['value', '', 'unit']]
    
    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        """
        This function must return True if the file is recognized as of your data format.
        Please be selective so as not to open other file formats by mistake.
        Are provided the file name, extension, and the 3 first lines of the file
        """
        print('File open GraphMydata', (fileExt == '.txt'), 'line1', line1)
        if fileExt == '.txt' and line1 == 'This is my own data format':
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        """
        This function parses your files
        The file can be opened using open(self.filename, 'r')
        """
        print('File open GraphMydata')
        # interpret headers - here only attribute 'label'
        attrs = {}
        f = open(self.filename, 'r')
        f.readline()
        attrs['label'] = f.readline().strip()
        f.close()
        # read data
        data = np.array([[np.nan], [np.nan]])
        kw = {'skip_header': 2, 'usecols': [0, 1]}
        try:
            data = np.transpose(np.genfromtxt(self.filename, **kw))
        except Exception as e:
            if not self.silent:
                print('WARNING cannot read file', self.filename)
                print(type(e), e)
        # create Curve object(s)
        try:
            from grapa.datatypes.curveMycurve import CurveMycurve
            self.append(CurveMycurve(data, attributes))
        except ImportError:
            self.append(Curve(data, attributes))
        self.curve(-1).update(attrs) # file content may override default attributes
        
        # cosmetics
        # some default settings
        self.update({'ylim': [-1.2, 1.2],
                     'xlabel': self.formatAxisLabel(GraphMydata.AXISLABELS[0]),
                     'ylabel': self.formatAxisLabel(GraphMydata.AXISLABELS[1])})
