# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 10:44:14 2017

@author: Romain
"""

import os
import numpy as np


from grapa.graph import Graph
from grapa.curve import Curve

class GraphMBElog(Graph):
    
    FILEIO_GRAPHTYPE = 'Small MBE log file'
    
    AXISLABELS = [['Time', 't', 'min'], ['Substrate heating', '', 'a.u.']]
    
    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', **kwargs):
        if fileExt == '.txt' and line1 == 'Time	Value':
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        """ Reads MBE log file. Curve is standard xyCurve. """
        fileName, fileext = os.path.splitext(self.filename)
        sample = (fileName.split('/')[-1]).split('_')[0]
        data = np.genfromtxt(self.filename, skip_header=1, delimiter='\t', 
                             usecols=[0, 1], invalid_raise=False)
        self.data.append(Curve(np.transpose(data), attributes))
        self.headers.update({'collabels': ['Time [min]',
                                           'Substrate heating [a.u.]']})
        self.update({'xlabel': self.formatAxisLabel(GraphMBElog.AXISLABELS[0]),
                     'ylabel': self.formatAxisLabel(GraphMBElog.AXISLABELS[1])})
        self.curve(-1).update({'label': sample})


