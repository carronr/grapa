# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 10:57:14 2017

@author: Romain
"""

import os
import numpy as np
from re import findall as refindall


from grapa.graph import Graph
from grapa.curve import Curve



class GraphMCAfit(Graph):
    
    FILEIO_GRAPHTYPE = 'XRF fit areas'
    
    AXISLABELS = [['Data', '', None], ['Value', '', None]]
    
    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', **kwargs):
        if fileExt == '.html' and line1[0:24] == '<HTML><HEAD><TITLE>PyMCA':
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        filenam_, fileext = os.path.splitext(self.filename)
        sample = filenam_.split('/')[-1].split('.mca')[0]
        f = open(self.filename, 'r')
        content = f.read()
        f.close()
        tmp = np.array([])
        content = content.replace(' align="left"', '').replace(' align="right"', '').replace(' bgcolor=#E6F0F9', '').replace(' bgcolor=#E6F0F9', '').replace(' bgcolor=#FFFFFF', '').replace(' bgcolor=#FFFACD', '').replace('  ', ' ').replace('<td >', '<td>').replace('<tr', '\n<tr').replace('<TR', '\n<TR').replace('<table', '\n<table').replace('\n\n', '\n')
        split = refindall('<tr><td>Cu</td><td>K</td><td>([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)</td><td>', content)
        tmp = np.append(tmp, float(split[0][0]))
        split = refindall('<tr><td>In</td><td>K</td><td>([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)</td><td>', content)
        tmp = np.append(tmp, float(split[0][0]))
        split = refindall('<tr><td>Ga</td><td>Ka</td><td>([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)</td><td>', content)
        tmp = np.append(tmp, float(split[0][0]))
        split = refindall('<tr><td>Se</td><td>K</td><td>([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)</td><td>', content)
        tmp = np.append(tmp, float(split[0][0]))
        tmp = np.append(tmp, 180) # time - assumed here, no value reading
    ## XRF calibration: MIGHT WANT TO UPDATE THAT!!!
        val_Cu = tmp[0] / 43434 * 22.56
        val_In = tmp[1] / 10992 * 13.66
        val_Ga = tmp[2] / 29578 * 11.41
        val_Se = tmp[3] /195460 * 52.40
    ## END OF XRF calibration: MIGHT WANT TO UPDATE THAT!!!
        val_sum = val_Cu + val_In + val_Ga + val_Se
        val_Cu /= val_sum
        val_In /= val_sum
        val_Ga /= val_sum
        val_Se /= val_sum
        tmp = np.append(tmp, val_Ga / (val_Ga + val_In)) # x calculation
        tmp = np.append(tmp, val_Cu / (val_Ga + val_In)) # y calculation
        tmp = np.append(tmp, tmp[0] * 450 / 10000000) # D calculation
        self.data.append(Curve(np.append(np.arange(tmp.size), tmp).reshape((2, tmp.size)), {}))
        c = self.curve(-1)
        c.update(attributes)
        c.update({'XRF fit Cu [fitarea/s]': float(tmp[0]) / float(tmp[4])})
        c.update({'XRF fit In [fitarea/s]': float(tmp[1]) / float(tmp[4])})
        c.update({'XRF fit Ga [fitarea/s]': float(tmp[2]) / float(tmp[4])})
        c.update({'XRF fit Se [fitarea/s]': float(tmp[3]) / float(tmp[4])})
        c.update({'XRF fit time [s]':float(tmp[4])})
        c.update({'XRF fit x [ ]'  : float(tmp[5])})
        c.update({'XRF fit y [ ]'  : float(tmp[6])})
        c.update({'XRF fit D [um]' : float(tmp[7])})
        # other info
        self.headers.update({'sample': sample, 'collabels': ['Element [ ]', 'Fit area / value [a.u.]']})
        self.update({'xlabel': self.formatAxisLabel(GraphMCAfit.AXISLABELS[0]),
                     'ylabel': self.formatAxisLabel(GraphMCAfit.AXISLABELS[1]),
                     'xtickslabels': [[0, 1, 2, 3, 4, 5, 6, 7],
                                      ['Cu', 'In', 'Ga', 'Se', 'Acq. time', 'GGI', 'CGI', 'D']]})
        c.update({'linespec': 'd'})
        c.update({'label': c.getAttribute('label').split('.')[0]})
        print('   XRF composition: GGI', '{:1.3f}'.format(tmp[5]),
              ', CGI', '{:1.3f}'.format(tmp[6]),
              ', D', '{:1.3f}'.format(tmp[7]),'(calibration x=0.3).')
        GaSe = 1.63   * tmp[2] / tmp[3]
        CuSe = 2.0661 * tmp[0] / tmp[3]
        GGI_ = 2 * GaSe / (1 + 1/3 * (1 - 2 * CuSe))
        CGI_ = 2 * CuSe / (1 + 1/3 * (1 - 2 * CuSe))
        D_ = 9.50E-06 * tmp[3]
        print('   XRF Se ratios composition: GGI', '{:1.3f}'.format(GGI_),
              ', CGI', '{:1.3f}'.format(CGI_), 'D', '{:1.3f}'.format(D_),
              '(calibration Ga/Se, Cu/Se)')
