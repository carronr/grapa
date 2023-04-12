# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:25:49 2021

@author: Romain Carron
Copyright (c) 2019, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

import numpy as np
import os

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.graphIO import GraphIO


class GraphPAIOS(Graph):
    """
    reads txt exports of PAIOS system
    """

    FILEIO_GRAPHTYPE = 'PAIOS export'

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        if (fileExt == '.txt'
                and line1.startswith('### ----')
                and line2.startswith('### PAIOS: Platform for All-In-One')
                and line3.startswith('### Fluxim AG, www.fluxim.com')):
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        GraphIO.readDataFromFileGeneric(self, attributes, lstrip=' #',
                                        delimiterHeaders=': ')
        # retrieve data into separate columns
        columns = []
        if len(self) > 0:
            columns.append({'data': self[0].x(), 'attributes': {}})
            columns[-1]['attributes'].update({'label': self.attr('column 01')})
        for c in range(len(self)):
            label = self.attr('column '+'{:02d}'.format(c+2))
            # print(c, 'label', label, '_', 'column '+'{:02d}'.format(c+2))
            columns.append({'data': self[c].y(),
                            'attributes': self[c].getAttributes()})
            columns[-1]['attributes'].update({'label': label})
        # remove all content
        while len(self) > 0:
            self.deleteCurve(0)
        # correctly arrange data into Curves
        labels = ['', '']
        taken = [False] * len(columns)
        c = 0
        while True:
            if not taken[c]:
                lbl = columns[c]['attributes']['label']
                tmp = lbl.split(', ')
                # print(c, 'lbl', lbl, 'tmp', tmp)
                for d in range(c+1, len(columns)):
                    lbl_ = columns[d]['attributes']['label']
                    tmp_ = lbl_.split(', ')
                    if not taken[d] and tmp[0] == tmp_[0]:
                        # print(' ', d, 'lbl_', lbl_, 'tmp_', tmp_)
                        attrs = columns[c]['attributes']
                        attrs.update(columns[d]['attributes'])
                        label = tmp[0]  # +', '+tmp_[1]+' vs '+tmp[1]
                        attrs.update({'label': label})
                        # remove possible padded 0 at the end
                        x = list(columns[c]['data'])  # work on a copy
                        y = list(columns[d]['data'])
                        while x[-1] == 0 and y[-1] == 0:
                            x.pop()
                            y.pop()
                        self.append(Curve([x, y], attrs))
                        labels = [tmp[1], tmp_[1]]
                        taken[c] = True
                        taken[d] = True
                        break
            c += 1
            if c >= len(columns):
                break
        if np.sum(not taken) > 0:  # warn the user if detected failed import
            print('CAUTION, Data curve not loaded! columns: ', taken)
        # graph cosmetics
        legtitle = os.path.splitext(os.path.basename(self[0].attr('filename')))
        legtitle = legtitle[0].replace('_', ' ').replace('  ', ' ')
        self.update({'legendtitle': legtitle,  # legendtitle from filename
                     'xlabel': self.formatAxisLabel(labels[0]),
                     'ylabel': self.formatAxisLabel(labels[1])})
        # cleaning: remove attributes 'column 01' etc. in curve 0
        if len(self) > 0:
            i = 1
            while self[0].attr('column '+'{:02d}'.format(i)) != '':
                self[0].update({'column '+'{:02d}'.format(i): ''})
                i += 1
        # setting-up properly: copy metadata in all curves
        attrs = dict(self[0].getAttributes())  # lets work on a copy
        del attrs['label']
        for c in range(1, len(self)):
            self[c].update(attrs)
        # convert if needed (to JV curve)
        if labels == ['Device Voltage (V)', 'Device Current (mA)']:
            GraphPAIOS._convertToJV(self)

    def _convertToJV(self):
        """
        if was detected that content are JV data, convert into suitable Curve
        type and do required changes and extract Jsc-Voc data
        """
        jscvoc = []
        for c in range(len(self)):
            self[c].setX(self[c].x()[::-1])  # data in ascending order
            self[c].setY(self[c].y()[::-1])
            self.castCurve('CurveJV', c, silentSuccess=True)
            jsc = self[c].attr('jsc', np.nan)
            voc = self[c].attr('voc', np.nan)
            temp = self[c].attr('temperature')
            if temp == '':
                temp = 273.15 + 25
            jscvoc.append([voc, jsc, temp])
        # extract Jsc-Voc curves
        if len(jscvoc) > 0:
            jscvoc = np.array(jscvoc)
            self.append(Curve([jscvoc[:, 0], jscvoc[:, 1]],
                              {'label': 'Jsc-Voc', 'type': 'scatter'}))
            self.append(Curve([jscvoc[:, 0], jscvoc[:, 2]],
                              {'label': 'Jsc-Voc - Temperature',
                               'type': 'scatter_c'}))
            self.castCurve('CurveJscVoc', -2, silentSuccess=True)
