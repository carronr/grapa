# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:37:38 2017

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

from re import findall as refindall
from copy import deepcopy
import numpy as np

from grapa.mathModule import is_number
from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.curve import Curve


class GraphCf(Graph):

    FILEIO_GRAPHTYPE = 'C-f curve'

    AXISLABELS = [['Frequency', 'f', 'Hz'], ['Capacitance', 'C', 'nF']]
    AXISLABELSNYQUIST = [['Real(Z)', 'Z\'', 'Ohm'],
                         ['- Imag(Z)', '-Z"', 'Ohm']]

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', **kwargs):
        if (fileExt == '.txt' and line1.strip()[:13] == 'Sample name: '
                and fileName[0:4] == 'C-f_'):
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        """
        Possible kwargs:
            - _CfLoadPhase: also load the phase, in degrees
        """
        len0 = len(self)
        GraphIO.readDataFromFileGeneric(self, attributes)
        self.castCurve('Curve Cf', len0, silentSuccess=True)
        # label based on file name, maybe want to base it on file content
        lbl = self[len0].attr('label').replace('C-f ', '').replace(' Cp', '').replace(' C', '').replace('T=', '').replace(' [nF]', '').replace(' (nF)', '')
        self[len0].update({'label': lbl, 'sample': lbl})
        # retrieve units, change label
        colName = self[len0].attr('frequency [Hz]', -1)  # retrieve actual unit
        ylabel = GraphCf.AXISLABELS
        convertToF = 1e-9
        if isinstance(colName, str):
            # retrieve yaxis information in file, interpret it
            ylabel = colName.replace('C ', 'Capacitance ').replace('Cp ', 'Capacitance ').replace('(','[').replace(')',']')
            expr = r'^(.* )\[(.*)\]$'
            f = refindall(expr, ylabel)
            if isinstance(f, list) and len(f) == 1 and len(f[0]) == 2:
                ylabel = [f[0][0], GraphCf.AXISLABELS[1][1], f[0][1]]
            if f[0][1] != 'nF':
                print('CurveCf read, detected capacitance unit', f[0][1], '.',
                      'Cannot proceed further with phase and Nyquist.')
                convertToF = None
            # TODO: add warning if not nF
        # identify Rp curve
        idxRp = None
        for c in range(len0, len(self)):  # retrieve resistance curve
            lbl = self[c].attr('frequency [Hz]', None)  # retrieve actual unit
            if lbl is None:
                lbl = self[c].attr('label')
            # print('scan curves label', c, lbl)
            if lbl.endswith('R') or lbl.endswith('Rp') or lbl.endswith('Rp [Ohm]'):
                idxRp = c
                break
        # normalize with area C, R
        area = self[len0].attr('cell area (cm2)', None)
        if area is None:
            area = self[len0].attr('cell area', None)
        if area is None:
            area = self[len0].attr('area', None)
        if area is not None:
            self[len0].setY(self[len0].y() / area)
            if idxRp is not None:
                self[idxRp].setY(self[idxRp].y() * area)
            self[len0].update({'cell area (cm2)': area})
            if not self.silent:
                print('Capacitance normalized to area', self[len0].getArea(),
                      'cm2.')
        # generate additional curves if interesting
        nb_add = 0
        # if phase is required
        if self.attr('_CfLoadPhase', False) is not False:
            f = self[len0].x()
            # C input assumed to be [nF], need [F] for calculation
            C = self[len0].y() * convertToF
            if idxRp is None:
                print('Warning CurveCf read file', self.filename, ': cannot',
                      'find R. Cannot compute phase.')
            else:
                conductance = 1 / self[idxRp].y()
                phase_angle = np.arctan(f*2*np.pi*C/conductance)*180. / np.pi
                self.append(Curve([f, phase_angle],
                                  deepcopy(self[len0].getAttributes())))
                self[-1].update({'_CfPhase': True})
                nb_add += 1

        if self.attr('_CfLoadNyquist', False) is not False:
            if idxRp is None:
                print('ERROR CurveCf read file', self.filename, ': cannot',
                      'find Rp. Cannot compute Nyquist plot.')
            else:
                # C input assumed to be [nF], need [F] for calculation
                C = self[len0].y() * convertToF
                omega = self[len0].x() * 2 * np.pi
                Rp = self[idxRp].y()
                Z = 1 / (1 / Rp + 1j * omega * C)
                self.append(Curve([Z.real, -Z.imag],
                                  deepcopy(self[len0].getAttributes())))
                self[-1].update({'_CfNyquist': True})
                nb_add += 1

        # check temperature is parsed
        if self[len0].attr('temperature [k]', None) is None:
            def guessed(value):
                if value > 5 and value < 1000:
                    # plausability check for guessed temperature
                    import os
                    print('File', os.path.basename(self.filename),
                          'temperature guessed', value)
                    self[len0].update({'temperature': value})
                    return True
                return False
            guess = self[len0].attr('label').split(' ')[-1]
            if guess.endswith('K') and is_number(guess[:-1]):
                flag = guessed(float(guess[:-1]))
            if not flag:
                for c in range(len(self)-1-nb_add, len0, -1):
                    if 'Temperature' in self[c].attr('label'):
                        try:
                            guessed = self[c].y()
                            guessed = np.average(guessed[~np.isnan(guessed)])
                            if guessed(guessed):
                                continue
                        except Exception:
                            pass
        # delete Rp, temperature, etc
        for c in range(len(self)-1-nb_add, len0, -1):
            self.deleteCurve(c)
        # cosmetics
        self.update({'typeplot': 'semilogx', 'alter': ['', 'idle']})
        self.update({'xlabel': self.formatAxisLabel(GraphCf.AXISLABELS[0]),
                     'ylabel': self.formatAxisLabel(ylabel)})  # default
        if self[len0].attr('cell area (cm2)', None) is not None:
            ylabel = self.attr('ylabel').replace('F', 'F cm$^{-2}$')
            self.update({'ylabel': ylabel})
