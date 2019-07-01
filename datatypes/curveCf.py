# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:37:38 2017

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np
from re import findall as refindall
from copy import deepcopy

from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.curve import Curve
from grapa.mathModule import is_number, derivative



class GraphCf(Graph):

    FILEIO_GRAPHTYPE = 'C-f curve'
    
    AXISLABELS = [['Frequency', 'f', 'Hz'], ['Capacitance', 'C', 'nF']] 
    AXISLABELSNYQUIST = [['Real(Z)', 'Z\'', 'Ohm'], ['- Imag(Z)', '-Z"', 'Ohm']]


    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', **kwargs):
        if fileExt == '.txt' and line1.strip()[:13] == 'Sample name: ' and fileName[0:4] == 'C-f_':
            return True
        return False
        
        
    def readDataFromFile(self, attributes, **kwargs):
        """
        Possible kwargs:
            - _CfLoadPhase: also load the phase, in degrees
        """
        len0 = self.length()
        GraphIO.readDataFromFileGeneric(self, attributes)
        self.castCurve('Curve Cf', len0, silentSuccess=True)
        # label based on file name, maybe want to base it on file content
        self.curve(len0).update({'label': self.curve(len0).getAttribute('label').replace('C-f ','').replace(' Cp','').replace(' C','').replace('T=','').replace(' [nF]','').replace(' (nF)','')})
        # retrieve units, change label
        colName = self.curve(len0).getAttribute('frequency [Hz]', -1) # retrieve actual unit
        ylabel = GraphCf.AXISLABELS
        convertToF = 1e-9
        if isinstance(colName, str): # retrieve yaxis information in file, interpret it
            ylabel = colName.replace('C ', 'Capacitance ').replace('Cp ', 'Capacitance ').replace('(','[').replace(')',']')
            expr = '^(.* )\[(.*)\]$'
            f = refindall(expr, ylabel)
            if isinstance(f, list) and len(f) == 1 and len(f[0]) == 2:
                ylabel = [f[0][0], GraphCf.AXISLABELS[1][1], f[0][1]]
            if f[0][1] != 'nF':
                print('CurveCf read, detected capacitance unit', f[0][1], '. Cannot proceed further with phase and Nyquist.')
                convertToF = None
            # TODO: add warning if not nF
        # identify Rp curve
        idxRp = None
        for c in range(len0, self.length()): # retrieve resistance curve
            lbl = self.curve(c).getAttribute('label')
            #print('scan curves label', c, lbl)
            if lbl.endswith('R') or lbl.endswith('Rp [Ohm]'):
                idxRp = c
                break
        # normalize with area C, R
        area = self.curve(len0).getAttribute('cell area (cm2)', None)
        if area is None:
            area = self.curve(len0).getAttribute('cell area', None)
        if area is None:
            area = self.curve(len0).getAttribute('area', None)
        if area is not None:
            self.curve(len0).setY(self.curve(len0).y() / area)
            if idxRp is not None:
                self.curve(idxRp).setY(self.curve(idxRp).y() * area)
            self.curve(len0).update({'cell area (cm2)': area})
            if not self.silent:
                print('Capacitance normalized to area', self.curve(len0).getArea(),'cm2.')
        # generate additional curves if interesting
        nb_add = 0
        # if phase is required
        if self.getAttribute('_CfLoadPhase', False) is not False:
            f = self.curve(len0).x()
            C = self.curve(len0).y() * convertToF # C input assumed to be [nF], need [F] for calculation
            if idxRp is None:
                print('Warning CurveCf read file', self.filename, ': cannot find R. Cannot compute phase.')
            else:
                conductance = 1 / self.curve(idxRp).y()
                phase_angle = np.arctan(f * 2 * np.pi * C / conductance) * 180. / np.pi
                self.append(Curve([f, phase_angle], deepcopy(self.curve(len0).getAttributes())))
                self.curve(-1).update({'_CfPhase': True})
                nb_add += 1
#            conductance = None
#            for c in range(len0+1, self.length()): # retrieve resistance curve, not always same place
#                lbl = self.curve(c).getAttribute('frequency [hz]')
#                if isinstance(lbl, str) and lbl[:1] == 'R':
#                    conductance = 1 / self.curve(c).y()
#                    break
        if self.getAttribute('_CfLoadNyquist', False) is not False:
            if idxRp is None:
                print('ERROR CurveCf read file', self.filename, ': cannot find Rp. Cannot compute Nyquist plot.')
            else:
                C = self.curve(len0).y() * convertToF # C input assumed to be [nF], need [F] for calculation
                omega = self.curve(len0).x() * 2 * np.pi
                Rp = self.curve(idxRp).y()
                Z = 1 / (1 / Rp +  1j * omega * C)
                self.append(Curve([Z.real, -Z.imag], deepcopy(self.curve(len0).getAttributes())))
                self.curve(-1).update({'_CfNyquist': True})
                nb_add += 1
        # delete Rp, temperature, etc
        for c in range(self.length()-1-nb_add, len0, -1):
            self.deleteCurve(c)
        # cosmetics
        self.update({'typeplot': 'semilogx', 'alter': ['', 'idle']})
        self.update({'xlabel': self.formatAxisLabel(GraphCf.AXISLABELS[0]),
                     'ylabel': self.formatAxisLabel(ylabel)}) # default
        if self.curve(len0).getAttribute('cell area (cm2)', None) is not None:
            self.update({'ylabel': self.getAttribute('ylabel').replace('F', 'F cm$^{-2}$')})
        



class CurveCf(Curve):
    
    CURVE = 'Curve Cf'
    
    def __init__(self, data, attributes, silent=False):
        """
        Constructor with minimal structure: Curve.__init__, and set the
        'Curve' parameter.
        """
        Curve.__init__(self, data, attributes, silent=silent)
        self.update ({'Curve': CurveCf.CURVE})


    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
        out.append([self.setArea, 'Set cell area', ['New area'], [self.getArea()]])
        out.append([self.printHelp, 'Help!', [], []])
        return out
    
    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        #out.append(['nm <-> eV', ['nmeV', ''], ''])
        out.append(['semilogx', ['','idle'], 'semilogx'])
        out.append(['-dC / dln(f)', ['', 'CurveCf.y_mdCdlnf'], 'semilogx'])
        out.append(['Frequency [Hz] vs Apparent depth [nm]', ['CurveCV.x_CVdepth_nm', 'x'], 'semilogy'])
        return out

    def y_mdCdlnf(self, index=np.nan, xyValue=None):
        # if ylim is set, keep the ylim restriction and not do not ignore ylim
        if xyValue is not None:
            return xyValue[1]
        # do not want to use self.x(index) syntax as we need more points to 
        # compute the local derivative. HEre we compute over all data, then
        # restrict to the desired datapoint
        val = - derivative(np.log(self.x(xyValue=xyValue)), self.y(xyValue=xyValue))
        if len(val) > 0:
            if np.isnan(index).any():
                return val[:]
            return val[index]
        return val
    
    # FUNCTIONS RELATED TO GUI (fits, etc.)
    def setArea(self, value):
        oldArea = self.getArea()
        self.update({'cell area (cm2)': value})
        self.setY(self.y() / value * oldArea)
        return True
    def getArea(self):
        return self.getAttribute('cell area (cm2)', 1)
        

    # Function related to data picker
    def getDataCustomPickerXY(self, idx, **kwargs):
        """
        Overrides Curve.getDataCustomPickerXY
        Returns Temperature, omega instead of f, C
        """
        if 'strDescription' in kwargs and kwargs['strDescription']:
            return '(Temperature, omega) instead of (f, C)'
        # actually fo things
        try:
            from grapa.datatypes.curveArrhenius import CurveArrheniusCfdefault
            attr = CurveArrheniusCfdefault.attr
        except Exception:
            attr = {'Curve': 'Fit Arrhenius',
                    '_Arrhenius_variant': 'Cfdefault',
                    'label': 'omega vs Temperature'}
            print('WARNING Exception during opening of CurveArrhenius module. Does not perturb saving of the data.')
        attr.update({'sample name': self.getAttribute('sample name')})
        T = np.nan
        for key in ['temperature [k]', 'temperature']:
            T = self.getAttribute(key, np.nan)
            if not np.isnan(T):
                if not is_number(T):
                    T = float(T)
                break
        if not np.isnan(T):
            print('Data picker CurveCf T =',T,', omega =',self.x(idx)[0]*2*np.pi,'.')
            return T, self.x(idx)[0] * 2 * np.pi, attr
        return Curve.getDataCustomPickerXY(self, idx, **kwargs)
        
        
    def printHelp(self):
        print('*** *** ***')
        print('CurveCf offer basic treatment of C-f curves of solar cells.')
        print('Default units are frequency [Hz] and [nF] (or [nF cm-2]).')
        print('Curve transforms:')
        print(' - Linear: standard is Capacitance per area [nF cm-2] vs [Hz].')
        print(' - semilogx: horizontal axis is log(f) instead of f.')
        print(' - -dC / dln(f): derivative of C with ln(f), to identify inflection points.')
        print('Analysis functions:')
        print(' - Set area: can normalize input data. For proper data analysis the units should be [nF cm-2].')
        print('Further analysis:')
        print(' - Report inflection point for different T, then the fit activation energy.')
        print('   Traps can follow omega = 2 ksi T^2 exp(- E_omega / (kT)')
        print('   plot ln(omega T^-2) = ln(2 ksi) - E_omega / (kT)')
        print('References:')
        print('Decock et al., J. Appl. Phys. 110, 063722 (2011); doi: 10.1063/1.3641987')
        print('Walter et al., J. Appl. Phys. 80, 4411 (1996); doi: 10.1063/1.363401')
        return True
