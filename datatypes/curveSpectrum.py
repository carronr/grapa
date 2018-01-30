# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:44:58 2016

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np
from os import path as ospath

from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.curve import Curve
from grapa.mathModule import is_number



class GraphSpectrum(Graph):
    
    FILEIO_GRAPHTYPE = 'Optical spectrum'

    AXISLABELS = [['Wavelength', '\lambda', 'nm'], ['Intensity', '', 'counts']]
    
    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        # HR2000 spectrometer
        if fileExt == '.txt' and line1 == 'SpectraSuite Datei':
           return True
        # TRPL setup PL spectrum
        if fileExt == '.dat' and line1[:29] == 'Excitation Wavelength[nm]	crv':
            return True
        return False
        
    def readDataFromFile(self, attributes, **kwargs):
        """ Read a HR2000 file (optical fiber spectrophotometer). """
        # TODO:  improve header parsing
        filenam_, fileext = ospath.splitext(self.filename) # , fileExt
        if fileext == '.txt':
            self.append(CurveSpectrum(np.transpose(np.genfromtxt(self.filename, skip_header=17, delimiter='\t', invalid_raise=False)), attributes))
            # BG = 1140 # seems not generally valid
            BG = 0
            self.curve(-1).setY(self.curve(-1).y() - BG)
            self.curve(-1).update({'label': filenam_.split('/')[-1]})
            self.headers.update({'collabels': ['Wavelength [nm]', 'Intensity [counts]']})
            self.update({'xlabel': self.formatAxisLabel(GraphSpectrum.AXISLABELS[0]),
                         'ylabel': self.formatAxisLabel(GraphSpectrum.AXISLABELS[1])})
        else:
            len0 = self.length()
            GraphIO.readDataFromFileGeneric(self, attributes)
            self.castCurve('Curve Spectrum', len0, silentSuccess=True)
            self.curve(len0).update({'label': self.curve(len0).getAttribute('label').replace(' crv[0] [Cnts.]','')})
            self.update({'xlabel': self.formatAxisLabel(GraphSpectrum.AXISLABELS[0]),
                         'ylabel': self.formatAxisLabel(GraphSpectrum.AXISLABELS[1])})
            self.update({'subplots_adjust': [0.2, 0.15]})
            

    
    

class CurveSpectrum (Curve):
    """
    Class handling optical spectra, with notably nm to eV conversion and
    background substraction.
    """
    
    CURVE = 'Curve Spectrum'
    
    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        self.update ({'Curve': CurveSpectrum.CURVE})


    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
        out.append([self.dataModifySwapNmEv, 'change data nm<->eV', [], []]) # one line per function
        out.append([self.addOffset, 'Add offset', ['new offset'], [self.getOffset()]]) # one line per function
        iddark = []
        if 'graph' in kwargs:
            for c in range(kwargs['graph'].length()):
                iddark.append(str(c)+' '+kwargs['graph'].curve(c).getAttribute('label')[:6])
        else:
            print('graph must be provided to funcListGUI')
            kwargs['graph'] = None
        out.append([self.substractBG, 'Substract dark',
                    ['id dark', '', '', ''],
                    ['0', '1 interpolate both', '1 new curve', '0 ignore offset'],
                    {'graph': kwargs['graph'], 'operator': 'sub'},
                    [{'field':'Combobox', 'bind':'beforespace', 'width':4, 'values':iddark},
                     {'field':'Combobox', 'width':13,'values':['1 interpolate both', '2 interpolate dark', '0 element-wise']},
                     {'field':'Combobox', 'width':8, 'values':['1 new curve', '0 replace']},
                     {'field':'Combobox', 'width':13,'values':['0 ignore offsets', '1 with offsets']}]])
        out.append([self.printHelp, 'Help!', [], []]) # one line per function
        return out
    
    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        out += [['nm <-> eV', ['nmeV', ''], '']]
        out += [['nm <-> cm-1', ['nmcm-1', ''], '']]
        return out

    
    def getOffset(self):
        return self.getAttribute('_spectrumOffset', 0)
    
    def addOffset(self, value):
        if is_number(value):
            self.setY(self.y() + value - self.getOffset())
            self.update({'_spectrumOffset': value})
            return True
        return False

            
    # more "usual" methods
    def dataModifySwapNmEv(self):
        self.setX(self.NMTOEV / self.x())



    def substractBG(self, idDark, interpolate, ifnew, offsets, graph=None, **kwargs):
        """ Substract background """
        # clean input from GUI: interpolate
        if interpolate in [0,1,2, '0','1','2']:
            interpolate = int(interpolate)
        elif isinstance(interpolate, str) and len(interpolate) > 0 and interpolate[0] in ['0','1','2']:
            interpolate = int(interpolate[0])
        else:
            interpolate = 1
        # clean input from GUI: ifNew
        def cleanInputBool(in_, default=False):
            if in_ in [0, 1, True, False, '0', '1', 'True', 'False']:
                if in_ in ['0', 'False']:
                    in_ = False
                else:
                    in_ = bool(in_)
            elif isinstance(in_, str) and len(in_) > 0 and in_[0] in ['0', '1']:
                in_ = True if in_[0] == '1' else False
            else:
                in_ = default
            return in_
        # clean input from GUI: ifNew, offsets
        ifnew = cleanInputBool(ifnew, default=True)
        offsets = cleanInputBool(offsets, default=False)
        idDark = min(int(idDark), graph.length()-1)
        j = None # index of 'curve' in graph
        for c in range(graph.length()):
            if graph.curve(c) == self:
                j = c
                break
        out = self.__add__(graph.curve(idDark), interpolate=interpolate, offsets=offsets, **kwargs)
        key = '_sub'
        msg = ('{Curve ' + (str(j) if j is not None else '') +
               ': ' + self.getAttribute('label') + '} - ' +
               '{Curve ' + str(idDark) + ': ' + graph.curve(idDark).getAttribute('label') + '}')
        out.update({key: msg})
        if not ifnew:
            if j is not None:
                graph.replaceCurve(out, j)
                print('CurveSpectrum.substractBG: modified Curve data.')
                return True
            print('CurveSpectrum.substractBG: cannot identify Curve!?! Created a new one instead.')
        print('CurveSpectrum.substractBG: created new Curve.')
        return out

    
    def printHelp(self):
        print('*** *** ***')
        print('CurveSpectrum offers some capabilities to process optical spectrum data.')
        print('Curve transforms:')
        print(' - nm <-> eV: switch [eV] or [nm] data into the other representation (eV =', self.NMTOEV,'/ nm).')
        print('Analysis functions')
        print(' - Change data nm<->eV: changes data from nm to eV or inversely (eV =', self.NMTOEV,'/ nm),')
        print(' - Add offset: adds a vertical offset to the data. The data are modified.')
        print('   The cumulated data correction is displayed, such that setting it to 0')
        print('   retrieves the original data (with some rounding errors)')
        print(' - Substract dark: substract another curve to the data.')
        print('   Parameters:')
        print('   id dark: index of of the Curve containing the dark spectrum.')
        print('   interpolate: 0: performs element-wise substraction,')
        print('      1: output on each data and dark Curves datapoints, interpolates both Curves,')
        print('      2: output on selected Curve datapoint, interpolates the dark Curve.')
        print('   new Curve: 1: creates a new Curve, 0: modifiy existing data.')
        print('   offsets: 0: ignore offset and muloffset information.')
        print('      1: substract data after offset and muloffset operation.')
        return True
