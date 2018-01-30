# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 18:08:32 2017

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np
from os import path as ospath
from copy import deepcopy

from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.curve import Curve
from grapa.mathModule import is_number, roundSignificant



class GraphTRPL(Graph):
    
    FILEIO_GRAPHTYPE = 'TRPL decay'
    
    AXISLABELS = [['Time', 't', 'time'], ['Intensity', '', 'counts']]

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        if fileExt == '.dat' and line1 == 'Time[ns]	crv[0] [Cnts.]':
            return True
        elif (fileExt == '.dat' and line1 == 'Parameters:' and
              line2.strip().startswith('Sample') and
              line3.strip().startswith('Solvent')) :
            return True
        return False
        
    def readDataFromFile(self, attributes, **kwargs):
        """ Read a TRPL decay. """
        len0 = self.length()
        kw = {}
        if 'line1' in kwargs and kwargs['line1'] == 'Parameters:':
            kw.update({'delimiterHeaders': ':'})
        GraphIO.readDataFromFileGeneric(self, attributes, **kw)
        self.castCurve(CurveTRPL.CURVE, len0, silentSuccess=True)
        filenam_, fileext = ospath.splitext(self.filename) # , fileExt
        self.curve(len0).update({'label': filenam_.split('/')[-1].split('\\')[-1]})
        xlabel = self.getAttribute('xlabel').replace('[',' [').replace('  ',' ').capitalize() # ] ]
        if xlabel in ['', ' ']:
            xlabel = GraphTRPL.AXISLABELS[0]
        self.update({'typeplot': 'semilogy', 'alter': ['', 'idle'],
                     'xlabel': self.formatAxisLabel(xlabel),
                     'ylabel': self.formatAxisLabel(GraphTRPL.AXISLABELS[1])})
        self.update({'subplots_adjust': [0.2, 0.15]})
        # cleaning
        if 'line1' in kwargs and kwargs['line1'] == 'Parameters:':
            attr = self.curve(len0).getAttributes()
            keys = list(attr.keys())
            for key in keys:
                val = attr[key]
                if isinstance(val, str) and val.startswith('\t'):
                    self.curve(len0).update({key:''})


    
    

class CurveTRPL(Curve):
    """ Class handling TRPL decays. """

    CURVE = 'Curve TRPL'
    SMOOTH_WINDOW = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    
    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        self.update ({'Curve': self.CURVE})


    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
        out.append([self.addOffset, 'Add offset', ['new vertical offset'], [self.getOffset()]]) # one line per function
        xOffset = self.getXOffset()
        out.append([self.addXOffset, 'Time offset', ['new horizontal offset (leave empty for autodetect)'], [xOffset if xOffset != 0 else '']]) # one line per function
        # fit
        if self.getAttribute('_fitFunc', None) is None or self.getAttribute('_popt', None) is None:
            ROI = [20, max(self.x())]
            out.append([self.CurveTRPL_fitExp,
                        'Fit exp',
                        ['Nb exp', 'ROI', 'Fixed values', 'show residuals'],
                        [2, ROI, [0, '', ''], False],
                        {},
                        [{'field':'Combobox', 'values':['1','2','3']},
                         {},{},{'field':'Combobox', 'values':['False', 'True']}]])
        else:
            values = roundSignificant(self.getAttribute('_popt'), 5)
            params = ['BG']
            while len(values) > len(params)+1:
                n = '{:1.0f}'.format((len(params)+1)/2)
                params += ['A'+n, 'tau'+n]
            out.append([self.updateFitParam, 'Update fit', params, values])
        out.append([self.CurveTRPL_smoothBin, 'Smooth & bin',
                    ['width', 'convolution', 'binning'],
                    ['9', 'hanning', '1'],
                    {},
                    [{},{'field':'Combobox', 'values':self.SMOOTH_WINDOW},{'field':'Combobox','values':['1','2','4','8','16']}]])
        out.append([self.printHelp, 'Help!', [], []]) # one line per function
        return out
    
    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        out += [['semilogy', ['', 'idle'], 'semilogy']]
        return out

    
    # handling of offsets - same as Curve Spectrum, with same keyword
    def getOffset(self):
        return self.getAttribute('_spectrumOffset', 0)
    def addOffset(self, value):
        if is_number(value):
            self.setY(self.y() + value - self.getOffset())
            self.update({'_spectrumOffset': value})
            return True
        return False
    
        # temporal offset
    def getXOffset(self):
        return self.getAttribute('_TRPLxOffset', 0)
    def addXOffset(self, value):
        if is_number(value):
            self.setX(self.x() + value - self.getXOffset())
            self.update({'_TRPLxOffset': value})
            return True
        else:
            return self.addXOffset( - self.findOnset())
        return False
    def findOnset(self):
        """ define onset as average between 25% and 95% percentile """
        y = self.y()
        v25 = np.percentile(y, 25)
        v95 = np.percentile(y, 95)
        target = (v25 + v95) / 2
        iMax = np.argmax(y)
        for i in range(iMax, -1, -1):
            if y[i] < target:
                return self.x(i)
        return self.x(0)
        

        
    def CurveTRPL_fitExp(self, nbExp=2, ROI=None, fixed=None, showResiduals=False, silent=False):
        """
        Returns a Curve based on a fit on TRPL decay.
        """
        # set nb of exponentials, adjust variable "fixed" accordingly
        while len(fixed) > 1 + 2 * nbExp:
            del fixed[-1]
        missing = int(1 + 2 * nbExp - len(fixed))
        if missing > 0:
            fixed += [''] * missing
        # clean input shworesiduals
        if isinstance(showResiduals, str):
            if showResiduals in ['True', '1']:
                showResiduals = True
            else:
                showResiduals = False
        popt = self.fit_fitExp(ROI=ROI, fixed=fixed)
        attr = {'color': 'k', '_ROI': ROI,
                '_popt': popt, '_fitFunc': 'func_fitExp',
                'filename': 'fit to '+self.getAttribute('filename').split('/')[-1].split('\\')[-1],
                'label': 'fit to '+self.getAttribute('label')}
        attr.update(self.getAttributes(['offset', 'muloffset']))
        mask = self.ROItoMask([0, max(self.x())])
        fitted = CurveTRPL([self.x(mask), self.func_fitExp(self.x(mask), *popt)], attr)
        if showResiduals:
            mask = self.ROItoMask(ROI)
            attrresid = {'color': [0.5,0.5,0.5], 'label': 'Residuals'}
            resid = CurveTRPL([self.x(mask), self.func_fitExp(self.x(mask), *popt)-self.y(mask)], attrresid)
            fitted = [fitted, resid]
        if not silent:
            taus = []
            for i in range(2, len(popt), 2):
                taus += [str(popt[i])]
            taus = ', '.join(taus)
            print('Fitted with', int(nbExp), 'exponentials: tau', taus, '.')
            print('Params\t', '\t'.join([str(p) for p in popt]))
        return fitted
    
    def fit_fitExp(self, ROI=None, fixed=None):
        # check for ROI
        mask = self.ROItoMask(ROI)
        datax = self.x()[mask]
        datay = self.y()[mask]
        # check for fixed params, construct p0
        p0default = [0, 1e4, 30, 1000, 100]
        while len(p0default) < len(fixed):
            p0default += [p0default[-2] / 2, p0default[-1]*2]
        p0 = []
        complementary = []
        isFixed = []
        for i in range(len(fixed)):
            isFixed.append(is_number(fixed[i]))
            if isFixed[i]:
                complementary.append(fixed[i])
            else:
                p0.append(p0default[i])
        # custom fit function handling fixed and free fit parameters 
        def func(datax, *p0):
            j = 0
            params = []
            for i in range(len(isFixed)):
                if isFixed[i]:
                    params.append(complementary[i])
                else:
                    params.append(p0[j])
                    j += 1
            return self.func_fitExp(datax, *params)
        # actual fitting
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(func, datax, datay, p0=p0)
        # construct output parameters including fixed ones
        j = 0
        params = []
        for i in range(len(isFixed)):
            if isFixed[i]:
                params.append(complementary[i])
            else:
                params.append(popt[j])
                j += 1
        # sort exp by ascending tau: TODO
        return params
        
    def func_fitExp(self, t, BG, A1, tau1, *args):
        """
        computes the sum of a cst plus an arbitrary number of exponentials
        """
        out = BG + A1 * np.exp(- t / tau1)
        i = 0
        while len(args) > i+1:
            out += args[i] * np.exp(- t / args[i+1])
            i += 2
        return out
        

    def ROItoMask(self, ROI=None):
        x = self.x()
        if ROI is None:
            ROI = [min(x), max(x)]
        mask = np.ones(len(x), dtype=bool)
        for i in range(len(mask)):
            if x[i] < ROI[0] or x[i] > ROI[1]:
                mask[i] = False
        return mask
        
    
    def CurveTRPL_smoothBin(self, window_len=9, window='hanning', binning=4):
        if not is_number(window_len) or window_len < 1:
            print('Warning CurveTRPL smoothBin: cannot interpret window_len value (got',window_len,', request int larger than 0.) Set 1.')
            window_len = 1
        window_len = int(window_len)
        if not is_number(binning) or binning < 1:
            print('Warning CurveTRPL smoothBin: cannot interpret binning value (got',binning,', request int larger than 0.) Set 1.')
            binning = 1
        binning = int(binning)
        from mathModule import smooth
        smt = smooth(self.y(), window_len, window)
        x = self.x()
        le = len(x)
        x_ = np.zeros(int(np.ceil(len(x) / binning)))
        y_ = np.zeros(int(np.ceil(len(x) / binning)))
        for i in range(len(x_)):
            x_[i] = np.average(  x[i*binning : min(le, (i+1)*binning)])
            y_[i] = np.average(smt[i*binning : min(le, (i+1)*binning)])
        attr = deepcopy(self.getAttributes())
        attr.update({'comment':'Smoothed curve ('+str(self.getAttribute('label'))+
                     ') smt '+str(window_len)+' '+str(window)+ ' bin '+str(binning)})
        attr.update({'label': str(self.getAttribute('label'))+' '+'smooth'})
        return CurveTRPL([x_, y_], attr)
        
    

    def printHelp(self):
        print('*** *** ***')
        print('CurveTRPL offers some support to fit time-resolved photoluminence (TRPL) spectra.')
        print('The associated functions are:')
        print(' - Add offset: can adjust the background level. Data are\n',
              '   modified. The previous adjustment is shown, and the original\n',
              '   data can be re-computed by setting it to 0.')
        print(' - Time offset: can add a temporal offset, in order to set \n',
              '   peak onset at t=0. This is especially useful as the fit\n',
              '   method considers the decay starts at t=0.\n',
              '   If value is empty, the software tries to autodetect the\n',
              '   leading edge, as the last point below a threshold defined as\n',
              '   the average of the 25% and 95% percentiles.')
        print(' - Fit exp: offers possibility of fitting with the sum of a\n',
              '   constant, plus an arbitrary number of exponential functions.\n',
              '   Formula: y(t) = BG + A1 exp(-t/tau1) + A2 exp(-t/tau2) + ...\n',
              '   thus y(0) = BG + A1 + A2 + ...\n',
              '   Options:,\n',
              '   The number of exponentials can be selected,\n',
              '   ROI, with minimum and maximum on horizontal axis,\n',
              '   Fixed: an array of values. When a number is set the\n',
              '      corresponding fit parameter is fixed. The fit parameters\n',
              '      go as follows: BG, A1, tau1, A2, tau2, ...\n',
              '   Residuals: when 1, also returns the fit residuals.')
        print(' - Smooth & bin: smooth the Curve, and then bin the data.\n',
              '   Options:\n',
              '   width: number of points in the smooth window,\n',
              '   convolution: the type of window. Possible values are \'hanning\',\n',
              '      \'hamming\', \'bartlett\', \'blackman\', or \'flat\' (moving average).\n',
              '   binning: how many points are merged.')
        return True
