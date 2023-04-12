# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2023, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

import numpy as np
from copy import deepcopy

from grapa.curve import Curve
from grapa.mathModule import is_number, roundSignificant, roundSignificantRange


class CurveTRPL(Curve):
    """ Class handling TRPL decays. """

    CURVE = 'Curve TRPL'
    SMOOTH_WINDOW = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        self.update({'Curve': self.CURVE})
        # retrieve parameters required for normalization
        if self.attr('_repetfreq_Hz', None) is None:
            try:
                rep = float(self.attr('sync_frequency', None).split(' ')[0])
            except Exception:
                rep = 1
            self.update({'_repetfreq_Hz': rep})
        if self.attr('_acquistime_s', None) is None:
            try:
                time0 = float(self.attr('meas_time', None).split(' ')[0])
                time1 = float(self.attr('meas_stop', None).split(' ')[0])
                time = time1 - time0
                if time == 0 or np.isnan(time):
                    time = 1
            except Exception:
                time = 1
            self.update({'_acquistime_s': time})
        if self.attr('_binwidth_s', None) is None:
            try:
                binw = float(self.attr('meas_binwidth', None).split(' ')[0]) * 1e-12
            except Exception:
                binw = 1
            self.update({'_binwidth_s': binw})
        # backward compatibility
        if (self.attr('_spectrumOffset', None) is not None
                and self.attr('_TRPLOffset', None) is None):
            self.update({'_TRPLOffset': self.attr('_spectrumOffset')})

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
        xOffset = self.getXOffset()
        out.append([self.addXOffset, 'Time offset',
                    ['new horizontal offset (leave empty for autodetect)'],
                    [xOffset if xOffset != 0 else '']])  # 1 line per function
        unit = self.attr('_unit', 'cts')
        out.append([self.addOffset, 'Add offset',
                    ['new vertical offset ('+unit+')'], [self.getOffset()]])
        # normalization
        unit = self.attr('_unit', None)
        revert = False if unit is None else True
        if not revert:
            out.append([self.normalize, 'Normalize intensity',
                        ['pulse freq Hz', 'acquis time s', 'bin ps'],
                        [self.attr('_repetfreq_Hz', 1), self.attr('_acquistime_s', 1), self.attr('_binwidth_s', 1)*1e12],
                        {},
                        [{'width': 10}, {'width': 7}, {'width': 6}]])
        else:  # get back cts data
            out.append([self.normalizerevert, 'Restore intensity cts',
                        ['Current intensity unit: '+str(unit)+'. factor'],
                        [self.getFactor()], {}, [{'field': 'Label'}]])
        # fit
        if self.attr('_fitFunc', None) is None or self.attr('_popt', None) is None:
            ROI = roundSignificantRange([20, max(self.x())], 2)
            out.append([self.CurveTRPL_fitExp,
                        'Fit exp',
                        ['Nb exp', 'ROI', 'Fixed values', 'show residuals'],
                        [2, ROI, [0, '', ''], False],
                        {},
                        [{'field': 'Combobox', 'values': ['1', '2', '3']},
                         {}, {}, {'field': 'Combobox', 'values': ['False', 'True']}]])
        else:
            values = roundSignificant(self.attr('_popt'), 5)
            params = ['BG']
            fieldred = {'width': 5} if len(values)-1 > 4 else {}
            fields = [fieldred]
            while len(values) > len(params)+1:
                n = '{:1.0f}'.format((len(params)+1)/2)
                params += ['A'+n, "\u03C4"+n]
                fields += [fieldred, fieldred]
            out.append([self.updateFitParam, 'Update fit', params, values, {},
                        fields])
        out.append([self.CurveTRPL_smoothBin, 'Smooth & bin',
                    ['width', 'convolution', 'binning'],
                    ['9', 'hanning', '1'],
                    {},
                    [{}, {'field': 'Combobox', 'values': self.SMOOTH_WINDOW},
                        {'field': 'Combobox', 'values': ['1', '2', '4', '8', '16']}]])
        # integration
        alter = str(kwargs['graph'].attr('alter')) if 'graph' in kwargs else "['', '']"
        ROI = roundSignificantRange([min(self.x()), max(self.x())], 2)
        out.append([self.integrate, 'Integrate',
                    ['ROI', 'data transform'], [ROI, alter], {},
                    [{}, {'field': 'Combobox', 'values': ['raw', alter]}]])
        # help
        out.append([self.printHelp, 'Help!', [], []])
        return out

    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        out += [['semilogy', ['', 'idle'], 'semilogy']]
        return out

    # handling of offsets - same as Curve Spectrum, with same keyword
    def getOffset(self):
        return self.attr('_TRPLOffset', 0)

    def getFactor(self):
        return self.attr('_TRPLFactor', 1)

    def setIntensity(self, offsetnew=None, factornew=None):
        """ Data stored as (raw + offset) * factor """
        """
        cts = self.y() / self.getFactor() - self.getOffset()
        if is_number(offsetnew):
            self.update({'_spectrumOffset': offsetnew})
        if is_number(factornew):
            self.update({'_spectrumFactor': factornew})
        self.setY((cts + self.getOffset()) * self.getFactor())
        """
        factorold = self.getFactor()
        cts = (self.y() - self.getOffset()) / factorold
        if is_number(offsetnew):
            self.update({'_TRPLOffset': offsetnew})
        if is_number(factornew):
            self.update({'_TRPLFactor': factornew})
            self.update({'_TRPLOffset': self.getOffset() * factornew / factorold})
        self.setY((cts * self.getFactor()) + self.getOffset())
        # backward compatibility
        if self.attr('_spectrumOffset', None) is not None:
            self.update({'_spectrumOffset': self.attr('_TRPLOffset')})
        return True

    def addOffset(self, value):
        if is_number(value):
            self.setIntensity(offsetnew=value)
            return True
        return False

    def normalize(self, repetfreq_Hz, duration_s, binwidth_ps):
        """
        Normalizes intensity of TRPL to account for repetition rate,
        acquisition duration and binwidth. Assumes data is in unit 'counts'.
        Data: (raw+offset) * (1 / (syncfreq * (measstop-meastime) * binwidth))
        """
        try:
            factor = float(1 / (repetfreq_Hz * duration_s * (1e-12*binwidth_ps)))
        except Exception:
            return False
        if factor == 0 or np.isinf(factor):
            print('CurveTRPL.normlize: non-sensical normalization factor (0',
                  'or inf).')
            return False
        # should not happen if using only the GUI
        if self.attr('_unit', None) is not None:
            print('CurveTRPL.normalize: data may have been already normalized',
                  '(Curve labelled as "'+self.attr('_unit')+'","',
                  self.attr('_unitfactor')+'").')
        self.setIntensity(factornew=factor)
        self.update({'_unit': 'cts/Hz/s/s'})
        # overwrite acquisition parameters, if significant deviation from
        # Actually, not. We keep default parameters. User need to change them
        # manually if he wishes
        # try:
        #     if np.abs(self.attr('_repetfreq_Hz', 1) - repetfreq_Hz) / repetfreq_Hz > 1e-6:
        #         self.update({'_repetfreq_Hz': repetfreq_Hz})
        #     if np.abs(self.attr('_acquistime_s') - duration_s) / duration_s > 1e-6:
        #         self.update({'_acquistime_s': duration_s})
        #     if np.abs(self.attr('_binwidth_s') - (1e-12*binwidth_ps)) / (1e-12*binwidth_ps) > 1e-6:
        #         self.update({'_binwidth_s': (1e-12*binwidth_ps)})
        # except Exception:
        #     pass
        return True

    def normalizerevert(self, *args):
        self.setIntensity(factornew=1)
        self.update({'_unit': ''})
        return True

    # temporal offset
    def getXOffset(self):
        return self.attr('_TRPLxOffset', 0)

    def addXOffset(self, value):
        if is_number(value):
            self.setX(self.x() + value - self.getXOffset())
            self.update({'_TRPLxOffset': value})
            return True
        else:
            return self.addXOffset(- self.findOnset())
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
        Fit exp: fits the data as a constant plus a sum of exponentials.
        Returns a Curve as the best fit to the TRPL decay.
        Formula: y(t) = BG + A1 exp(-t/tau1) + A2 exp(-t/tau2) + ...
        thus y(0) = BG + A1 + A2 + ...
        Parameters:
        - nbExp: The number of exponentials in the fit
        - ROI: minimum and maximum on horizontal axis, e.g. [min, max]
        - fixed: an array of values. When a number is set, the corresponding
          fit parameter is fixed. The order of fit parametersis as follows:
          BG, A1, tau1, A2, tau2, .... e.g. [0,'','']
        - showResiduals: when 1, also returns a Curve as the fit residuals.
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
                'filename': 'fit to '+self.attr('filename').split('/')[-1].split('\\')[-1],
                'label': 'fit to '+self.attr('label')}
        attr.update(self.getAttributes(['offset', 'muloffset', '_repetfreq_Hz',
                                        '_acquistime_s', '_binwidth_s']))
        mask = self.ROItoMask([0, max(self.x())])
        fitted = CurveTRPL([self.x(mask), self.func_fitExp(self.x(mask), *popt)], attr)
        if showResiduals:
            mask = self.ROItoMask(ROI)
            attrresid = {'color': [0.5, 0.5, 0.5], 'label': 'Residuals'}
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
        # fixed: has the desired length, to be done by calling function
        # fixed: e.g. [0, '', 10]
        # check for ROI
        mask = self.ROItoMask(ROI)
        datax = self.x()[mask]
        datay = self.y()[mask]
        # check for fixed params, construct p0
        p0default = [0, np.max(datay) * 0.5, np.abs(np.max(datax)) * 0.02]
        while len(p0default) < len(fixed):  # additional: tau*2, A/2
            p0default += [p0default[-2] / 4, p0default[-1] * 4]
        # fixed parameters: input value in initial guess, before recondition
        isFixed = []  # list of booleans
        for i in range(len(fixed)):
            isFixed.append(is_number(fixed[i]))
            if isFixed[i]:
                p0default[i] = fixed[i]
        print('p0default', p0default)

        # p0default = [(p if p % 2 or p == 0 else np.log(p)) for p in p0default]

        def recondition(params):
            # all fitfunc parameters, including fixed variables
            p_ = params[0:3]
            for i in range(3, len(params), 2):
                p_.append(params[i])  # A value
                p_.append(params[i-1] / params[i+1])  # tau value
            return p_

        def decondition(params, ifFixed, fixed):
            # all fitfunc parameters, including fixed variables
            p_ = params[0:3]
            for i in range(3, len(params), 2):
                p_.append(params[i])  # A value
                p_.append(params[i-1] / params[i+1])  # tau value
                if isFixed[i+1]:
                    p_[i+1] = fixed[i+1]  # actual fixed value needed to decondition next variable values
            return p_

        def mergedvarfixed(params, isFixed, complementary):
            p = []
            j, k = 0, 0
            for i in range(len(isFixed)):
                if isFixed[i]:
                    p.append(complementary[k])
                    k += 1
                else:
                    p.append(params[j])
                    j += 1
            return p

        # recondition problem: tau are written differently
        p0default = recondition(p0default)
        # handle fixed and actual variables
        p0 = []
        isFixed = []  # list of booleans
        complementary = []
        for i in range(len(fixed)):
            isFixed.append(is_number(fixed[i]))
            if isFixed[i]:
                # fixed[i] may not == p0default[i] due to recondition
                complementary.append(fixed[i])
            else:
                p0.append(p0default[i])

        # custom fit function handling fixed and free fit parameters
        def func(datax, *p0):
            params = mergedvarfixed(p0, isFixed, complementary)
            print('func a', params)
            # during decondition, need to fix again fixed values, e.g. tau1 changed by fit algorithm but tau2 was fixed by used
            params = decondition(params, isFixed, fixed)
            print('     b', params)
            return self.func_fitExp(datax, *params)

        # actual fitting
        print('TRPL p0', p0)
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(func, datax, datay, p0=p0)

        # construct output parameters including fixed ones
        params = mergedvarfixed(popt, isFixed, complementary)
        # back to standard parametrization
        params = decondition(params, isFixed, fixed)

        # sort output by ascending tau values
        if np.sum(isFixed[1:]) == 0:  # only if no fixed parameter
            pairs = [[params[i], params[i+1]] for i in range(1, len(params), 2)]
            taus = [pair[1] for pair in pairs]
            pairs = [x for _, x in sorted(zip(taus, pairs))]
            p_ = [params[0]]
            for pair in pairs:
                p_ += pair
            params = p_

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
        """
        Smooth & bin: returns a copy of the Curve after smoothening and data
        binning. Parameters:
        - width (window_len): number of points in the smooth window,
        - convolution (window): the type of window. Possible values: 'hanning',
          'hamming', 'bartlett', 'blackman', or 'flat' (moving average).
        - binning: how many points are merged.
        """
        if not is_number(window_len) or window_len < 1:
            print('Warning CurveTRPL smoothBin: cannot interpret window_len',
                  'value (got', window_len, ', request int larger than 0.)',
                  'Set 1.')
            window_len = 1
        window_len = int(window_len)
        if not is_number(binning) or binning < 1:
            print('Warning CurveTRPL smoothBin: cannot interpret binning',
                  'value (got', binning, ', request int larger than 0.)',
                  'Set 1.')
            binning = 1
        binning = int(binning)
        from mathModule import smooth
        smt = smooth(self.y(), window_len, window)
        x = self.x()
        le = len(x)
        x_ = np.zeros(int(np.ceil(len(x) / binning)))
        y_ = np.zeros(int(np.ceil(len(x) / binning)))
        for i in range(len(x_)):
            x_[i] = np.average(  x[i*binning:min(le, (i+1)*binning)])
            y_[i] = np.average(smt[i*binning:min(le, (i+1)*binning)])
        attr = deepcopy(self.getAttributes())
        comment = ('Smoothed curve (' + str(self.attr('label')) + ') smt '
                   + str(window_len) + ' ' + str(window) + ' bin '
                   + str(binning))
        attr.update({'comment': comment})
        attr.update({'label': str(self.attr('label'))+' '+'smooth'})
        return CurveTRPL([x_, y_], attr)

    def integrate(self, ROI=None, alter=None, curve=None):
        """
        Integrate: returns the integral of the curve, within ROI. Parameters:
        - ROI: example [xmin, xmax]
        - data transform (alter): 'raw', or any Graph 'alter' value including
          (mul-)offsets.
        """
        # curve and not self: tweak to be able to integrate a Curve not
        # CurveTRPL (eg from GUI)
        if curve is None:
            curve = self
        mask = CurveTRPL.ROItoMask(curve, ROI)
        if alter is not None and alter not in ['raw']:
            if isinstance(alter, str):
                alter = ['', alter]
            datax = curve.x_offsets(alter=alter[0])[mask]
            datay = curve.y_offsets(alter=alter[1])[mask]
        else:
            datax = curve.x()[mask]
            datay = curve.y()[mask]
        integral = np.trapz(datay, datax)
        return integral

    def printHelp(self):
        print('*** *** ***')
        print('CurveTRPL offers some support to fit time-resolved',
              'photoluminence (TRPL) spectra.')
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
        self.printHelpFunc(self.normalize)
        self.printHelpFunc(self.CurveTRPL_fitExp)
        self.printHelpFunc(self.CurveTRPL_smoothBin)
        self.printHelpFunc(self.integrate)
        return True