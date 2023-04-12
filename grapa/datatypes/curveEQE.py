# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:46:13 2016

@author: Romain Carron
Copyright (c) 2023, Empa, Laboratory for Thin Films and Photovoltaics,
Romain Carron
"""

import numpy as np
import os
from scipy import interpolate
from scipy.signal import medfilt

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.mathModule import roundSignificant, roundSignificantRange
from grapa.gui.GUIFuncGUI import FuncGUI

class CurveEQE(Curve):
    """
    CurveEQE offer basic treatment of (external) quantum efficiency curves of
    solar cells.
    Input units should be in [nm] and values within [0-1]. If needed,
    CurveSpectrum can convert [eV] in [nm].
    """

    CURVE = 'Curve EQE'

    EQE_AM15_REFERENCE = None
    EQE_AM15_REFERENCE_FILE = 'AM1-5_Ed2-2008.txt'
    EQE_AM0_REFERENCE = None
    EQE_AM0_REFERENCE_FILE = 'AM0_2000_ASTM_E-490-00.txt'

    EQE_REF_SPECTRA = None
    EQE_REF_SPECTRA_FILE = 'EQE_referenceSpectra.txt'

    def __init__(self, data, attributes, silent=False):
        # modify default label
        if 'label' in attributes:
            if not isinstance(attributes['label'], str):
                attributes['label'] = str(attributes['label'])
            if '$' not in attributes['label']:
                attributes['label'] = attributes['label'].replace('_', ' ')
        if not isinstance(data, (np.ndarray)):
            data = np.array(data)
        # detect possible empty lines at beginnning (some versions of EQE files
        # ahave different number of header lines)
        flag = True if len(data.shape) > 1 else False  # check if data loaded
        try:
            while flag:
                if data[0, 0] == 0:
                    data = data[:, 1:]
                else:
                    flag = False
        except IndexError:
            pass
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        # bandgap calculation
        popt = self.attr('_popt')
        if isinstance(popt, str) and popt == '':
            bandgap = [np.nan, np.nan]
            try:
                bandgap = self.bandgapFromTauc(self.x(), self.y())
            except Exception:
                if not self.silent:
                    print('ERROR readDataFromFileEQE during Eg calculation',
                          self.filename, self.data[-1].data())
            # also useful to store information that we were *not* able to
            # compute bandgap
            self.update({'bandgap': bandgap[0], 'bandgapslope': bandgap[1]})
        # for saved files further re-reading
        self.update({'Curve': CurveEQE.CURVE})

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
        mulOffset = self.attr('mulOffset', default=1)
        if isinstance(mulOffset, list):
            mulOffset = mulOffset[1]
        try:
            mulOffset = float(mulOffset)
        except TypeError:
            mulOffset = 1

        # tauc (E*EQE)**2
        if (not self.attributeEqual('_popt')
                and self.attributeEqual('_fitfunc', 'func_bandgapTaucCurve')):
            out.append([self.updateFitParam, 'Update fit', ['Eg', 'slope'],
                        roundSignificant(self.attr('_popt'), 5)])
        elif self.attributeEqual('_popt'):
            out.append([self.CurveEQE_bandgap_print,
                        'Bandgap fit Tauc (eV*EQE)^2', ['Range EQE'],
                       ['['+str(0.25*mulOffset)+', '+str(0.70*mulOffset)+']']])

        # Eg from derivative method
        if (not self.attributeEqual('_popt')
                and self.attributeEqual('_fitfunc', 'bandgapDeriv_savgol')):
            out.append([None, 'Cannot update, must create a new one from '
                        + 'initial EQE curve', [], []])
        elif self.attributeEqual('_popt'):
            xdata = self.x()
            try:
                tmpmed = medfilt(self.y(), 5)
                ROI = [xdata[np.argmax(tmpmed)], np.max(xdata)]
            except Exception:
                if len(xdata) > 0:
                    ROI = [np.min(xdata), np.max(xdata)]
                else:
                    ROI = [0, 0]
            out.append([self.CurveEQE_bandgapDeriv_print, 'Bandgap derivative',
                        ['Savitsky Golay width', 'degree', 'nm range'],
                        [5, 2, roundSignificantRange(ROI, 3)]])

        # tauc (E*ln(1-EQE))**2
        if (not self.attributeEqual('_popt')
                and self.attributeEqual('_fitfunc', 'func_bandgapTaucCurveLog')):
            out.append([self.updateFitParam, 'Update fit', ['Eg', 'slope'],
                        roundSignificant(self.attr('_popt'), 5)])
        elif self.attributeEqual('_popt'):
            mulOffset_suf = ' [%]' if mulOffset == 100 else ''
            out.append([self.CurveEQE_bandgapLog_print,
                        'Bandgap fit Tauc (eV*ln(1-EQE))^2',
                        ['Range EQE'+mulOffset_suf],
                        ['['+str(0.30*mulOffset)+', '+str(0.88*mulOffset)+']']])
        
        # EQE current
        if self.attributeEqual('_popt'):
            fnames = [self.EQE_AM15_REFERENCE_FILE.replace('.txt', ''),
                      self.EQE_AM0_REFERENCE_FILE.replace('.txt', '')]
            if len(self.x()) > 0:
                mM = [min(self.x()), max(self.x())]
            else:
                mM = [0, 0]
            line = FuncGUI(self.currentCalc, 'EQE current', hiddenvars={'ifGUI': True})
            line.append('ROI [nm]', mM, options={'width': 15})
            line.append('interpolation', 'linear',
                        options={'width': 8, 'field': 'Combobox', 'values': ['linear', 'quadratic', 'cubic']})
            line.append('', '', widgetclass='Frame')
            line.append('          spectrum', fnames[0], options={'width': 15, 'field': 'Combobox', 'values': fnames})
            line.append('integrated current as curve', False, options={'field': 'Checkbutton'})
            out.append(line)

        # reference spectra - 20.4%, 20.8%, etc.
        refs = self._referenceEQESpectra()
        lbls = [curve.attr('label') for curve in refs]
        out.append([self.referenceEQESpectrum, 'Add', ['reference spectrum'],
                   [lbls[0]], {},
                   [{'field': 'Combobox', 'width': 25, 'values': lbls}]])

        # Urbach fitting
        if self.attributeEqual('_popt'):
            try:  # Urbach fitting
                from grapa.datatypes.curveArrhenius import CurveArrheniusExpDecay
                try:
                    x0 = 1 + np.where(list(reversed(self.y() > 0.1)))[0][0]
                except IndexError:
                    x0 = 1
                try:
                    x1 = 1 + np.where(list(reversed(self.y() > 8e-4)))[0][0]
                except IndexError:
                    x1 = len(self.x())
                x0, x1 = self.shape()[1] - x0, self.shape()[1] - x1
                if len(self.x()) > 0:
                    ROI = [self.x(index=x0, alter='nmeV'),
                           self.x(index=x1, alter='nmeV')]
                else:
                    ROI = [0, 0]
                ROI.sort()
                ROI = roundSignificantRange(ROI, 3)
                mulOffset = self.attr('mulOffset', 1)
                out.append([self.CurveUrbachEnergy,
                            CurveArrheniusExpDecay.BUTTONFitLabel(),
                            [CurveArrheniusExpDecay.BUTTONFitROI()], [ROI]])
            except ImportError as e:
                print('WARNING CurveEQE: do not find',
                      'grapa.datatypes.curveArrhenius.')
                print(type(e), e)

        # ERE external radiative efficiency
        if self.attributeEqual('_popt'):
            Emin, unit = self.ERE_EminAuto()
            out.append([self.ERE_GUI, 'ERE external radiative eff.',
                        ['cell Voc', 'T', 'cut data', ''],
                        ['0.700', 273.15+25, Emin, unit],
                        {},
                        [{'width': 6}, {'width': 7}, {'width': 7},
                         {'field': 'Combobox', 'values': ['nm', 'eV'], 'width': 4}]])

        # Absortion edge (CdS, etc)
        if self.attributeEqual('_popt'):
            out.append([self.CurveEQE_absorptionEdge,
                        'Quick & dirty CdS estimate',
                        ['thickness [nm]', 'I0', 'material'],
                        [50, 1.0, 'CdS'], {},
                        [{'width': 6}, {'width': 7}, {'width': 7}]])
        if (not self.attributeEqual('_popt')
                and self.attributeEqual('_fitfunc', 'func_absorptionedge')):
            out.append([self.updateFitParam, 'Update fit',
                        ['thickness [nm]', 'I0', 'material'],
                        self.attr('_popt')])

        # Help button!
        out.append([self.printHelp, 'Help!', [], []])
        return out

    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        out.append(['nm <-> eV', ['nmeV', ''], ''])
        out.append(['Tauc plot', ['nmeV', 'tauc'], ''])
        if (not self.attributeEqual('_popt')
                and self.attributeEqual('_fitfunc', 'func_bandgapTaucCurveLog')):
            out.append(['Tauc plot (E ln(1-EQE)^2)',
                        ['nmeV', 'taucln1-eqe'], ''])
        return out

    def updateFitParamFormatPopt(self, f, param):
        """ override, for func_absorptionedge """
        if f == 'func_absorptionedge':
            return list(param)  # not np.array: last parameter can be str
        return Curve.updateFitParamFormatPopt(self, f, param)

    # FUNCTIONS RELATED TO FIT
    def bandgapFromTauc(self, nm, EQE, yLim=[.25, .70], xLim=[600, 1500], mode='EQE'):
        """
        Performs fit of low-energy side of EQE and returns [bandgap, slope].
        Executed at initialization.
        mode: 'EQE', or 'ln1-EQE'
        """
        if max(EQE) > 10:
            print('Function bandgapFromTauc: max(EQE) > 1. Multiplied by',
                  '0.01 for datapoint selection.')
            EQE = EQE * 0.01
        # select suitable data range
        mask = np.ones(len(nm), dtype=bool)
        for i in reversed(range(len(mask))):
            if (EQE[i] < yLim[0] or EQE[i] > yLim[1] or nm[i] < xLim[0]
                    or nm[i] > xLim[1]):
                mask[i] = False
        nm = nm[mask]
        EQE = EQE[mask]
        # perform fit
        eV = Curve.NMTOEV / nm
        tauc = (eV * np.log(1-EQE))**2 if mode == 'log1-EQE' else (eV * EQE)**2
        if len(tauc > 1):
            z = np.polyfit(eV, tauc, 1, full=True)[0]
            bandgap = -z[1]/z[0]  # p = np.poly1d(z)
            return [bandgap, z[0]]
        # print ('Function bandgapFromTauc: not enough suitable datapoints.')
        return [np.nan, np.nan]

    def CurveEQE_bandgap_print(self, yLim=None):
        out = self.CurveEQE_bandgap(yLim=yLim)
        print('Bandgap (Tauc (E*EQE)^2):',
              roundSignificant(out.attr('_popt')[0], 4), 'eV')
        return out

    def CurveEQE_bandgap(self, yLim=None):
        if yLim is None:
            yLim = [.25, .70]
        try:
            yLim = np.array(yLim) / self.attr('mulOffset', default=1)
            bandgap = self.bandgapFromTauc(self.x(), self.y(), yLim=yLim)
        except Exception:
            pass
        x = self.x()
        nm = np.arange(min(x), max(x), (max(x)-min(x))/1000)
        fit = self.func_bandgapTaucCurve(nm, *bandgap)
        return CurveEQE([nm, fit],
                        {'_popt': bandgap, '_fitFunc': 'func_bandgapTaucCurve',
                         'color': 'k', 'data filename': self.attr('filename'),
                         'muloffset': self.attr('mulOffset', 1)})

    def func_bandgapTaucCurve(self, nm, *bandgap):
        z = [bandgap[1], -bandgap[0]*bandgap[1]]
        p = np.poly1d(z)
        fit = p(Curve.NMTOEV / nm)
        for i in range(len(fit)):
            if fit[i] >= 0:
                fit[i] = np.sqrt(fit[i]) / (Curve.NMTOEV / nm[i])
            else:
                fit[i] = np.nan
        return fit

    def CurveEQE_bandgapLog_print(self, yLim=None):
        out = self.CurveEQE_bandgapLog(yLim=yLim)
        print('Bandgap (Tauc (E*ln(1-EQE))^2):',
              roundSignificant(out.attr('_popt')[0], 4), 'eV')
        return out

    def CurveEQE_bandgapLog(self, yLim=None):
        if yLim is None:
            yLim = [.25, .70]
        try:
            yLim = np.array(yLim) / self.attr('mulOffset', default=1)
            bandgap = self.bandgapFromTauc(self.x(), self.y(), yLim=yLim,
                                           mode='log1-EQE')
        except Exception:
            pass
        x = self.x()
        nm = np.arange(min(x), max(x), (max(x)-min(x))/1000)
        fit = self.func_bandgapTaucCurveLog(nm, *bandgap)
        return CurveEQE([nm, fit],
                        {'_popt': bandgap,
                         '_fitFunc': 'func_bandgapTaucCurveLog', 'color': 'k',
                         'data filename': self.attr('filename'),
                         'muloffset': self.attr('mulOffset', default=1)})

    def func_bandgapTaucCurveLog(self, nm, *bandgap):
        z = [bandgap[1], -bandgap[0]*bandgap[1]]
        p = np.poly1d(z)
        fit = p(Curve.NMTOEV / nm)
        for i in range(len(fit)):
            if fit[i] >= 0:
                fit[i] = 1 - np.exp(- np.sqrt(fit[i]) / (Curve.NMTOEV/nm[i]))
            else:
                fit[i] = np.nan
        return fit

    # bandgap derivative
    def bandgapDeriv(self, SGwidth=None, SGdegree=None, x_y_savgol=None, ROI=None):
        """
        Computes the bandgap based on the savitsky golay derivative,
        with a gaussian fit to the highest values
        SGwidth: 5. Parameters of Savitsky Golay filtering
        SGdegree: 3
        x_y_savgol (optional) to avoid calculating twice if already computed
            somewhere else
        """
        # need dummy x argument to use the updateFitParam method
        from scipy.optimize import curve_fit
        if x_y_savgol is None:
            x_, y_, savgol = self.bandgapDeriv_savgol(SGwidth, SGdegree,
                                                      ROI=ROI)
        else:
            x_, y_, savgol = x_y_savgol
        order = True if x_[1] > x_[0] else False
        # locate max, fit gaussian around location of max
        iMax_ = np.argmax(savgol)
        ROIfit = range(max(0, iMax_-1 - int(not order)),
                       min(len(savgol), iMax_+2 + int(order)))
        datax, datay = x_[ROIfit], savgol[ROIfit]
        p0 = [savgol[iMax_], x_[iMax_], 2*(datax[1]-datax[0])]

        def func_gaussSimple(x, a, x0, sigma):
            return a * np.exp(- (x - x0)**2 / (2 * sigma**2))

        try:
            popt, pcov = curve_fit(func_gaussSimple, datax, datay, p0=p0)
        except Exception as e:
            print('Exception CurveEQE.bandgapDeriv. Provided to curve_fit',
                  'length x', len(datax), 'length y', len(datay), ', p0', p0)
            print(type(e), e)
            popt = [np.nan, np.nan, np.nan]
        self.update({'bandgapDeriv': Curve.NMTOEV/popt[1]})
        return popt

    def bandgapDeriv_savgol(self, SGwidth=None, SGdegree=None, ROI=None):
        """ returns a reduced x,y range and the savitsky golay derivative """
        from scipy.signal import savgol_filter
        if SGwidth is None:
            SGwidth = 5
        if SGdegree is None:
            SGdegree = 2
        SGdegree = int(SGdegree)
        SGwidth = int(SGwidth)
        if SGdegree >= SGwidth:
            print('Curve EQE bandgap derivative: degree must be lower than',
                  'width for Savitsky-Golay filtering.')
            return None, None, None
        if SGwidth % 2 == 0:
            print('Curve EQE bandgap derivative: width must be odd for',
                  'Savitsky-Golay filtering.')
            return None, None, None
        # compute derivative
        x, y = self.x(), self.y()
        if len(x) < 5:
            print('Not enough datapoints, cannot continue with processing.')
            return False
        # determines if sorted asc. or desc. Do NOT check if not monotoneous
        order = True if x[1] > x[0] else False
        # data selection - up to EQE maximum
        iMax = np.argmax(y)
        if ROI is None:
            if order:
                ROI_ = range(max(0, iMax - 2), len(y))
            else:
                ROI_ = range(0, min(iMax + 2, len(y)))
        else:
            ROI_ = (x >= min(ROI)) * (x <= max(ROI))
        x_, y_, eV = x[ROI_], y[ROI_], Curve.NMTOEV / x[ROI_]
        # locate peak using Savitsky Golay filtering; then correct for possible
        # uneven data point spacing
        savgol = savgol_filter(y_, SGwidth, SGdegree, deriv=1)
        savgol /= np.append(np.append(eV[1] - eV[0], (eV[2:] - eV[:-2]) / 2),
                            eV[-1] - eV[-2])
        return x_, y_, savgol

    def CurveEQE_bandgapDeriv_print(self, SGwidth=None, SGdegree=None, ROI=None):
        out = self.CurveEQE_bandgapDeriv(SGwidth=SGwidth, SGdegree=SGdegree,
                                         ROI=ROI)
        print('Bandgap (derivative):',
              roundSignificant(out.attr('bandgapDeriv'), 4), 'eV')
        return out

    def CurveEQE_bandgapDeriv(self, SGwidth=None, SGdegree=None, ROI=None):
        """ returns the curve, with correspoding bandgap in the attributes """
        x_, y_, savgol = self.bandgapDeriv_savgol(SGwidth=SGwidth,
                                                  SGdegree=SGdegree, ROI=ROI)
        popt = self.bandgapDeriv(SGwidth=SGwidth, SGdegree=SGdegree,
                                 x_y_savgol=[x_, y_, savgol])
        out = CurveEQE([x_, savgol],
                       {'color': 'k', '_popt': [SGwidth, SGdegree],
                        'bandgapDeriv': Curve.NMTOEV/popt[1],
                        '_fitFunc': 'bandgapDeriv_savgol',
                        'data filename': self.attr('filename')})
        out.update({'muloffset': self.attr('mulOffset', 1)*0.1})
        # , 'offset':-max(out.y())})
        return out

    # reference spectra
    def referenceEQESpectrum(self, label=''):
        """
        label: the label of the reference spectrum to return
        """
        refs = self._referenceEQESpectra()
        for curve in refs:
            if curve.attr('label') == label:
                return curve
        print('CurveEQE referenceEQESpectrum: cannot find curve "', label, '"')
        print('Labels available:', [curve.attr('label') for curve in refs])
        return False

    @classmethod
    def _referenceEQESpectra(cls):
        """ returns the reference EQE spectra """
        if CurveEQE.EQE_REF_SPECTRA is None:
            path = os.path.dirname(os.path.abspath(__file__))
            refs = Graph(os.path.join(path, CurveEQE.EQE_REF_SPECTRA_FILE))
            CurveEQE.EQE_REF_SPECTRA = refs
        return CurveEQE.EQE_REF_SPECTRA

    def CurveEQE_returnAM15referenceCurve(self):
        # mechanisms to load file only once and store output as class variable
        if CurveEQE.EQE_AM15_REFERENCE is None:
            path = os.path.dirname(os.path.abspath(__file__))
            refs = Graph(os.path.join(path, CurveEQE.EQE_AM15_REFERENCE_FILE))
            if len(refs) <= 1:
                print('ERROR CurveEQE_returnAM15referenceCurve: cannot find',
                      'reference file', self.EQE_AM15_REFERENCE_FILE)
                return 0
            CurveEQE.EQE_AM15_REFERENCE = refs[1]
        return CurveEQE.EQE_AM15_REFERENCE

    def CurveEQE_returnAM0referenceCurve(self):
        """
        Returns AM0 reference spectrum.
        Data are in W/m2/um, returns spectral photon irradiance
        """
        if CurveEQE.EQE_AM0_REFERENCE is None:
            path = os.path.dirname(os.path.abspath(__file__))
            refs = Graph(os.path.join(path, CurveEQE.EQE_AM0_REFERENCE_FILE))
            if refs.length() == 0:
                print('ERROR CurveEQE_returnAM0referenceCurve: cannot find',
                      'reference file', self.EQE_AM0_REFERENCE_FILE)
                return 0
            CurveEQE.EQE_AM0_REFERENCE = refs[0]
        curve = CurveEQE.EQE_AM0_REFERENCE
        # h * c / wavelength -> energy WL conversion in vacuum for AM0
        energy = 6.62607004E-34 * 299792458 / (1e-9 * curve.x())
        irrad = curve.y() / energy / 1000
        return Curve([curve.x(), irrad], {})

    def _curveReference(self, file):
        """
        Returns some reference spectrum (spectral photon irradiance)
        file: the user can specify its own file
        """
        if file.endswith('.txt'):
            file = file[:-4]
        if file == self.EQE_AM15_REFERENCE_FILE[:-4]:
            return self.CurveEQE_returnAM15referenceCurve()
        if file == self.EQE_AM0_REFERENCE_FILE[:-4]:
            return self.CurveEQE_returnAM0referenceCurve()
        # maybe a custom input ?
        path = os.path.dirname(os.path.abspath(__file__))
        ref = Graph(os.path.join(path, file), silent=True)
        if ref.length() > 0:
            return ref.curve(0)
        print('ERROR CurveEQE._curveReference: cannot find reference file',
              file)
        return False

    # current calc
    def currentCalc(self, ROI=None, interpolatekind='linear', spectralPhotonIrrad=None, showintegratedcurrent=False, silent=False, ifGUI=False):
        """
        Computes the current EQE current using AM1.5 spectrum.
        Assumes the EQE vales are in range [0,1] and NOT [0,100].
        ROI: [nm_min, nm_max]
        interpolatekind: order for interpolation of EQE data. default 'linear'.
        spectralPhotonIrrad: filename in folder datatypes.
            By default will use AM1.5G.
        ifGUI: True will print the value, False will return the value.
        """
        if spectralPhotonIrrad is None:
            spectralPhotonIrrad = CurveEQE.EQE_AM15_REFERENCE_FILE
        refIrrad = self._curveReference(spectralPhotonIrrad)
        if not refIrrad:
            print('Error CurveEQE currentCalc, cannot find reference spectrum',
                  spectralPhotonIrrad)
            return
        ROIdef = [min(self.x()), max(self.x())]
        if ROI is None:
            ROI = ROIdef
        else:
            ROI = [max(ROIdef[0], ROI[0]), min(ROIdef[1], ROI[1])]
        # localize range of interest
        refROI = (refIrrad.x() >= ROI[0]) * (refIrrad.x() <= ROI[1])
        refDataX = refIrrad.x()[refROI]
        refDataY = refIrrad.y()[refROI]
        # interpolate data on ref x sampling
        # -> implicitely assuming sampling with more datapoints in ref than in data
        f = interpolate.interp1d(self.x(), self.y(), kind=interpolatekind)
        interpData = f(refDataX)
        # compute final spectrum
        finalSpectrum = refDataY * interpData
        # integriere QE*spektrum (auf gewähltem range)
        Jo = np.trapz(finalSpectrum, refDataX)
        EQEcurrent = Jo * 1.602176487E-19 / 10
        if not silent or ifGUI:
            print('Curve', self.attr('label'), 'EQE current:', EQEcurrent, 'mA/cm2')
        # return curve with integrated current
        if showintegratedcurrent:
            cumsum = [0]
            for i in range(1, len(finalSpectrum)):
                cumsum.append(cumsum[-1] + (finalSpectrum[i]+finalSpectrum[i-1])/2 * (refDataX[i] - refDataX[i-1]))
            cumsum = np.array(cumsum) * 1.602176487E-19 / 10
            color = self.attr('color', '')
            if color == '':
                color = 'k'
            curvecumsum = Curve([refDataX, cumsum], {'color': 'k' })
            curvecumsum.update({'label': self.attr('label') + ' cumulative EQE current',
                                'ax_twinx': True, 'color': color})
            return curvecumsum
        if ifGUI:
            return True
        return EQEcurrent

    # Urbach energy
    def CurveUrbachEnergy(self, ROIeV):
        """ Return fit in a Curve object, giving Urbach energy decay """
        from grapa.datatypes.curveArrhenius import CurveArrhenius, CurveArrheniusExpDecay
        ROIeV.sort()
        curve = CurveArrhenius(self.data, CurveArrheniusExpDecay.attr)
        out = curve.CurveArrhenius_fit(ROIeV)
        out.update(self.getAttributes(['offset', 'muloffset']))
        return out

    # ERE
    def ERE(self, Voc, T, Emin='auto', EminUnit='nm'):
        """
        Computes the External Radiative Efficiency from the EQE curve and the
        cell Voc.
        Returns ERE, and a Curve with the integrand
        Parameters:
        - Voc: cell Voc voltage in [V]
        - T: temperature in [K]
        - Emin: min E on which the integral is computed.
            Can be given in eV or in nm, see Eminunit.
        - Eminunit: 'eV' if Emin is given in eV, 'nm' otherwise. Default 'nm'
        """
        # does not matter if EQE is [0,1] or [0,100], this is corrected by the
        # calculation of Jsc (beware the day Jsc is patched!)
        import scipy.integrate as integrate
        CST_q = 1.602176634e-19  # elemental charge [C]
        CST_h = 6.62606979e-34  # Planck [J s]
        CST_c = 299792458  # speed of light [m s-1]
        CST_kb = 1.38064852e-23  # [J K-1]
        # variables check
        if Emin == 'auto':
            Emin, EminUnit = self.ERE_EminAuto()
        if EminUnit != 'eV':
            EminUnit = 'nm'
        if Emin != 'auto' and EminUnit == 'nm':
            Emin = Curve.NMTOEV / Emin  # convert nm into eV
        # retrieve data
        nm, EQE = self.x(), self.y()
        E = Curve.NMTOEV / nm * CST_q  # photon energy [J]
        Jsc = self.currentCalc(silent=True) * 10  # [A m-2] instead of [mAcm-2]
        # mask for integration
        mask = (E >= Emin * CST_q) if Emin != 'auto' else (E > 0)
        # start computing the expression
        integrand = EQE * E**2 / (np.exp(E / (CST_kb * T)) - 1)  # [J2]
        integral = np.abs(integrate.trapz(integrand[mask], x=E[mask]))  # [J3]
        ERE = 2 * np.pi * CST_q / CST_h**3 / CST_c**2 / Jsc  # [J-3]
        ERE *= np.exp(CST_q * Voc / CST_kb / T)  # new term unitless
        ERE *= integral  # output [unitless]
        lbl = ['ERE integrand to '+self.attr('label'), '', 'eV$^2$']
        return ERE, Curve([nm[mask], integrand[mask]/(CST_q)**2],
                          {'label': Graph().formatAxisLabel(lbl)})

    def ERE_GUI(self, Voc, T, Emin='auto', EminUnit='nm'):
        ERE, curve = self.ERE(Voc, T, Emin=Emin, EminUnit=EminUnit)
        print('External radiative efficiency estimate:',
              "{:.2E}".format(ERE), '(input Voc:', Voc, ')')
        curve.update({'ax_twinx': 1, 'color': 'k'})
        return curve

    def ERE_EminAuto(self):
        # smart Emin autodetect
        nm, EQE = self.x(), self.y()
        try:
            # identify nm where EQE = 0.5
            nmMax = np.max(nm[(EQE > 0.5*np.max(EQE))])
        except Exception:  # no suitable point
            return [0, 'nm']
        mask = (nm > nmMax) * (EQE > 0)
        nm_, EQElog = nm[mask], np.log10(EQE[mask])
        E = Curve.NMTOEV / nm_
        # sort nm & EQE in ascending nm order
        EQElog = EQElog[nm_.argsort()]
        nm_.sort()
        diff = (EQElog[1:] - EQElog[:-1]) / (E[1:] - E[:-1])  # d/dE log(EQE)
        nfault = 0
        Emin = np.max(nm) + 1  # by default, just below lowest point
        for i in range(1, len(diff)):
            if diff[i] < np.min(diff[:i]) * 0.5:  # 0.5 can be adjusted
                nfault += 1  # or another algorithm implemented
            if nfault > 1 or diff[i] < 0:
                Emin = roundSignificantRange([np.mean(nm_[i:i+2]), nm_[i]], 2)[0]
    #    graph2 = Graph([(0.5*(nm_[:-1]+nm_[1:])), diff])
    #    graph2.curve(0).update({'linespec': '-x'})
    #    graph2.plot()
        return [Emin, 'nm']

    # Quick & dirty calculation Layer thickness estiamte (e.g. CdS)
    def _materialRefractivek(self, material):
        """
        Returns imaginary part of refractive index of CdS, k
        Energy to be given in nm
        """
        from grapa.graph import Graph
        path = os.path.dirname(os.path.abspath(__file__))
        matclean = 'EQE_absorption_' + material.replace('/', '').replace('\\', '') + '.txt'
        pathtest = os.path.join(path, matclean)
        if os.path.exists(pathtest):
            return Graph(pathtest)[0]
        try:
            # try open the file, and return first Curve assumed to be (nm, k)
            return Graph(material)[0]
        except Exception as e:
            print('CurveEQE._materialRefractivek: please choose a material, ',
                  'or a file with (nm,k) data as 2-column. Input:', material)
            print('Exception', type(e), e)
            return False
        return False

    def func_absorptionedge(self, nm, thickness, I0, material='CdS'):
        try:  # check parameters are numeric
            thickness, I0 = float(thickness), float(I0)
        except ValueError:
            print('CurveEQE.func_absorptionedge: Did you really enter',
                  'thickness and I0 as float?')
            return False
        k = self._materialRefractivek(material)
        if not k:
            return False
        alpha = 4 * np.pi * k.y() / (k.x() * 1e-7)  # in cm-1
        transmitted = I0 * np.exp(-alpha * thickness * 1e-7)
        f = interpolate.interp1d(k.x(), transmitted, kind='linear',
                                 bounds_error=False)
        return f(nm)

    def CurveEQE_absorptionEdge(self, thickness, I0, material='CdS'):
        """
        Returns a Curve for quick-and-dirty estimate layer thickness, e.g. CdS.
        Curve represents the light transmitted through the layer and is
        computed from Beer-Lambert law as I0 * exp(-alpha * thickness),
        with alpha computed from k imaginary refractive index of material.
        thickness: layer thickness, in nm
        I0: amount of light entering the layer. Default: 1
        material: data  provided for 'CdS'. A file path might be provided,
            storing k data in 2-column file (nm, k)
        """
        values = self.func_absorptionedge(self.x(), thickness, I0, material)
        return CurveEQE([self.x(), values],
                        {'_fitfunc': 'func_absorptionedge',
                         '_popt': [thickness, I0, material],
                         'muloffset': self.attr('muloffset')})

    # Help
    def printHelp(self):
        print('*** *** ***')
        print('CurveEQE offer basic treatment of (external) quantum')
        print('efficiency of solar cells.')
        print('Input units should be [nm] and [0-1] (if needed, CurveSpectrum')
        print('can convert [eV] in [nm]).')
        print('Curve transforms:')
        print(' - nm <-> eV: switches the horizontal axis from nm to eV.')
        print(' - Tauc plot: displays (eV*EQE)^2 vs eV. The corresponding')
        print('   Tauc fit is a straight line.')
        print(' - (optional) Tauc plot (E ln(1-EQE)^2): only appears if a ')
        print('   Curve was fitted with the corresponding formula.')
        print('Analysis functions:')
        print(' - Bandgap fit Tauc (eV*EQE)^2: fit the bandgap with the')
        print(' indicated formula. Parameters:')
        print('   Range EQE: select data with EQE value in indicated range.')
        print(' - Bandgap derivative: determines the bandgap following the')
        print('   derivative approach. The derivative dEQE/dE is used instead')
        print('   of dEQE/dlambda as the resulting curve is often more')
        print('   symmetric. The algorithm computes the derivative with the')
        print('   Savitsky-Golay algorithm. Parameters:')
        print('   Savitsky Golay width, and degree: width should be odd, and')
        print('      degree must be between 1 and width. By default 5, 2.')
        print('      With parameters 3, 1 ones retrieves the symmetrical')
        print('      discrete difference (y_1+1 - y_i-1)  / (x_i+1 - x_i-1).')
        print(' - EQE current: integration of the product of the EQE with the')
        print('   reference AM1.5 solar spectrum '+self.EQE_AM15_REFERENCE_FILE+'.')
        print('   The code is similar to that of CurrentCalc matlab code.')
        print(' - Bandgap fit Tauc (eV*ln(1-EQE))^2: fits the bandgap with')
        print('   the indicated formula. In principle more exact, however')
        print('   highly sensitive to reflective and collection losses. Less')
        print('   robust than (eV*EQE)^2. Parameters:')
        print('   Range EQE: select data with EQE value in indicated range.')
        print(' - Urbach energies from fit to exponential decay.')
        print('   A new Curve is created, with main parameters the decay')
        print('   energy U and the energy at 100% EQE.')
        print('   ROI [eV]: limits to the fitted data, in eV.')
        print(' - ERE estimate of the external radiative efficiency, based on')
        print('   the EQE and the cell Voc. Parameters:')
        print('   Voc: cell Voc voltage, in [V].')
        print('   Emin: min E on which the integral is computed.')
        print('   EminUnit: "eV" if Emin is given in eV, "nm" otherwise.')
        print('   Ref: Green M. A., Prog. Photovolt: Res. Appl. 2012; 20:472–476')
        print(' - Quick & dirty CdS estimate: for crude estimation of layer')
        print('   thickness, e.g. CdS. The light transmitted is computed from')
        print('   Beer-Lambert law I0*exp(-alpha*thickness).')
        print('   thickness: layer thickness, in nm')
        print('   I0: amount of light entering the layer. Default: 1')
        print('   material: data provided for "CdS". Custom 2-column files')
        print('      might be provided (nm, k).')
        return True
