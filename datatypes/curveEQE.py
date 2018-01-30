# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:46:13 2016

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np
import os
from scipy import interpolate


from grapa.graph import Graph
from grapa.curve import Curve
from grapa.mathModule import roundSignificant



class GraphEQE(Graph):
    
    FILEIO_GRAPHTYPE = 'EQE curve'
    FILEIO_GRAPHTYPE_OLD = 'EQE curve (old)'
    
    AXISLABELS = [['Wavelength', '\lambda', 'nm'], ['Cell EQE', '', '%']]
    
    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', **kwargs):
        # new QE files
        if fileExt == '.sr' and line1 == 'This is a QE measurment':
            return True
        # old setup QE files (2013(?) or older)
        if fileExt == '.sr' and line1[0:16] == 'Reference Cell: ':
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        ifOld = False
        # retrieve sample name - code should work for old and new file format
        f = open(self.filename, 'r')
        line = f.readline() # retrieve sample, should be stored on the 2nd line
        if line[0:16] == 'Reference Cell: ':
            ifOld = True
        line = list(filter(len,
                f.readline().replace(': ', '\t').strip(' \r\n\t').split('\t')))
        # look for acquisition attributes
        attributesFile = {}
        interests = [['Amplification settings', 'amplification'],
                     ['Lockin settings', 'lockin']]
        for l in f:
            for interest in interests:
                if l.startswith(interest[0]):
                    for param in l.strip().split('\t'):
                        if ': ' in param:
                            tmp = param.split(': ')
                            attributesFile.update({interest[1]+tmp[0].replace(' ',''): tmp[1]})
            if len(l) == 1 and l[0] in ['\n']:
                break
        f.close()
        data = np.array([np.nan, np.nan])
        try:
            kw = {'delimiter': '\t', 'invalid_raise': False}
            if ifOld:
                kw.update({'skip_header': 14, 'usecols': [0, 6]})
            else:
                kw.update({'skip_header': 15, 'usecols': [0, 5]})
            data = np.transpose(np.genfromtxt(self.filename, **kw))
        except Exception:
            if not self.silent:
                print('readDataFromFileEQE cannot read file', self.filename)
        # normalize data
        self.append(CurveEQE(data, attributes))
        self.curve(-1).update(attributesFile)
        if len(data.shape) > 1:
            self.curve(-1).update({'mulOffset': 100})
        # update label with information stored inside the file
        self.update({'label': line[-1].replace('_', ' ')})
        # some default settings
        self.update({'xlabel': self.formatAxisLabel(GraphEQE.AXISLABELS[0]),
                     'ylabel': self.formatAxisLabel(GraphEQE.AXISLABELS[1]),
                     'ylim': [0, 100], 'xlim': [300, np.nan]})
        self.headers.update({'collabels': ['Wavelength [nm]', 'EQE [%]']})
        # newer versions of EQE include phase signal -> one more header line
        self.update({'sample': self.curve(0).getAttribute('label')})
        # default value in files old setup
        if self.headers['sample'] == 'Ref':
            filenam_, fileext = os.path.splitext(self.getAttribute('filename'))
            self.curve(-1).update({'label':
                                   (filenam_.split('/')[-1]).split('\\')[-1]})
            self.headers.update({'sample': filenam_})
        if ifOld:
            self.headers.update({'meastype': GraphEQE.FILEIO_GRAPHTYPE_OLD})



    

class CurveEQE(Curve):
    
    CURVE = 'Curve EQE'

    EQE_AM15_REFERENCE = 'AM1-5_Ed2-2008.txt'
    EQE_BEST_CELL_REF = 'EQE_20.4_cell.txt'
    EQE_BEST_CELL_LABEL = 'Empa 20.4%'

    
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
        flag = True if len(data.shape) > 1 else False # check if data loaded
        while flag:
            if data[0, 0] == 0:
                data = data[:, 1:]
            else:
                flag = False
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        # bandgap calculation
        popt = self.getAttribute('_popt')
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
        self.update ({'Curve': CurveEQE.CURVE})



    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
        # tauc (E*EQE)**2
        if (not self.attributeEqual('_popt')
            and self.attributeEqual('_fitfunc', 'func_bandgapTaucCurve')):
            out.append([self.updateFitParam, 'Update fit', ['Eg', 'slope'],
                        roundSignificant(self.getAttribute('_popt'),5)])
        elif self.attributeEqual('_popt'):
            mulOffset = self.getAttribute('mulOffset', default=1)
            out.append([self.CurveEQE_bandgap_print,
                        'Bandgap fit Tauc (eV*EQE)^2', ['Range EQE'],
                       ['['+str(0.25*mulOffset)+', '+str(0.70*mulOffset)+']']])
        # Eg from derivative method
        if (not self.attributeEqual('_popt')
            and self.attributeEqual('_fitfunc', 'bandgapDeriv_savgol')):
            out.append([None, 'Cannot update, must create a new one from '
                        +'initial EQE curve', [], []])
        elif self.attributeEqual('_popt'):
            out.append([self.CurveEQE_bandgapDeriv_print, 'Bandgap derivative',
                        ['Savitsky Golay width', 'degree'], [5,2]])
        # EQE current
        if self.attributeEqual('_popt'):
            out.append([self.currentCalc, 'EQE current', ['ROI'], [[min(self.x()), max(self.x())]]])
        # EQE 20.4%
        out.append([self.CurveEQE_Empa20p4, self.EQE_BEST_CELL_LABEL, [], []])
        # tauc (E*ln(1-EQE))**2
        if (not self.attributeEqual('_popt')
            and self.attributeEqual('_fitfunc', 'func_bandgapTaucCurveLog')):
            out.append([self.updateFitParam, 'Update fit', ['Eg', 'slope'],
                        roundSignificant(self.getAttribute('_popt'),5)])
        elif self.attributeEqual('_popt'):
            mulOffset = self.getAttribute('mulOffset', default=1)
            out.append([self.CurveEQE_bandgapLog_print,
                        'Bandgap fit Tauc (eV*ln(1-EQE))^2', ['Range EQE'],
                    ['['+str(0.30*mulOffset)+', '+str(0.88*mulOffset)+']']])
        # Urbach fitting
        if self.attributeEqual('_popt'):
            try: # Urbach fitting
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
                ROI = [self.x(index=x0, alter='nmeV'), self.x(index=x1, alter='nmeV')]
                ROI.sort()
                mulOffset = self.getAttribute('mulOffset', 1)
                out.append([self.CurveUrbachEnergy,
                            CurveArrheniusExpDecay.BUTTONFitLabel(),
                            [CurveArrheniusExpDecay.BUTTONFitROI()], [ROI]])
            except ImportError as e:
                print('WARNING CurveEQE: do not find grapa.datatypes.curveArrhenius.')
                print(type(e), e)
            
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



    # FUNCTIONS RELATED TO FIT

    def bandgapFromTauc(self, nm, EQE, yLim=[.25, .70], xLim=[600, 1500],
                        mode='EQE') :
        """
        Performs fit of low-energy side of EQE and returns [bandgap, slope].
        Executed at initialization.
        mode: 'EQE', or 'ln1-EQE'
        """
        if max(EQE) > 10 :
            print ('Function bandgapFromTauc: max(EQE) > 1. Multiplied by 0.01'
                   + ' for datapoint selection.')
            EQE = EQE * 0.01
        # select suitable data range
        mask = np.ones(len(nm), dtype=bool)
        for i in reversed(range (len (mask))) :
            if (EQE[i] < yLim[0] or EQE[i] > yLim[1] or nm[i] < xLim[0]
                or nm[i] > xLim[1]):
                mask[i] = False
        nm = nm[mask]
        EQE = EQE[mask]
        # perform fit
        eV = Curve.NMTOEV / nm
        tauc = (eV * np.log(1-EQE))**2 if mode == 'log1-EQE' else (eV * EQE)**2
        if len (tauc > 1):
            z = np.polyfit(eV, tauc, 1, full=True)[0]
            bandgap = -z[1]/z[0] #p = np.poly1d(z)
            return [bandgap, z[0]]
        #print ('Function bandgapFromTauc: not enough suitable datapoints.')
        return [np.nan, np.nan]
    
    def CurveEQE_bandgap_print(self, yLim=None):
        out = self.CurveEQE_bandgap(yLim=yLim)
        print ('Bandgap (Tauc (E*EQE)^2):',
               roundSignificant(out.getAttribute('_popt')[0],4), 'eV')
        return out
    def CurveEQE_bandgap(self, yLim=None):
        if yLim is None:
            yLim = [.25, .70]
        try:
            yLim = np.array(yLim) / self.getAttribute('mulOffset', default=1)
            bandgap = self.bandgapFromTauc(self.x(), self.y(), yLim=yLim)
        except:
            pass
        x = self.x()
        nm = np.arange(min(x), max(x), (max(x)-min(x))/1000)
        fit = self.func_bandgapTaucCurve(nm, *bandgap)
        return CurveEQE([nm, fit],
                {'_popt': bandgap, '_fitFunc': 'func_bandgapTaucCurve',
                 'color': 'k', 'data filename': self.getAttribute('filename'), 
                 'muloffset': self.getAttribute('mulOffset', default=1)})
    def func_bandgapTaucCurve(self, nm, *bandgap):
        z = [bandgap[1], -bandgap[0]*bandgap[1]]
        p = np.poly1d(z)
        fit = p(Curve.NMTOEV / nm)
        for i in range(len(fit)):
            if fit[i] >=0:
                fit[i] = np.sqrt(fit[i]) / (Curve.NMTOEV / nm[i])
            else:
                fit[i] = np.nan
        return fit
  
    def CurveEQE_bandgapLog_print(self, yLim=None):
        out = self.CurveEQE_bandgapLog(yLim=yLim)
        print ('Bandgap (Tauc (E*ln(1-EQE))^2):',
               roundSignificant(out.getAttribute('_popt')[0],4), 'eV')
        return out
    def CurveEQE_bandgapLog(self, yLim=None):
        if yLim is None:
            yLim = [.25, .70]
        try:
            yLim = np.array(yLim) / self.getAttribute('mulOffset', default=1)
            bandgap = self.bandgapFromTauc(self.x(), self.y(), yLim=yLim,
                                           mode='log1-EQE')
        except:
            pass
        x = self.x()
        nm = np.arange(min(x), max(x), (max(x)-min(x))/1000)
        fit = self.func_bandgapTaucCurveLog(nm, *bandgap)
        return CurveEQE([nm, fit],
                        {'_popt': bandgap,
                        '_fitFunc': 'func_bandgapTaucCurveLog', 'color': 'k',
                        'data filename': self.getAttribute('filename'),
                       'muloffset': self.getAttribute('mulOffset', default=1)})
    def func_bandgapTaucCurveLog(self, nm, *bandgap):
        z = [bandgap[1], -bandgap[0]*bandgap[1]]
        p = np.poly1d(z)
        fit = p(Curve.NMTOEV / nm)
        for i in range(len(fit)):
            if fit[i] >=0:
                fit[i] = 1 - np.exp(- np.sqrt(fit[i]) / (Curve.NMTOEV/nm[i]))
            else:
                np.nan
        return fit


    def bandgapDeriv(self, SGwidth=None, SGdegree=None, x_y_savgol=None):
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
            x_, y_, savgol = self.bandgapDeriv_savgol(SGwidth, SGdegree)
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
            return a * np.exp( - (x-x0)**2 / (2*sigma**2))
        popt, pcov = curve_fit(func_gaussSimple, datax, datay, p0=p0)
        self.update({'bandgapDeriv': Curve.NMTOEV/popt[1]})
        return popt
    def bandgapDeriv_savgol(self, SGwidth=None, SGdegree=None):
        """ returns a reduced x,y range and the savitsky golay derivative """
        from scipy.signal import savgol_filter
        if SGwidth is None:
            SGwidth = 5
        if SGdegree is None:
            SGdegree = 2
        SGdegree = int(SGdegree)
        SGwidth = int(SGwidth)
        if SGdegree >= SGwidth:
            print('Curve EQE bandgap derivative: degree must be lower than width for Savitsky-Golay filtering.')
            return None, None, None
        if SGwidth % 2 == 0:
            print('Curve EQE bandgap derivative: width must be odd for Savitsky-Golay filtering.')
            return None, None, None
        #compute derivative
        x, y = self.x(), self.y()
        if len(x) < 5:
            print('Not enough datapoints, cannot continue with processing.')
            return False
        # determines if sorted asc. or desc. Do NOT check if not monotoneous
        order = True if x[1] > x[0] else False
        # data selection - up to EQE maximum
        iMax = np.argmax(y)
        if order:
            ROI = range(max(0,iMax-2), len(y))
        else:
            ROI = range(0, min(iMax+2, len(y)))
        x_, y_, eV = x[ROI], y[ROI], Curve.NMTOEV / x[ROI]
        # locate peak using Savitsky Golay filtering; then correct for possible
        # uneven data point spacing
        savgol = savgol_filter(y_, SGwidth, SGdegree, deriv=1)
        savgol /= np.append(np.append(eV[1]-eV[0], (eV[2:]-eV[:-2])/2),
                                      eV[-1]-eV[-2])
        return x_, y_, savgol
    def CurveEQE_bandgapDeriv_print(self, SGwidth=None, SGdegree=None):
        out = self.CurveEQE_bandgapDeriv(SGwidth=SGwidth, SGdegree=SGdegree)
        print ('Bandgap (derivative):',
               roundSignificant(out.getAttribute('bandgapDeriv'), 4), 'eV')
        return out
    def CurveEQE_bandgapDeriv(self, SGwidth=None, SGdegree=None):
        """ returns the curve, with correspoding bandgap in the attributes """
        x_, y_, savgol = self.bandgapDeriv_savgol(SGwidth=SGwidth,
                                                  SGdegree=SGdegree)
        popt = self.bandgapDeriv(SGwidth=SGwidth, SGdegree=SGdegree,
                                 x_y_savgol=[x_,y_,savgol])
        out = CurveEQE([x_, savgol],
                       {'color': 'k', '_popt': [SGwidth, SGdegree],
                       'bandgapDeriv': Curve.NMTOEV/popt[1],
                       '_fitFunc': 'bandgapDeriv_savgol',
                       'data filename': self.getAttribute('filename')})
        out.update({'muloffset': -self.getAttribute('mulOffset', 1)*0.1,
                    'offset':-max(out.y())})
        return out




    def CurveEQE_Empa20p4(self):
        """ Returns the 20.4% Empa cell EQE. """
        import os.path
        from grapa.graph import Graph
        path = os.path.dirname(os.path.abspath(__file__))
        graph = Graph(os.path.join(path, self.EQE_BEST_CELL_REF), silent=True)
        if graph.length() > 0:
            return graph.curve(-1)
        graph = Graph(self.EQE_BEST_CELL_REF, silent=True)
        if graph.length() > 0:
            return graph.curve(-1)
        graph = Graph('_modules/'+self.EQE_BEST_CELL_REF, silent=True)
        if graph.length() > 0:
            return graph.curve(-1)
        print ('Data file EQE Empa 20.4% not found!')



    # other functions
    def CurveEQE_returnAM15referenceCurve(self):
        """ returns AM1.5 reference spectrum """
        import os.path
        from grapa.graph import Graph
        path = os.path.dirname(os.path.abspath(__file__))
        ref = Graph(os.path.join(path, self.EQE_AM15_REFERENCE), silent=True)
        if ref.length() > 0:
            return ref.curve(1)
        ref = Graph(self.EQE_AM15_REFERENCE, silent=True)
        if ref.length() > 0:
            return ref.curve(1)
        ref = Graph('_modules/'+self.EQE_AM15_REFERENCE, silent=True)
        if ref.length() > 0:
            return ref.curve(1)
        print('ERROR CurveEQE_returnAM15reference: cannot find reference file',
              self.EQE_AM15_REFERENCE)
        return 0
    
    
    
    def currentCalc(self, ROI=None, silent=False):
        """ Computes the current EQE current using AM1.5 spectrum """
        ROIdef = [min(self.x()), max(self.x())]
        if ROI is None:
            ROI = ROIdef
        else:
            ROI = [max(ROIdef[0], ROI[0]), min(ROIdef[1], ROI[1])]
        refIrrad = self.CurveEQE_returnAM15referenceCurve()
        # localize range of interest
        refROI = (refIrrad.x() >= ROI[0]) * (refIrrad.x() <= ROI[1])
        refDataX = refIrrad.x()[refROI]
        refDataY = refIrrad.y()[refROI]
        # interpolate data on ref x sampling
        # spline interpolation degree 1
        f = interpolate.interp1d(self.x(), self.y(), kind='cubic')
        interpData = f(refDataX)
        # compute final spectrum
        finalSpectrum = refDataY * interpData
        # integriere QE*spektrum (auf gewähltem range)
        Jo = np.trapz(finalSpectrum, refDataX)
        EQEcurrent = Jo*1.602176487E-19/10
        if not silent:
            print('Curve', self.getAttribute('label'), 'EQE current:', 
                  EQEcurrent, 'mA/cm2')
        return EQEcurrent
        
    
    def CurveUrbachEnergy(self, ROIeV):
        """ Return fit in a Curve object, giving Urbach energy decay """
        from datatypes.curveArrhenius import CurveArrhenius, CurveArrheniusExpDecay
        ROIeV.sort()
        curve = CurveArrhenius(self.data, CurveArrheniusExpDecay.attr)
        out = curve.CurveArrhenius_fit(Tlim=ROIeV)
        out.update(self.getAttributes(['offset', 'muloffset']))
        return out
        
    
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
        print( '  Curve was fitted with the corresponding formula.')
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
        print('   reference AM1.5 solar spectrum '+self.EQE_AM15_REFERENCE+'.')
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
        return True

        