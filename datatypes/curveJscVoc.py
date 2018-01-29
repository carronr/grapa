# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:46:13 2016

@author: car
Copyright (c) 2018, Empa, Romain Carron
"""

import numpy as np


from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.curve import Curve
from grapa.mathModule import is_number, roundSignificant



class GraphJscVoc(Graph):
    
    FILEIO_GRAPHTYPE = 'Jsc-Voc curve'

    AXISLABELS = [['Voc', '', 'V'], ['Jsc', '', 'mA cm$^{-2}$']]
    
    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        if fileExt == '.txt' and fileName[:7] == 'JscVoc_' and line1[:14] == 'Sample name: 	':
           return True # can open this stuff
        return False
    
    def readDataFromFile(self, attributes, **kwargs):
        le = self.length()
        GraphIO.readDataFromFileGeneric(self, attributes, **kwargs)
        # expect 3 columns data
        self.castCurve(CurveJscVoc.CURVE, le, silentSuccess=True)
        # remove strange characters from attributes keys
        attr = self.curve(le).getAttributes()
        dictUpd = {}
        for key in attr:
            if key.replace('²', '2').replace('Â²','2') != key:
                dictUpd.update({key.replace('²', '2').replace().replace('Â²','2'): attr[key]})
                dictUpd.update({key: ''})
        self.curve(le).update(dictUpd)
        self.curve(le).update({'type': 'scatter', 'cmap': [[0,0,1], [1,0.43,0]], 'markeredgewidth': 0})
        lbl = self.getAttribute('label').replace('JscVoc ','').replace('Â²','2').replace(' Values Jsc [mA/cm2]','')
        self.curve(le).update({'label': lbl})
        self.curve(le+1).update({'type': 'scatter_c'})
        #self.curve(le+1).swapShowHide() # hide T curve
        self.update({'typeplot': 'semilogy', 'alter': ['', 'abs'],
                     'xlabel': self.formatAxisLabel(GraphJscVoc.AXISLABELS[0]),
                     'ylabel': self.formatAxisLabel(GraphJscVoc.AXISLABELS[1])})

    
    def findCurveWithX(self, curve):
        """
        Find the Curve with same x data as the given Curve
        JscVoc: that Curve should store temperatures
        """
        for cu in range(self.length()):
            if self.curve(cu) == curve:
                # start looping at position of the curve
                for c in range(cu+1, self.length()):
                    if np.array_equiv(curve.x(), self.curve(c).x()):
                        return self.curve(c)
                # could not find, start looking from beginning
                for c in range(cu):
                    if np.array_equiv(curve.x(), self.curve(c)):
                        return self.curve(c)
                return False
    def splitTemperatures(self, curve, threshold=3):
        """
        Splits the compiled data into different data (one for each T)
        curve: stores the Jsc-Voc pairs - will need to find the T
        threshold in °C
        """
        T = GraphJscVoc.findCurveWithX(self, curve)
        if T == False:
            print('Error JscVoc: cannot find temperature Curve, must have same Voc list as Jsc-Voc Curve. Aborted.')
            return [], []
        Voc = curve.x()
        Jsc = curve.y()
        Tem = T.y()
        datas = []
        temps = []
        j = 0
        data, temp = [], []
        for i in range(len(Voc)):
            # check if is new temperature
            if j != 0:
                if np.abs(Tem[i] - np.average(temp)) > threshold:
                    datas.append(data)
                    temps.append(np.average(temp))
                    data = []
                    temp = []
                    j = 0
            if j == 0: # is new temperature
                data = [[Voc[i]], [Jsc[i]]]
                temp = [Tem[i]]
            else: # is not new
                data[0].append(Voc[i])
                data[1].append(Jsc[i])
                temp.append(Tem[i])
            j += 1
        datas.append(data)
        temps.append(np.average(temp))
        return datas, temps
    def CurvesJscVocSplitTemperature(self, threshold=3, curve=None):
        """
        Splits the compiled data into different data (one for each T)
        threshold in °C
        curve: stores the Jsc-Voc pairs. If None: error. Weird prototype design
            to allow calls from GUI
        """
        if curve is None:
            print('Error CurvesJscVocSplitTemperature, you must provide argument key "curve" with a Jsc-Voc Curve.')
        datas, temps = GraphJscVoc.splitTemperatures(self, curve, threshold=3)
        attr = curve.getAttributes()
        out = []
        for i in range(len(datas)):
            out.append(CurveJscVoc(datas[i], attr))
            out[-1].update({'temperature': temps[i], 'type': '', 'cmap': ''})
            out[-1].update({'label': out[-1].getAttribute('label')+' '+'{:.0f}'.format(temps[i])+'K'})
        return out
 
    # handling of curve fitting
    def CurveJscVoc_fitNJ0(self, Voclim=None, Jsclim=None, threshold=3, graphsnJ0=True, curve=None, silent=False):
        """
        Fit the Jsc-Voc data, returns fitted Curves.
        If required, first splits data in different temperatures.
        Voclim, Jsclim: fit limits for Voc and Jsc, in data units
        threshold in °C
        graphsnJ0: if True, also returns A vs T, J0 vs T and J0 vs A*T
        curve: stores the Jsc-Voc pairs. If None: error. Weird prototype design
            to allow calls from GUI
        """
        if curve is None:
            print('Error CurveJscVoc_fitNJ0, you must provide argument key "curve" with a Jsc-Voc Curve.')
            return False
        try:
            T = curve.T(default=np.nan, silent=True)
        except Exception:
            print('Warning CurveJscVoc_fitNJ0, method T() not found. Curve type not suitable?')
            T = np.nan
        # check if is single or multiple temperature
        if np.isnan(T):
            datas, temps = GraphJscVoc.splitTemperatures(self, curve, threshold=threshold)
        else:
            datas, temps = [[curve.x(), curve.y()]], [T]
        # fit different data series (different illuminations at same T)
        out = []
        ns, J0s = [], []
        for i in range(len(datas)):
            n, J0 = curve.fit_nJ0(Voclim=Voclim, Jsclim=Jsclim, data=datas[i], T=temps[i])
            attr = {'color': 'k', 'area': curve.getArea(), '_popt': [n, J0], 
                    '_fitFunc': 'func_nJ0', '_Voclim': Voclim, '_Jsclim': Jsclim,
                    'temperature': temps[i],
                    'filename': 'fit to '+curve.getAttribute('filename').split('/')[-1].split('\\')[-1]}
            if not silent:
                print('Fit Jsc-Voc (T =',temps[i],'): ideality factor n =', n, ', J0 =', '{:1.4e}'.format(J0), '[mA/cm2].')
            x, y = curve.selectData(xlim=Voclim, ylim=Jsclim, data=datas[i])
            out.append(CurveJscVoc([x, curve.func_nJ0(x, n, J0, T=temps[i])], attr))
            ns.append(n)
            J0s.append(J0)
        if graphsnJ0:
            from grapa.datatypes.curveArrhenius import CurveArrhenius, CurveArrheniusJscVocJ00
            # generate n vs T, J0 vs T
            lbl = curve.getAttribute('label')
            if len(lbl) > 0:
                lbl += ' '
            out.append(Curve([temps, ns],  {'linestyle': 'none', 'linespec': 'o', 'label': lbl+'Ideality factor A vs T [K]'}))
            out.append(Curve([temps, J0s], {'linestyle': 'none', 'linespec': 'o', 'label': lbl+'J0 vs T [K]'}))
            out.append(CurveArrhenius([np.array(temps)*np.array(ns), J0s], CurveArrheniusJscVocJ00.attr))
            out[-1].update({'label': lbl+out[-1].getAttribute('label')})# 'label': lbl+'J0 vs A*T'
            #out[-1].update({'type': 'scatter', 'markeredgewidth':0, 'markersize':100, 'cmap': 'inferno'})# 'label': lbl+'J0 vs A*T'
            #out.append(Curve([np.array(temps)*np.array(ns), np.array(temps)], {'linestyle': 'none', 'type': 'scatter_c'}))
        return out

            
        
class CurveJscVoc(Curve):

    CURVE = 'Curve JscVoc'
    
    q = 1.60217657E-19 # [C]
    k = 1.38E-23    # [J K-1]

    Tdefault = 273.15 + 25
    
    
    def __init__ (self, data, attributes, silent=False) :
        # delete area from attributes, to avoid normalization during initialization
        Curve.__init__ (self, data, attributes, silent=silent)
        self.update ({'Curve': CurveJscVoc.CURVE}) # for saved files further re-reading

    
    # RELATED TO GUI
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...] (, [default1, default2, ...]) ]
        lbl = 'Area [cm2] (old value ' + "{:4.3f}".format(self.getArea()) + ')'
        out.append([self.setArea, 'Area correction', [lbl], [ "{:6.5f}".format(self.getArea())]]) # one line per function
        # fit function
        if self.getAttribute('_fitFunc') == '' or self.getAttribute('_popt', None) is None:
            out.append([GraphJscVoc.CurveJscVoc_fitNJ0, 'Fit J0 & ideality', ['Voclim', 'Jsclim', 'T max fluct.', 'compile'], [[0, max(self.x())], [0.1, max(self.y())], 3, True], {'curve': self}])
        else: # if fitted Curve
            out.append([self.updateFitParam, 'Modify fit', ['n', 'J0'], roundSignificant(self.getAttribute('_popt'), 5)])
        # split according to temperatures
        out.append([GraphJscVoc.CurvesJscVocSplitTemperature, 'Separate according to T', ['T max fluctuations'], [3], {'curve': self}])
        out.append([self.printHelp, 'Help!', [], []])
        return out
    
    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        out.append(['Log10 abs', ['', 'abs'], 'semilogy'])
        return out


    # Handling of cell area
    def getArea(self):
        """ return area of the cell as stored in the Curve parameters """
        area = self.getAttribute('area [cm2]')
        if is_number(area):
            return area
        return 1
    def setArea(self, new):
        """ correct the cell area, and scale the y (list of Jsc) accordingly """
        old = self.getArea()
        self.setY(self.y() * old / new)
        self.update({'area [cm2]': new})
        return True
        
    def fit_nJ0(self, Voclim=None, Jsclim=None, data=None, T=None):
        """ perform fitting, returns best fit parameters """
        datax, datay = self.selectData(xlim=Voclim, ylim=Jsclim, data=data)
        if T is None:
            T = self.T()
        # actual fitting
        datay = np.log(datay)
        z = np.polyfit(datax, datay, 1, full=True)[0]
        n = self.q / self.k / T / z[0]
        J0 = np.exp(z[1])
        return [n, J0]
    def func_nJ0(self, Voc, n, J0, T=None):
        """ fit function """
        if T is None:
            T = self.T()
        out = J0 * np.exp(self.q * Voc / (n * self.k * T))
        return out
    
    def T(self, default=None, silent=False):
        """ Returns the acquisition temperature, otherwise default. """ 
        test = self.getAttribute('temperature', 0)
        if test != 0:
            return test
        if not silent:
            print('Curve JscVoc cannot find keyword temperature.', self.getAttributes())
        if default is None:
            return CurveJscVoc.Tdefault
        return default

    def printHelp(self):
        print('*** *** ***')
        print('CurveJV offer basic treatment of Jsc-Voc pairs of solar cells.')
        print('Based on Thomas Weiss script, and on the following references:')
        print('   Schock, Scheer, p111, footnote 20')
        print('   Hages et al., JAP 115, 234504 (2014)')
        print('Curve transforms:')
        print(' - Linear: standard is current density [mA cm-2] versus [V], at different light intensitites.')
        print(' - Log 10 abs: logarithm of J vs V. Same display as JV curve, to visualize the diode behavior.')
        print('Analysis functions:')
        print(' - Area correction: scale Jsc data according to cell area.')
        print(' - Fit J0 & ideality: fit Jsc vs Voc data and extract J0 and ideality factor A. Parameters:')
        print('   Voclim, Jsclim: fit limits for Voc and Jsc')
        print('   T max fluct.: identify groups of temperatures to fit only relevant data together.')
        print('       A new temperature is identified if a point deviates more than "value" from the average.')
        print('   compile: after fitting data, returns Curves with results: ideality factor versus T,')
        print('      J0 versus T, and J0 vs A * T.')
        print(' - Separate according to T: split data in several Curves according to the temperature identified.')
        print('      A new Curve is created once a point deviates more than "value" from the average.')
        return True

