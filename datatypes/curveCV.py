# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 07:34:12 2017
@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np
import warnings
from copy import deepcopy


from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.curve import Curve
from grapa.mathModule import roundSignificant, derivative, is_number



class GraphCV(Graph):

    FILEIO_GRAPHTYPE = 'C-V curve'

    AXISLABELS = [['Voltage', 'V', 'V'], ['Capacitance', 'C', 'nF']] 

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        if fileExt == '.txt' and line1.strip()[:12] == 'Sample name:' and fileName[0:4] == 'C-V_':
           return True
        return False
    
        
    def readDataFromFile(self, attributes, **kwargs):
        len0 = self.length()
        GraphIO.readDataFromFileGeneric(self, attributes)
        self.castCurve('Curve CV', len0, silentSuccess=True)
        # label based on file name, maybe want to base it on file content
        self.curve(len0).update({'label': self.curve(len0).getAttribute('label').replace('C-V ','').replace(' [nF]','').replace(' Capacitance', '').replace('T=','')})
        #set [nF] units
        self.curve(len0).setY(1e9 * self.curve(len0).y())
        # compute phase angle if required
        nb_add = 0
        if self.getAttribute('_CVLoadPhase', False) is not False:
            f = 1e5 # [Hz]
            C = self.curve(len0).y() * 1e-9 # in [F], not [nF]
            conductance = None
            for c in range(len0+1, self.length()): # retrieve resistance curve, not always same place
                lbl = self.curve(c).getAttribute('Voltage')
                if isinstance(lbl, str) and lbl[:1] == 'R':
                    conductance = 1 / self.curve(c).y()
                    break
            if conductance is None:
                print('ERROR Read', self.filename, 'as CurveCf with phase: cannot find R.')
            else:
                phase_angle = np.arctan(f * 2 * np.pi * C / conductance) * 180. / np.pi
#                phase_angle = np.arctan(C / conductance) * 180. / np.pi
                self.append(Curve([self.curve(len0).x(), phase_angle], self.curve(len0).getAttributes()))
                nb_add += 1
        # delete unneeded Curves
        for c in range(self.length()-1-nb_add, len0, -1):
            self.deleteCurve(c)
        # cosmetics
        axisLabels = deepcopy(GraphCV.AXISLABELS)
        # normalize with area C
        area = self.curve(len0).getAttribute('cell area (cm2)', None)
        if area is None:
            area = self.curve(len0).getAttribute('cell area', None)
        if area is None:
            area = self.curve(len0).getAttribute('area', None)
        if area is not None:
            self.curve(len0).setY(self.curve(len0).y() / area)
            self.curve(len0).update({'cell area (cm2)': area})
            axisLabels[1][2] = axisLabels[1][2].replace('F', 'F cm$^{-2}$')
            if not self.silent:
                print('Capacitance normalized to area', self.curve(len0).getArea(),'cm2.')
        # graph cosmetics
        self.update({'xlabel': self.formatAxisLabel(axisLabels[0]),
                     'ylabel': self.formatAxisLabel(axisLabels[1])}) # default



class CurveCV(Curve):
    
    CURVE = 'Curve CV'

    CST_q = 1.6021766208e-19 # C # electrical charge
    CST_eps0 = 8.85418782e-12 # m-3 kg-1 s4 A2 # vacuum permittivity
    CST_epsR = 10 # relative permittivity
    
    CST_MottSchottky_Vlim_def = [0, 0.4]
    CST_MottSchottky_Vlim_adaptative = [-0.5, np.inf]
    
    
    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        # for saved files further re-reading
        self.update ({'Curve': CurveCV.CURVE})
    


    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
        out.append([self.setArea, 'Set cell area', ['New area'], [self.getArea()]])
        # fit Mott-Schottky
        if self.getAttribute('_fitFunc', None) is None or self.getAttribute('_popt', None) is None:
            out.append([self.CurveCV_fitVbiN, 'Fit Mott-Schotty', ['ROI [V]'], [CurveCV.CST_MottSchottky_Vlim_def]])
            out.append([self.CurveCV_fitVbiN_smart, 'Fit Mott-Schotty (around min N_CV)', ['Within range [V]'], [CurveCV.CST_MottSchottky_Vlim_adaptative]])
        else:
            param = roundSignificant(self.getAttribute('_popt'), 5)
            out.append([self.updateFitParam, 'Update fit', ['V_bi', 'N_CV'], [param[0], '{:1.4e}'.format(param[1])]])
        # curve extraction for doping at 0 V
        out.append([self.CurveCV_0V, 'Show doping at', ['V='], [0]])
        # set epsilon
        out.append([self.setEpsR, 'Set epsilon r', ['default = 10'], [self.getEpsR()]])
        out.append([self.printHelp, 'Help!', [], []])
# 1/C2 = 2 / q / Nc / Ks / eps0 / A2 (Vbi-V)
        
        return out
    
    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        #out.append(['nm <-> eV', ['nmeV', ''], ''])
        out.append(['Mott-Schottky (1/C^2 vs V)', ['', 'CurveCV.y_ym2'], ''])
        out.append(['Carrier density N_CV [cm-3] vs V', ['', 'CurveCV.y_CV_Napparent'], ''])
        out.append(['Carrier density N_CV [cm-3] vs depth [nm]', ['CurveCV.x_CVdepth_nm', 'CurveCV.y_CV_Napparent'], ''])
        return out

    # FUNCTIONS used for curve transform (alter)
    def y_ym2(self, xyValue=None, **kwargs):
        """ Mott-Schottky plot: 1 / C**2 """
        if xyValue is not None:
            return xyValue[1]
        return 1 / (self.y(**kwargs) ** 2)
        
    def x_CVdepth_nm(self, **kwargs):
        """ apparent probing depth, assuming planar capacitor. """
        eps_r = CurveCV.getEpsR(self)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = eps_r * CurveCV.CST_eps0 / (1e-5 * self.y(**kwargs)) * 1e9 # y defined as nF/cm2, output depth in [nm]
        return w
    def y_CV_Napparent(self, xyValue=None, **kwargs):
        """ apparent carrier density N_CV """
        if xyValue is not None:
            return xyValue[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dCm2dV = derivative(self.x(**kwargs), (1e-5 * self.y(**kwargs)) ** (-2)) # C from [nF cm2] to [F m-2]
        eps_r = CurveCV.getEpsR(self)
        N = - 2 / (CurveCV.CST_q * eps_r * CurveCV.CST_eps0 * (dCm2dV)) * 1e-6 # output unit in [cm-3]
        return N
    
    
    # FUNCTIONS RELATED TO GUI (fits, etc.)
    def setArea(self, value):
        oldArea = self.getArea()
        self.update({'cell area (cm2)': value})
        self.setY(self.y() / value * oldArea)
        return True
    def getArea(self):
        return self.getAttribute('cell area (cm2)', 1)
    
    # handle custom-defined eps_r
    def getEpsR(self):
        if self.getAttribute('epsR', 0) != 0:
            return self.getAttribute('epsR')
        return CurveCV.CST_epsR
    def setEpsR(self, value):
        self.update({'epsR': value})

        

    # functions for fitting Mott-Schottky plot
    def CurveCV_fitVbiN(self, Vlim=None, silent=False):
        """
        Returns a Curve based on a fit on Mott-Schottky plot.
        Calls fit_MottSchottkyto get parameters, then call func_MottSchottky
        to generate data.
        """
        Vbi, N_CV = self.fit_MottSchottky(Vlim=Vlim)
        attr = {'color': 'k', 'area': self.getArea(), '_popt': [Vbi, N_CV], '_fitFunc': 'func_MottSchottky', '_Vlim': Vlim, 'filename': 'fit to '+self.getAttribute('filename').split('/')[-1].split('\\')[-1]}
        if not silent:
            print('Fit Mott-Schottky plot: Vbi =', Vbi, 'V, apparent doping N_CV =', '{:1.4e}'.format(N_CV) + '.')
        return CurveCV([self.x(), self.func_MottSchottky(self.x(), Vbi, N_CV)], attr)
    def CurveCV_fitVbiN_smart(self, Vrange=None, window=[-2,2], silent=False):
        """
        Returns a Curve based on a fit on Mott-Schottky plot, after first 
        guessing best possible range for fifting (where N_CV is lowest)
        """
        Vlim = self.smartVlim_MottSchottky(Vlim=Vrange, window=window)
        return self.CurveCV_fitVbiN(Vlim=Vlim, silent=False)

    def fit_MottSchottky(self, Vlim=None):
        """
        Fits the C-V data on Mott-Schottky plot, in ROI Vlim[0] to Vlim[0].
        Returns built-in voltage Vbi [V] and apparent doping density N_CV [cm-3].
        """
        datax, datay = self.selectData(xlim=Vlim)
        if len(datax) == 0:
            return np.nan, np.nan
        datay =  1 / (datay * 1e-5) ** 2  # calculation in SI units [F m-2]
        z = np.polyfit(datax, datay, 1, full=True)[0]
        Vbi = - z[1] / z[0]
        N_CV = - 2 / (CurveCV.CST_q * CurveCV.getEpsR(self) * CurveCV.CST_eps0) / z[0]
        return Vbi, N_CV * 1e-6 # N_CV in [cm-3]
    def func_MottSchottky(self, V, Vbi, N_CV):
        """
        Returns C(V) which will appear linear on a Mott-Schottky plot.
        V_bi built-in voltage [V]
        N_CV apparent doping density [cm-3]
        Output: C [nF cm-2]
        """
        out = V * np.nan
        if np.isnan(Vbi) or N_CV < 0:
            return out
        mask = (V < Vbi)
        Cm2 = 2 / (CurveCV.CST_q * CurveCV.getEpsR(self) * CurveCV.CST_eps0 * (N_CV * 1e6)) * (Vbi - V)
        out[mask] = (Cm2[mask] ** (-0.5))
        return out * 1e5 # output unit [nF is cm-2]
        
        
    def smartVlim_MottSchottky(self, Vlim=None, window=[-2,2]):
        """
        Returns Vlim [Vmin, Vmax] offering a possible best range for Mott-
        Schottky fit.
        Assumes V in monotoneous increasing/decreasing.
        Window: how many points around best location are taken. Default [-2,2]
        """
        V = self.x()
        N = self.y(alter='CurveCV.y_CV_Napparent')
        # identify index within given V limits
        if Vlim is None:
            Vlim = CurveCV.CST_MottSchottky_Vlim_adaptative
        Vlim = [min(Vlim), max(Vlim)]
        ROI = [np.inf, -np.inf]
        for i in range(len(V)):
            if V[i] >= Vlim[0]:
                ROI[0] = min(i, ROI[0])
            if V[i] <= Vlim[1]:
                ROI[1] = max(i, ROI[1])
        # identify best: take few points around minimum of N
        N[N<0] = np.inf					   
        from scipy.signal import medfilt
        N_ = medfilt(N, 3) # thus we eliminate faulty points
        idx = np.argmin(N_[ROI[0]:ROI[1]])
        #print(self.getAttribute('temperature [k]'), [V[ROI[0]+idx+window[0]], V[ROI[0]+idx+window[1]]])
        lim0 = max(ROI[0]+idx+window[0], 0)
        lim1 = min(ROI[0]+idx+window[1], len(V)-1)
        return [V[lim0], V[lim1]]
        
        
    def CurveCV_0V(self, Vtarget=0):
        """
        Creates a curve with require data to compute doping at V=0
        Parameters:
            Vtarget: extract doping around other voltage
        """ 
        if not is_number(Vtarget):
            print('CurveCV.CurveCV_0V: please provide a number')
            return False
        Vtarget
        i = np.argmin(np.abs(self.x() - Vtarget))
        if i > 0 and i < len(self.x()) - 1:
            x = np.concatenate(([0]     , self.x()[i-1:i+2], [0]))
            y = np.concatenate(([np.nan], self.y()[i-1:i+2], [np.nan]))
            curve = CurveCV([x, y], self.attributes)
            curve.update({'linespec':'s', 'markeredgewidth':0, 'labelhide':1})
            return curve
        return False

        
        
    # other functions
    

    def printHelp(self):
        print('*** *** ***')
        print('CurveCV offer basic treatment of C-V curves of solar cells.')
        print('Input units must be [V] and [nF] (or [nF cm-2]).')
        print('Curve transforms:')
        print(' - Linear: standard is Capacitance per area [nF cm-2] vs [V].')
        print(' - Mott-Schottky: [1/(C cm-2)^2] vs V. Allows extraction of built-in voltage V_bi')
        print('   (intercept with y=0) and carrier density (from slope).')
        print(' - Carrier density N_CV [cm-3] vs V: the apparent carrier density is calculated')
        print('   from formula N_CV = -2 / (eps0 epsr d/dV(C^-2)).')
        print(' - Carrier density N_CV [cm-3] vs depth [nm]: the apparent depth is calculated from')
        print('   parallel plate capacitor C = eps0 epsr / d.')
        print('Analysis functions:')
        print(' - Set area: can normalize input data. For proper data analysis the units should be [nF cm-2].')
        print(' - Extract N_CV, Vbi: fits the linear segment on the Mott-Schottky plot.')
        print('   Select the suitable ROI before fitting (enter min and max Voltages).')
        return True
    