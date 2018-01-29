# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 13:49:24 2017

@author: Romain
Copyright (c) 2018, Empa, Romain Carron
"""

import numpy as np
import warnings

from grapa.curve import Curve
from grapa.mathModule import roundSignificant


        
class CurveArrhenius(Curve):
    """
    Offers some support for fitting Arrhenius plots.
    A bit weid implementation: different types of data are handled with
    subclasses, but no instance is ever fabricated of these subclasses.
    Reason is, we don't knwo a priori which subclasse the instance will be,
    and we want to be able to easily switch from on type to the other.
    New types of plots can be implemented as subclasses without modifying the
    CurveArrhenius cain class.
    *** *** ***
    Generally assumes x as temperature [K] and y as a some kind of signal [eV].
    
    By default the output of the fit is E, Offset, with E the energy [eV] and
    Offset the value at 1/T = 0.
    The default fit function is Offset - E/(k*T), with k in [eV/K].
    
    Alternative implementations are:
     - _Arrhenius_variant is set to 'Cfdefects'
       Assumes input as omega [s-1] vs temperature [K].
       Useful to process C-f data. Relevant quantity to plot is
       log(omega * T**-2), and corresponding fit is log(2*ksi) - E /(kT).
    """
    
    CURVE = 'Curve Arrhenius'
    
    k_eVoverK = 8.6173303e-5 # [eV K-1]
    

    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        self.update({'Curve': CurveArrhenius.CURVE})

    
    @classmethod
    def classNameGUI(cls):
        return cls.CURVE.replace('Curve', 'Fit')
    

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
        variant = self.getVariant()
        if self.getAttribute('_fitFunc', None) is None or self.getAttribute('_popt', None) is None:
            out.append([self.CurveArrhenius_fit, variant.BUTTONFitLabel(), [variant.BUTTONFitROI()], [[min(self.x_1000overK())*0.9, 1.1*max(self.x_1000overK())]]])
        else:
            lbl = variant.poptLabel
            param = roundSignificant(self.getAttribute('_popt'), 5)
            out.append([self.updateFitParam, 'Update fit', [lbl[0], lbl[1]], [param[0], '{:1.4e}'.format(param[1])]])
        # switch Arrhenius type
        subclasses = [variant.name, variant.switchModel, '', '== other possibilities ==']
        for child in CurveArrhenius.__subclasses__():
            if hasattr(child, 'name'):
                subclasses.append(child.name)
        out.append([self.setVariant, 'Switch model',
                    ['to model (now '+variant.name+')'],
                    [variant.switchModel], {},
                    [{'field':'Combobox', 'values':subclasses}]])
        # suggest default axis label
        if 'graph' in kwargs:
            alter = kwargs['graph'].getAttribute('alter')
            if alter in ['', ['','']]:
                if hasattr(variant, 'dataLabel'):
                    out.append([kwargs['graph'].updateValuesDictkeys,
                                'Save',
                                ['xlabel', 'ylabel'],
                                variant.dataLabel,
                                {'keys': ['xlabel', 'ylabel']}])
            elif alter == ['CurveArrhenius.x_1000overK', 'CurveArrhenius.y_Arrhenius']:
                if hasattr(variant, 'dataLabelArrhenius'):
                    out.append([kwargs['graph'].updateValuesDictkeys,
                                'Save',
                                ['xlabel', 'ylabel'],
                                variant.dataLabelArrhenius,
                                {'keys': ['xlabel', 'ylabel']}])
                
        out.append([self.printHelp, 'Help!', [], []])
        return out
    
    def alterListGUI(self):
        variant = self.getVariant()
        if hasattr(variant, 'alterListGUIspecific'):
            return getattr(variant, 'alterListGUIspecific')(self)
        out = Curve.alterListGUI(self)
        out += [['Arrhenius vs 1000/T', ['CurveArrhenius.x_1000overK', 'CurveArrhenius.y_Arrhenius'], '']]
        return out
    
    # these one can be overloaded
    @classmethod
    def BUTTONFitLabel(cls):
        return 'Fit Arrhenius'
    @classmethod
    def BUTTONFitROI(cls):
        return 'ROI [1000/K]'
    
    def x_1000overK(self, index=np.nan, **kwargs):
        """ x-axis transform function """
        variant = CurveArrhenius.getVariant(self) # syntax: not necessarily a CurveArrhenius
        if variant != CurveArrhenius and 'x_1000overK' in variant.__dict__: # maybe to do a non-Arrhenius fit?
            return variant.x_1000overK(self, index=index, **kwargs)
        return 1000 / self.x(index, **kwargs)
    
    def y_Arrhenius(self, index=np.nan, **kwargs):
        """ y-axis transform function, getting the Arrhenius straight line """
        variant = CurveArrhenius.getVariant(self) # syntax: not necessarily a CurveArrhenius
        if variant != CurveArrhenius and 'y_Arrhenius' in variant.__dict__: # only if y_Arrhenius explicitely defined, not inherited
            return variant.y_Arrhenius(self, index=index, **kwargs)
        print('CurveArrhenius y_Arrhenius, normally should not come to this point!')
        return self.y(index, **kwargs)
    
    # handling of subclasses of CurveArrhenius
    def getVariant(self):
        var = self.getAttribute('_Arrhenius_variant')
        subclasses = CurveArrhenius.__subclasses__()
        for i in range(len(subclasses)):
            if var == subclasses[i].name:
                return subclasses[i]
        return CurveArrheniusDefault
        
    def setVariant(self, newVariant):
        subclasses = CurveArrhenius.__subclasses__()
        for i in range(len(subclasses)):
            if newVariant == subclasses[i].name:
                self.update({'_Arrhenius_variant': newVariant})
                return True
        if newVariant not in ['', 'CurveArrhenius', 'default']:
            print('ERROR Arrhenius: cannot find model', newVariant,', going for default.')
        self.update({'_Arrhenius_variant': 'default'})
        return True
        
    # functions for Arrhenius plot
    def CurveArrhenius_fit(self, Tlim=None, silent=False):
        """ Returns a Curve based on a fit on Arrhenius plot. """
        variant = self.getVariant()
        popt = self.fit_Arrhenius(Tlim=Tlim, silent=silent)
        attr = {'color': 'k', '_popt': popt, '_fitFunc': 'func_Arrhenius',
                '_Arrhenius_variant': variant.name}
        lbl = variant.poptLabel
        if not silent:
            print('Fit results:', lbl[0],'=', lbl[2].format(popt[0]) + ',',
                                  lbl[1],'=', lbl[3].format(popt[1]) + '.')
        return CurveArrhenius([self.x(), self.func_Arrhenius(self.x(), *popt)], attr)
        
    def _fit_Arrhenius_getData(self, Tlim=None):
        T = self.x_1000overK()
        if Tlim is None:
            Tlim = [min(T), max(T)]
        mask = np.ones(len(self.x()), dtype=bool)
        for i in range(len(mask)):
            if T[i] < Tlim[0] or T[i] > Tlim[1]:
                mask[i] = False
        datax = self.x_1000overK()[mask]
        datay = self.y_Arrhenius()[mask]
        return datax, datay
    def fit_Arrhenius(self, Tlim=None, silent=False):
        """
        Fits the data on the Arrhenius plot, in ROI Tlim[0] to Tlim[1].
        Returns Energy barrier [eV] and prefactor.
        """
        datax, datay = self._fit_Arrhenius_getData(Tlim=Tlim)
        z = np.polyfit(datax, datay, 1, full=True)[0]
        # output
        variant = self.getVariant()
        if not silent:
            print('CurveArrhenius fit variant', variant.name,', assumed input as',
                  variant.dataLabel[1], 'vs', variant.dataLabel[0])
            print(variant.getHelp())
        return variant.popt(z) # [-z[0]*CurveArrhenius.k_eVoverK * 1000, z[1]]
    
    def func_Arrhenius(self, T, E, prefactor):
        """
        Returns function(K) which will appear linear on a Arrhenius plot.
        E is energy barrier
        prefactor is value on plot at 1/T = 0
        """
        variant = self.getVariant()
        if hasattr(variant, 'func_Arrhenius'):
            return variant.func_Arrhenius(T, E, prefactor)
        print('CurveArrhenius func_Arrhenius, normally should not come to this point!')
        return prefactor - E / (CurveArrhenius.k_eVoverK * T)
            

    def printHelp(self):
        print('*** *** ***')
        print('Curve Arrhenius general help:')
        print('Offers some support for fitting Arrhenius plots.')
        print('Generally assumes x as temperature [K] and y as a some kind of signal [eV].')
        print('By default the output of the fit is E, Offset, with E the energy [eV] and Offset the value at 1/T = 0.')
        print('The default fit function is Offset - E/(k*T), with k in [eV/K].')
        print('')
        print('List of alternative implementations:')
        dct = {}
        variant = self.getVariant()
        subclasses = CurveArrhenius.__subclasses__()
        for cls in subclasses:
            if cls != variant:
                dct.update({cls.name: cls})
        lst = [key for key in dct]
        lst.sort()
        for key in lst:
            print(key, ':', dct[key].getHelp(short=True))
        print('')
        print('Help on this specific implementation of Arrhenius Curve:')
        print(variant.getHelp())
        dataLabel = self.getVariant().dataLabel
        print('Expected data input:', dataLabel[0], ',', dataLabel[1])
        return True


        
        
class CurveArrheniusDefault(CurveArrhenius):
    """
    The instances will be of CurveArrhenius, but main class will call the
    methods implemented in the desired child classes.
    """
    name = 'default' # identifier
    dataLabel = ['Temperature [K]', 'Value [eV]']
    poptLabel = ['E [eV]', 'Offset [eV]', '{:1.4e}', '{:.4f}']
    switchModel = '' # sggestion to user, a other model to try on his data
    # default Curve attributes such that the CurveArrhenius is correctly defined
    attr = {'Curve': 'Curve Arrhenius', '_Arrhenius_variant': 'default',
            'label': 'value vs Temperature'}
    def popt(z): # transforms polynom z into correct fit parameters
        return [-z[0]*CurveArrhenius.k_eVoverK * 1000, z[1]]
    def getHelp(short=False): # the help dedicated to the specific implementation
        return 'Arrhenius plot is: value = offset - E /(kT)'
    def y_Arrhenius(self, index=np.nan, **kwargs):
         return self.y(index, **kwargs)
    def func_Arrhenius(T, E, prefactor):
        return prefactor - E / (CurveArrhenius.k_eVoverK * T)
    
class CurveArrheniusJscVocJ00(CurveArrhenius):
    name = 'JscVocJ00'
    dataLabel = ['A(T) * Temperature [K]', 'J0 [mA cm-2]']
    poptLabel = ['E_A [eV]', 'J00 [mA cm-2]', '{:1.4f}', '{:1.4f}']
    switchModel = 'JscVocJ00'
    attr = {'Curve': 'Curve Arrhenius', '_Arrhenius_variant': 'JscVocJ00',
            'label': 'J0 [mA cm-2] vs A(t)*T [K]',
            'linestyle': 'none', 'linespec': 'o'}
    def popt(z): # fit polynom z to desired output values
        return [-z[0]*CurveArrhenius.k_eVoverK * 1000, np.exp(z[1])]
    def func_Arrhenius(T, E, prefactor): # replace default fit function
        return np.exp(np.log(prefactor) - E / (CurveArrhenius.k_eVoverK * T))
    def y_Arrhenius(self, index=np.nan, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = np.log(self.y(index=index, **kwargs))
        return out
    def getHelp(short=False):
        out = 'Computes E_A the activation energy of J0, and J00 its value extrapolated at 1/T = 0.'
        if not short:
            out+= '\nThe equation is: log(J0) = log(J00) - E_A / (k AT),\n'
            out+= '   stemming from J0 = J00 * exp(- E_A / (k AT))'
        return out


class CurveArrheniusCfDefects(CurveArrhenius):
    """ 'fit': 'ln2ksi-E/kT', 'y': 'ln(omega*T**-2)' """
    name = 'Cfdefects'
    dataLabel = ['Temperature [K]', 'omega [s-1]']
    poptLabel = ['E [eV]', 'ksi', '{:1.5f}', '{:.4e}']
    switchModel = 'Cfdefault'
    attr = {'Curve': 'Curve Arrhenius', '_Arrhenius_variant': 'Cfdefects',
            'label': 'omega vs Temperature', 'linestyle': 'none'}
            # by default hidden - would cause strange value on derivative plot
    dataLabelArrhenius = [r'1000 / T', r'ln($\omega T^{-2}$)']
    def popt(z):
        return [-z[0] * CurveArrhenius.k_eVoverK * 1000, 0.5 * np.exp(z[1])]
    def func_Arrhenius(T, E, prefactor):
        return T**2 * np.exp(np.log(2 * prefactor) - E / (CurveArrhenius.k_eVoverK * T))
    def y_Arrhenius(self, index=np.nan, **kwargs):
        # must compute omega T**-2 and show log, corresponding fit expression is 'log2ksi-omega/kT'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = np.log(self.y(index, **kwargs) * self.x(index, **kwargs)**-2)
        return out
    def getHelp(short=False):
        if short:
            return 'Handles omega [s-1] vs T [K] inflection points of C-f curves, using a weak T**-2 temperature dependency.'
        out = 'Assumes traps can follow frequency: omega = 2 ksi T**2 exp(-E/(kT))\n'
        out+= '    then Arrhenius plot is: ln(omega * T**-2) = ln(2*ksi) - E /(kT)'
        return out

     
class CurveArrheniusCfdefault(CurveArrhenius):
    """ 'fit': 'ln2ni-E/kT', 'y': 'ln(omega)' """
    name = 'Cfdefault'
    dataLabel = ['Temperature [K]', 'omega [s-1]']
    poptLabel = ['E [eV]', 'nu', '{:1.5f}', '{:.4e}']
    switchModel = 'Cfdefects'
    attr = {'Curve': 'Curve Arrhenius', '_Arrhenius_variant': 'Cfdefault',
            'label': 'omega vs Temperature', 'linestyle': 'none'}
            # by default hidden - would cause strange value on derivative plot
    dataLabelArrhenius = [r'1000 / T', 'ln($\omega$)']
    def popt(z):
        return [-z[0] * CurveArrhenius.k_eVoverK * 1000, 0.5 * np.exp(z[1])]
    def func_Arrhenius(T, E, prefactor):
        return np.exp(np.log(2 * prefactor) - E / (CurveArrhenius.k_eVoverK * T))
    def y_Arrhenius(self, index=np.nan, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = np.log(self.y(index, **kwargs))
        return out
    def getHelp(short=False):
        if short:
            return 'Handles omega [s-1] vs T [K] inflection points of C-f curves, using no weak T**-2 temperature dependency.'
        out = 'Assumes a (barrier / trap / something without weak T dependency) can follow frequency:\n'
        out+= 'omega = 2 nu exp(-E/(kT))\n'
        out+='    then Arrhenius plot is: ln(omega) = ln(2*nu) - E /(kT)'
        return out


class CurveArrheniusExtrapolToZero(CurveArrhenius):
    """
    This is not an Arrhenius, but we can recyle the same code for any 1-degree
    polynomial interpolation.
    """
    @classmethod
    def BUTTONFitLabel(cls):
        return 'Fit extrapolate to 0'
    @classmethod
    def BUTTONFitROI(cls):
        return 'ROI [K]'
    name = 'ExtrapolationTo0' # is NOT an Arrhenius fit, still, can use polynom fit
    dataLabel = ['Temperature [K]', 'Value [unit]']
    poptLabel = ['Increase [unit / K]', 'Value at 0 [unit]', '{:.4e}', '{:.4e}']
    switchModel = ''
    attr = {'Curve': 'Curve Arrhenius', '_Arrhenius_variant': 'ExtrapolationTo0'}
    dataLabelArrhenius = ['Temperature [K]', 'Value [unit]']
    def popt(z):
        return [z[0], z[1]]
    def func_Arrhenius(T, E, prefactor):
        return prefactor + E * T
    def y_Arrhenius(self, index=np.nan, **kwargs):
        return self.y(index, **kwargs)
    def x_1000overK(self, index=np.nan, **kwargs): # NOT ARRHENIUS
        return self.x(index, **kwargs)
    def getHelp(short=False):
        out = '(not Arrhenius relation) Fits the data with 1-degree polynom.'
        if short:
            return out
        out += '\nThe equation is: value = ValueAt0 + Increase * T'
        return out
    

class CurveArrheniusExpDecay(CurveArrhenius):
    """
    This is not an Arrhenius, but we can recyle the same code for another
    expression
    Created to handle Urbach decays of EQE curves 
    """
    @classmethod
    def BUTTONFitLabel(cls):
        return 'Fit exp. decay'
    @classmethod
    def BUTTONFitROI(cls):
        return 'ROI [eV]'
    name = 'expDecay' # is NOT an Arrhenius fit, still, can use polynom fit
    dataLabel = ['Wavelength [nm]', 'Value [unit]']
    poptLabel = ['Decay energy [meV]', 'Energy @ 100% [eV]', '{:.4e}', '{:.4e}']
    switchModel = ''
    attr = {'Curve': 'Curve Arrhenius', '_Arrhenius_variant': 'expDecay'}
    dataLabelArrhenius = ['Energy [eV]', 'log(value)']
    def alterListGUIspecific(self):
        out = Curve.alterListGUI(self)
        return out
    def popt(z):
        return [1/z[0]*1000, -z[1]/z[0]]
    def func_Arrhenius(nm, U, E_at_1):
        return np.exp(((Curve.NMTOEV / nm) - (E_at_1)) / (U*0.001))
    def y_Arrhenius(self, index=np.nan, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.log(self.y(index, **kwargs))
    def x_1000overK(self, index=np.nan, **kwargs): # NOT ARRHENIUS
        return self.x(index, alter='nmeV', **kwargs)
    def getHelp(short=False):
        out = 'Characterize an exponential decay (ie. Urbach).'
        if short:
            return out
        out += '\nThe equation is: value = exp((E - E_at_1) / (E_decay/1000))'
        return out
    
