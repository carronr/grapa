# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:38:53 2017

@author: Romain
Copyright (c) 2018, Empa, Romain Carron
"""

import numpy as np
from grapa.curve import Curve

class CurveMycurve(Curve):

    # so the Curve class information is retrieved when opening an existing file
    CURVE = 'Curve mycurve'
    
    def __init__(self, data, attributes, silent=False):
        """
        Constructor with minimal structure: Curve.__init__, and then set the
        'Curve' parameter.
        """
        Curve.__init__(self, data, attributes, silent=silent)
        self.update ({'Curve': CurveMycurve.CURVE})


    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        """
        Fills in the Curve actions specific to the Curve type. Syntax:
        [func,
         'Button text',
         ['label 1', 'label 2', ...],
         ['value 1', 'value 2', ...],
         {'hidden': 'value'}, (optional)
         [dictFieldAttributes, {}, ...]] (optional)
        """
        out = Curve.funcListGUI(self, **kwargs)
        # already implemented easy access to Curve attributes
        at = ['label', 'color']
        out.append([self.updateValuesDictkeys, 'Set', at,
                    [self.getAttribute(a) for a in at], {'keys': at}])
        # some function
        choices =['min, max','number of points','just make a copy and add 0.1']
        out.append([self.statsAndAddition,
                    'Go',
                    ['Do some statistics'],
                    ['min, max'],
                    {},
                    [{'field':'Combobox', 'values': choices}]])
        out.append([self.printHelp, 'Help!', [], []])
        return out
    
    def alterListGUI(self):
        """
        Determines the possible curve visualisations. Syntax:
        ['GUI label', ['alter_x', 'alter_y'], 'semilogx']
        A few alter keywords are available but feel free to implement our own
        """
        out = Curve.alterListGUI(self)
        out.append(['derivative', ['', 'CurveMycurve.y_derivative'], ''])
        out.append(['log(abs(y))', ['', 'CurveMycurve.y_abs'], 'semilogy'])
        return out

    # data transform function
    def y_derivative(self, index=np.nan, xyValue=None):
        """
        Function computing the derivative on-the-fly during plotting.
        The same function is used to modify the graph xlim and ylim. In such
        cases the limits are passed through the xyValue.
        """
        # ylim cannot be computed from the initial limit values (derivative)
        if xyValue is not None:
            return np.nan
        # do not want to use self.x(index) syntax as we need more points to 
        # compute the local derivative. Here we compute over all data, then
        # restrict to the desired datapoint
        x, y = self.x(xyValue=xyValue), self.y(xyValue=xyValue)
        # compute derivative as symmetrical finite difference
        # for noisy data have a look to Savitsky-Golay or similar
        val = [(y[1]  - y[0]  ) / (x[1]  - x[0])]
        val+= list((y[2:] - y[:-2]) / (x[2:] - x[:-2]))
        val+= [(y[-2] - y[-1] ) / (x[-2] - x[-1])]
        # NB: This formula might not be valid if length is smaler than 2
        val = val[:len(x)]
        # restrict over required index datapoints
        if len(val) > 0:
            if np.isnan(index).any():
                return val[:]
            return val[index]
        return val
        
    # data transform function
    def y_abs(self, index=np.nan, xyValue=None):
        if xyValue is not None:
            # this way a negative ylim value can be ignored
            return np.abs([max(0.0, v) for v in xyValue[1]])
        return np.abs(self.y(index=index, xyValue=xyValue))
    
    # function called by user
    def statsAndAddition(self, what):
        if what == 'min, max':
            print('Values min', np.min(self.y()), ', max', np.max(self.y()))
        elif what == 'number of points':
            print('Number of points', len(self.y()))
        elif what == 'just make a copy and add 0.1':
            # return a copy of self after adding 1 to the y values
            out = self + 0.1 # basic operations are implemented and affect y
            out.update({'label': out.getAttribute('label')+' + 0.1'})
            return out
        else:
            print('Unknown keyword', what)
        return True
        
        
    def printHelp(self):
        print('*** *** ***')
        print('CurveMycurve is there just for show.')
        print('Maybe here some help for the user')
        return True
