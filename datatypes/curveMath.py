# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 15:10:29 2017

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np
from copy import deepcopy

from grapa.curve import Curve
from grapa.graph import Graph
from grapa.mathModule import is_number


class CurveMath(Curve):
    """
    Class handling optical spectra, with notably nm to eV conversion and
    background substraction.
    """
    
    CURVE = 'Curve Math'
    
    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        self.update ({'Curve': CurveMath.CURVE})


    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        listCurves = ['']
        graph = kwargs['graph'] if 'graph' in kwargs else None
        if graph is not None:
            for i in range(kwargs['graph'].length()):
                listCurves.append(str(i) + ' ' + str(kwargs['graph'].curve(i).getAttribute('label'))[0:6])
        fieldprops = {'field':'Combobox', 'width':10, 'values':listCurves, 'bind':'beforespace'}
        default = ''
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
#        out.append([Graph._curveMethod_graphRef, 'Add', ['{this Curve} + cst', '+ Curve_idx'], [0, default], {'curve': self, 'method': 'add', 'operator': 'add'}, [{},fieldprops]]) # one line per function
#        out.append([Graph._curveMethod_graphRef, 'Sub', ['{this Curve} - cst', '- Curve_idx'], [0, default], {'curve': self, 'method': 'add', 'operator': 'div'}, [{},fieldprops]]) # one line per function
#        out.append([Graph._curveMethod_graphRef, 'Mul', ['{this Curve} * cst', '* Curve_idx'], [1, default], {'curve': self, 'method': 'add', 'operator': 'mul'}, [{},fieldprops]]) # one line per function
#        out.append([Graph._curveMethod_graphRef, 'Div', ['{this Curve} / cst', '/ Curve_idx'], [1, default], {'curve': self, 'method': 'add', 'operator': 'div'}, [{},fieldprops]]) # one line per function
        out.append([self.add, 'Add', ['{this Curve} + cst', '+ Curve_idx'], [0, default], {'graph': graph, 'operator': 'add'}, [{},fieldprops]]) # one line per function
        out.append([self.add, 'Sub', ['{this Curve} - cst', '- Curve_idx'], [0, default], {'graph': graph, 'operator': 'sub'}, [{},fieldprops]]) # one line per function
        out.append([self.add, 'Mul', ['{this Curve} * cst', '* Curve_idx'], [1, default], {'graph': graph, 'operator': 'mul'}, [{},fieldprops]]) # one line per function
        out.append([self.add, 'Div', ['{this Curve} / cst', '/ Curve_idx'], [1, default], {'graph': graph, 'operator': 'div'}, [{},fieldprops]]) # one line per function
        out.append([self.neg, '0 - {this Curve} ', [], []]) # one line per function
        out.append([self.inv, '1 / {this Curve} ', [], []]) # one line per function
        out.append([self.swapXY, 'x <-> y', [], []])
        out.append([self.printHelp, 'Help!', [], []]) # one line per function
        return out
    
    @classmethod
    def classNameGUI(cls):
        return cls.CURVE.replace('Curve ', '') + ' operations'

        
    
    def add(self, cst, curves, graph=None, operator='add'):
        
        def op(x, y, operator):
            if operator == 'sub':
                return x - y
            if operator == 'mul':
                return x * y
            if operator == 'div':
                return x / y
            if operator == 'pow':
                return x ** y
            if operator != 'add':
                print('WARNING CureMath.add: unexpected operator argument (' + operator + ').')
            return x + y
        strjoin_ = {'add': ' + ', 'sub': ' - ', 'mul': ' * ', 'div': ' / ', 'pow': ' ** '}
        strjoin = strjoin_[operator] if operator in strjoin_ else ' + '
         # fabricate a copy of the curve
        out = self + 0
        lbl = self.getAttribute('label')
        idx = np.nan
        for c in range(graph.length()):
            if graph.curve(c) == self:
                idx = c
                break
        lst = ['{Curve ' + str(int(idx)) + (': '+lbl if lbl != '' else '') + '}']
        # constants
        if not isinstance(cst, (list, tuple)):
            cst = [cst]
        for c in cst:
            out = op(out, c, operator)
        lst += [str(c) for c in cst]
        # curves
        if graph is not None:
            if not isinstance(curves, (list, tuple)):
                curves = [curves]
            for c in curves:
                if is_number(c):
                    out = op(out, graph.curve(int(c)), operator)
                    lbl = graph.curve(int(c)).getAttribute('label')
                    lst.append('{Curve ' + str(int(c)))
                    if lbl != '':
                        lst[-1] = lst[-1] + ': ' + str(lbl)
                    lst[-1] = lst[-1] + '}'
        txt = strjoin.join(lst)
        math = self.getAttribute('math')
        out.update({'math': math + '; ' + txt if math != '' else txt})
        return out
    
    def neg(self):
        out = 0 - self
        lbl = self.getAttribute('label')
        txt = ' - {Curve' + (': '+lbl if lbl != '' else '') + '}'
        math = self.getAttribute('math')
        out.update({'math': math + '; ' + txt if math != '' else txt})
        return out
        
    def inv(self):
        out = 1 / self
        lbl = self.getAttribute('label')
        txt = '1 / {Curve' + (': '+lbl if lbl != '' else '') + '}'
        math = self.getAttribute('math')
        out.update({'math': math + '; ' + txt if math != '' else txt})
        return out

    def swapXY(self):
        out = 0 + self # work on copy
        out.setX(self.y())
        out.setY(self.x())
        txt = 'swap x<->y'
        math = self.getAttribute('math')
        out.update({'math': math + '; ' + txt if math != '' else txt})
        return out

    def printHelp(self):
        print('*** *** ***')
        print('CurveMath offers some mathematical transformation of the data.')
        print('Functions:')
        print(' - Add. Parameters:')
        print('   cst: a constant, or a list of constants,')
        print('   curves: the index of a Curve, or a list of Curves indices.')
        print(' - Sub. Substraction, similar as Add.')
        print(' - Mul. Multiplication, similar as Add.')
        print(' - Div. Division as Add.')
        print(' - Neg. 0 - Curve.')
        print(' - Inv. 1 / Curve.')
        return True
