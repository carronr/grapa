# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:19:52 2017

@author: Romain Carron
Copyright (c) 2023, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

# from grapa.graph import Graph
from grapa.curve import Curve


class Curve_Inset(Curve):
    """
    The purpose is this class is to provid GUI support to create insets
    """

    CURVE = 'inset'

    def __init__(self, *args, **kwargs):
        Curve.__init__(self, *args, **kwargs)
        # define default values for important parameters
        if self.attr('insetfile') == '':
            self.update({'insetfile': ''})
        if self.attr('insetcoords') == '':
            self.update({'insetcoords': [0.3, 0.2, 0.4, 0.3]})
        if self.attr('insetupdate') == '':
            self.update({'insetupdate': {}})

        if self.attr('insetfile') == ' ':
            if self.attr('subplotfile') not in ['', ' ']:
                self.update({'insetfile': self.attr('subplotfile')})
                self.update({'subplotfile': ''})

        self.update({'Curve': Curve_Inset.CURVE})

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        out.append([self.updateValuesDictkeys, 'Set', ['file inset'],
                    [self.attr('insetfile')], {'keys': ['insetfile']}])
        out.append([self.updateValuesDictkeys, 'Set',
                    ['coords, in figure fraction [left, bottom, width, height]'],
                    [self.attr('insetcoords')], {'keys': ['insetcoords']}])
        out.append([self.updateValuesDictkeys, 'Set', ['update inset'],
                    [self.attr('insetupdate')], {'keys': ['insetupdate']}])
        out.append([self.printHelp, 'Help!', [], []])  # one line per function
        return out

    def printHelp(self):
        print('*** *** ***')
        print('Class Curve_Inset facilitates the creation and customization',
              'of insets inside a Graph.')
        print('Important parameters:')
        print('- insetfile: a path to a saved Graph, either absolute or',
              'relative to the main Graph.')
        print('  if set to ' ' (whitespace) or if no Curve are found in the',
              'given file, the next Curves of the main graph will be',
              'displayed in the inset.')
        print('- insetcoords: coordinates of the inset axis, relative to the',
              'main graph. Prototype: [left, bottom, width, height]')
        print('- insetupdate: a dict which will be applied to the inset',
              'graph. Provides basic support for run-time customization of',
              'the inset graph.')
        print("  Examples: {'fontsize':8}, or {'xlabel': 'An updated label'}")
        return True
