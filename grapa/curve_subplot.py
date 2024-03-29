# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:03:50 2017

@author: Romain Carron
Copyright (c) 2023, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

from grapa.graph import Graph
from grapa.curve import Curve


class Curve_Subplot(Curve):
    """
    The purpose is this class is to provid GUI support to handle subplots
    """

    CURVE = 'subplot'

    def __init__(self, *args, **kwargs):
        Curve.__init__(self, *args, **kwargs)
        # define default values for important parameters
        self.update({'Curve': Curve_Subplot.CURVE})

        def checkattr(key, default, typ_):
            val = self.attr(key)
            if val == '':
                self.update({key: default})
            elif not isinstance(val, typ_):
                self.update({key: typ_(val)})

        checkattr('subplotfile', ' ', str)
        checkattr('subplotrowspan', 1, int)
        checkattr('subplotcolspan', 1, int)
        checkattr('subplotupdate', {}, dict)
        if self.attr('subplotfile') == ' ':
            if self.attr('insetfile') not in ['', ' ']:
                self.update({'subplotfile': self.attr('insetfile')})
                self.update({'insetfile': ''})

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        graph = None
        if 'graph' in kwargs:
            graph = kwargs['graph']
            del kwargs['graph']
        out.append([self.updateValuesDictkeys, 'Set', ['file subplot'],
                    [self.attr('subplotfile')], {'keys': ['subplotfile']}])
        dictspan = {'field': 'Combobox', 'bind': 'beforespace',
                    'values': ['', '1', '2', '3', '4', '1 number']}
        out.append([self.updateValuesDictkeys, 'Set', ['colspan', 'rowspan'],
                    [self.attr('subplotcolspan'), self.attr('subplotrowspan')],
                    {'keys': ['subplotcolspan', 'subplotrowspan']},
                    [dictspan, dictspan]])
        out.append([self.updateValuesDictkeys, 'Set', ['update subplot'],
                    [self.attr('subplotupdate')],  {'keys': ['subplotupdate']}])
        if graph is not None and isinstance(graph, Graph):
            out.append([None, 'Graph options', [], []])
            ncols = int(graph.attr('subplotsncols', 2))
            out.append([graph.updateValuesDictkeys, 'Set',
                        ['n cols', 'transpose', 'show id? or ('+u"\u0394x,\u0394y)"],
                        [ncols, str(graph.attr('subplotstranspose')), str(graph.attr('subplotsid'))],
                        {'keys': ['subplotsncols', 'subplotstranspose', 'subplotsid']},
                        [{'field': 'Combobox', 'values': ['', '1', '2', '3', '4', '1 number']},
                         {'field': 'Combobox', 'values': ['', '0 False', '1 True'], 'bind':'beforespace'},
                         {}]])
            out.append([graph.updateValuesDictkeys, 'Set',
                        ['width ratios (i.e. [1,2,1])', 'height_ratios'],
                        [graph.attr('subplotswidth_ratios'), graph.attr('subplotsheight_ratios')],
                        {'keys': ['subplotswidth_ratios', 'subplotsheight_ratios']}])
            out.append([graph.updateValuesDictkeys, 'Set',
                        ['subplots_adjust [left, bottom, right, top, wspace, hspace]'],
                        [graph.attr('subplots_adjust')],
                        {'keys': ['subplots_adjust']}])
        out.append([self.printHelp, 'Help!', [], []])  # one line per function
        return out

    def printHelp(self):
        print('*** *** ***')
        print('Class Curve_Subplot facilitates the creation and customization',
              'of subplots inside a Graph.')
        print('The main graph will be subdivised in an array of subplots.')
        print('Important parameters:')
        print('- subplotfile: saved Graph file which must be shown in'
              'subplot. The next Curves will also be shown in the created',
              'axis, until the next Curve with same Curve_Subplot type.')
        print('- subplotcolspan: on how many column the subplot will be',
              'plotted.')
        print('- subplotrowspan: on how many rows the subplot will be',
              'plotted.')
        print('- subplotupdate: a dict which will be applied to the subplot',
              'Graph. Provides basic support for run-time customization of',
              'the inset graph.')
        print("Examples: {'fontsize': 8}, or {'xlabel': 'An updated label'}")
        print()
        return True
