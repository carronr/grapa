# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:44:58 2016

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

import os
import numpy as np
import copy

from grapa.curve import Curve
from grapa.mathModule import is_number


class CurveMCA(Curve):

    CURVE = 'Curve MCA'

    PEAKS_FILENAME = 'XRF_photonenergiesintensities.txt'
    PEAKS_DATA = None
    PEAKS_ELEMENTS = None
    PEAKS_LINES = None

    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        self.update({'Curve': CurveMCA.CURVE})
        self.update({'_MCA_CtokeV_offset': -3.35,
                     '_MCA_CtokeV_mult': 0.031490109515627})
        if CurveMCA.PEAKS_DATA is None:
            CurveMCA.peaks_data()

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        at = ['_MCA_CtokeV_offset', '_MCA_CtokeV_mult']
        out.append([self.updateValuesDictkeys, 'Save',
                    ['keV = (channel +', ') * '],
                    [self.attr(a) for a in at], {'keys': at}])
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
        # out.append([self.dataModifySwapChannelkeV, 'change data Channel<->keV', [], []]) # one line per function
        self.peaks_data()
        if CurveMCA.PEAKS_DATA is not None:
            elements = self.PEAKS_ELEMENTS
            lines = ['all'] + self.PEAKS_LINES
            addrem = ['add', 'remove']
            out.append([self.addElementLabels, 'Save',
                        ['', 'element', 'series', 'mult.', 'vline bottom'],
                        [addrem[0], elements[0], lines[0], 0.1, 0],
                        {'graph': kwargs['graph']},
                        [{'field': 'Combobox', 'width': 5, 'values': addrem},
                         {'field': 'Combobox', 'width': 5, 'values': elements},
                         {'field': 'Combobox', 'width': 4, 'values': lines},
                         {}, {}]])
        out.append([self.printHelp, 'Help!', [], []])  # one line per function
        return out

    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        out += [['Channel <-> keV', ['MCAkeV', ''], '']]
        return out

    @classmethod
    def peaks_data(cls):
        if cls.PEAKS_DATA is None:
            path = os.path.dirname(os.path.abspath(__file__))
            data = np.genfromtxt(os.path.join(path, cls.PEAKS_FILENAME),
                                 dtype=None, skip_header=3, delimiter='\t')
            elements, eldict = [], {}
            lines = []
            for row in data:
                if int(row[1]) not in eldict:
                    eldict[int(row[1])] = row[2].decode('UTF-8') + ' ' + str(row[1])
                tmp = row[3].decode('UTF-8')[0]
                if tmp not in lines:
                    lines.append(tmp)
            m = np.max(list(eldict.keys())) + 1
            for key in range(m):
                if key in eldict:
                    elements.append(eldict[key])
            lines.sort()
            cls.PEAKS_DATA = data
            cls.PEAKS_ELEMENTS = elements
            cls.PEAKS_LINES = lines
        return cls.PEAKS_DATA

    def addElementLabels(self, addrem, element, line, multiplier=1, vlinebottom=0, **kwargs):
        """
        Add labels for transition for a given element, according to the
        database https://xdb.lbl.gov/Section1/Table_1-3.pdf
        addrem: 'add', or 'remove' sets of lines.
        element: text (str starting with element 2-letter), or numeric (Z
            number). Possible values: 'Cu', 'Zn 30', 25, etc.
        line: 'K', 'M, 'L', 'all'
        multiplier: multiplier to the tabulated peak intensity. Default 1.
        vlinebottom: value at which the vertical line ends. Default 0.
        """
        to_keV = 0.001
        if 'graph' not in kwargs:
            print('CurveMCA addElementLabels expect "graph" in kwargs')
            return False
        graph = kwargs['graph']
        data = self.peaks_data()

        def textformat(element, line):
            line = line.replace('a', r'$\alpha$').replace('b', r'$\beta$').replace('g', r'$\gamma$')
            # line = line.replace('a', r'$\alpha$').replace('b', r'$\beta$').replace('g', r'$\gamma$')
            if '$' in line and line[-1] != '$':  # indices
                tmp = line.split('$')
                tmp[-1] = '$_{' + tmp[-1] + '}$'
                line = '$'.join(tmp)
                line = line.replace('$$', '')
            return element + ' ' + line

        def removeTextStartswith(start):
            if start == '':
                return
            graph.checkValidText()
            texts = graph.attr('text', None)
            idxdel = []
            if texts is not None:
                for i in range(len(texts)):
                    if texts[i].startswith(start):
                        idxdel.append(i)
            for i in idxdel[::-1]:
                # print('remove text i', i, texts[i])
                graph.removeText(i)
            return True

        if is_number(element):
            element = int(element)
            for row in data:
                if int(row[1]) == element:
                    element = row[2].decode('UTF-8')
                    break
        if is_number(element):
            print('CurveMCA addElementLabels cannot find data for element',
                  element)
            return False
        element = element.split(' ')[0]
        # if user wants to remove some text
        if addrem == 'remove':
            start = element
            if line != 'all':
                start += ' ' + line  # also safe vs $
            removeTextStartswith(start)
        elif self.PEAKS_DATA is not None:
            # if add text
            lines = self.PEAKS_LINES if line == 'all' else [line]
            adds = []
            argdef = {'verticalalignment': 'bottom', 'annotation_clip': False,
                      'horizontalalignment': 'center', 'textcoords': 'data',
                      'arrowprops': {'headwidth': 0, 'facecolor': 'k',
                                     'width': 0, 'shrink': 0,
                                     'set_clip_box': True}}
            for row in data:
                el = row[2].decode('UTF-8')
                li = row[3].decode('UTF-8')
                if el == element and li[0] in lines:
                    text = textformat(row[2].decode('UTF-8'), li)
                    textxy = [row[0] * to_keV, row[4] * multiplier]
                    args = copy.deepcopy(argdef)
                    args['xy'] = [row[0]*to_keV, vlinebottom]
                    adds.append([text, textxy, args])
            for add in adds:
                # print('add text', add[0])
                removeTextStartswith(add[0])
                graph.addText(add[0], add[1], textargs=add[2])
        return True

    def printHelp(self):
        print('*** *** ***')
        print('CurveMCA offers capabilities to display raw XRF data.')
        print('Data are automatically given a multiplicative offset',
              '1/acquisition time.')
        print('Curve transforms:')
        print('- Channel <-> eV: switch [channel] data into keV representation',
              'based on properties _MCA_CtokeV_offset and _MCA_CtokeV_mult.')
        print('Analysis functions:')
        self.printHelpFunc(CurveMCA.addElementLabels)
