# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

from re import findall as refindall
import numpy as np

from grapa.mathModule import is_number, strToVar
from grapa.graph import Graph
from grapa.datatypes.curveMCA import CurveMCA


class GraphMCA(Graph):

    FILEIO_GRAPHTYPE = 'XRF MCA raw data'

    AXISLABELS = [['XRF detector channel', '', ' '],
                  ['Counts per s', '', 'cps']]

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        if fileExt == '.mca' and line1 == '<<PMCA SPECTRUM>>':
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        def categ(line, old):
            if line.startswith('<<'):
                return line.replace('<', '').replace('>', '')
            return old

        # parse file
        y = np.array([])
        line = ''
        category = ''
        with open(self.filename, 'r') as file:
            # first section
            while line != '<<DATA>>':
                line = file.readline().strip(' \r\n\t')
                category = categ(line, category)
                split = refindall('([^-]+) - ([^-]+)', line)
                if len(split) > 0 and len(split[0]) > 1:
                    at = category+'.'+split[0][0]
                    attributes.update({at: strToVar(split[0][1])})
            # data
            flag = True
            while flag:
                line = file.readline().strip(' \r\n\t')
                if is_number(line):
                    y = np.append(y, float(line))
                else:
                    category = categ(line, category)
                    flag = False
            # parameters at end
            while 1:
                line = file.readline().strip(' \r\n\t')
                if not line:
                    break
                category = categ(line, category)
                expr = r'([^0-9]+): ([-+]?[0-9]*\.?[0-9]+) *([^0-9]*)'
                split = refindall(expr, line)
                if len(split) == 0:
                    expr = r'([^0-9]+): (\w+)()'
                    split = refindall(expr, line)
                if len(split) > 0 and len(split[0]) > 2:
                    tmp = split[0][2]
                    if len(split[0][2]) > 0:
                        tmp = ' ['+split[0][2]+']'
                    at = category + '.' + split[0][0] + tmp
                    attributes.update({at: strToVar(split[0][1])})
                else:
                    pass  # not interesting input (section title, etc.)
            # close file at en of with section
        # format data, create Curve
        data = np.append(np.arange(y.size), y).reshape((2, y.size))
        at = 'DPP CONFIGURATION.Preset'
        if at+' [sec]' not in attributes and at+' [min]' in attributes:
            attributes[at+' [sec]'] = attributes[at+' [min]'] * 60
        self.append(CurveMCA(data, attributes))
        self[-1].update({'muloffset': "1/" + str(self.attr(at+' [sec]'))})
        if self[-1].attr('sample') == '':
            self[-1].update({'sample': self[-1].attr('label')})
        # graph cosmetics
        self.update({'collabels': ['Channel [ ]', 'Counts [ ]']})
        self.update({'xlabel': self.formatAxisLabel(GraphMCA.AXISLABELS[0]),
                     'ylabel': self.formatAxisLabel(GraphMCA.AXISLABELS[1])})
        if y.size == 1024:
            self.update({'xlim': [0, 1024]})
        print('Opened MCA raw data file. To know composition, open .html file',
              'instead.')
