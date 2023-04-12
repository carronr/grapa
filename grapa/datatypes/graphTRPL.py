# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

from os import path as ospath

from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.datatypes.curveTRPL import CurveTRPL


class GraphTRPL(Graph):

    FILEIO_GRAPHTYPE = 'TRPL decay'

    AXISLABELS = [['Time', 't', 'time'], ['Intensity', '', 'counts']]

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='',
                       **kwargs):
        if fileExt == '.dat' and line1 == 'Time[ns]	crv[0] [Cnts.]':
            return True
        elif (fileExt == '.dat'
                and line1 == 'Parameters:'
                and ((      line2.strip().startswith('Sample')
                        and line3.strip().startswith('Solvent'))
                     or (   line2.strip().split(' : ')[0] in ['Exc_Wavelength']
                        and line3.strip().split(' : ')[0] in ['Exc_Bandpass'])
                     )
              ):
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        """ Read a TRPL decay. """
        len0 = len(self)
        kw = {}
        if 'line1' in kwargs and kwargs['line1'] == 'Parameters:':
            kw.update({'delimiterHeaders': ' : '})
        GraphIO.readDataFromFileGeneric(self, attributes, **kw)
        self.castCurve(CurveTRPL.CURVE, len0, silentSuccess=True)
        # label management
        filenam_, fileext = ospath.splitext(self.filename)  # , fileExt
        # self.curve(len0).update({'label': filenam_.split('/')[-1].split('\\')[-1]})
        lbl = filenam_.split('/')[-1].split('\\')[-1].replace('_', ' ').split(' ')
        smp = str(self[len0].attr('sample'))
        try:
            if float(int(float(smp))) == float(smp):
                smp = str(int(float(smp)))
        except Exception:
            pass
        smp = smp.replace('_', ' ').split(' ')
        new = lbl
        if len(smp) > 0:
            new = [l for l in lbl if l not in smp] + smp
        # print('label', self.attr('label'), [l for l in lbl if l not in smp], smp)
        self[len0].update({'label': ' '.join(new)})
        # clean values
        for key in self[len0].getAttributes():
            val = self[len0].attr(key)
            if isinstance(val, str) and 'Â°' in val:
                self[len0].update({key: val.replace('Â°', '°')})
        # graph porperties
        xlabel = self.attr('xlabel').replace('[', ' [').replace('  ', ' ').capitalize()  # ] ]
        if xlabel in ['', ' ']:
            xlabel = GraphTRPL.AXISLABELS[0]
        self.update({'typeplot': 'semilogy', 'alter': ['', 'idle'],
                     'xlabel': self.formatAxisLabel(xlabel),
                     'ylabel': self.formatAxisLabel(GraphTRPL.AXISLABELS[1])})
        self.update({'subplots_adjust': [0.2, 0.15]})
        # cleaning
        if 'line1' in kwargs and kwargs['line1'] == 'Parameters:':
            attr = self[len0].getAttributes()
            keys = list(attr.keys())
            for key in keys:
                val = attr[key]
                if isinstance(val, str) and val.startswith('\t'):
                    self[len0].update({key: ''})
