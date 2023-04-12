# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 19:49:25 2018

@author: Romain
"""

from os import path as ospath
import numpy as np

from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.datatypes.curveSpectrum import CurveSpectrum


class GraphSpectrum(Graph):
    """
    This is a generic class to open spectrum-like files. The purpose is to
    limit the endless multiplication of files in folder datatypes.
    Each child class of GraphSpectrum are constructed the same way as similar
    classes.
    Each types of
    """

    FILEIO_GRAPHTYPE = 'Optical spectrum'

    AXISLABELS = [['Wavelength', '\lambda', 'nm'],
                  ['Intensity', '', 'counts']]

    WHICHSUBCLASS = []

    @classmethod
    def isFileReadable(cls, fileName, fileExt, **kwargs):
        # ask every child class to know if it can open the file
        # if yes, remembers which one can
        GraphSpectrum.WHICHSUBCLASS = []
        for child in GraphSpectrum.__subclasses__():
            if child.isFileReadable(fileName, fileExt, **kwargs):
                GraphSpectrum.WHICHSUBCLASS.append(child)
        if len(GraphSpectrum.WHICHSUBCLASS) > 0:
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        # ask the child class which said they can, to open the file
        # By design readDataFromFile shall be called immediately after
        # isFileReadable, so the design is ~safe. Maybe we should explicitely
        # perform the readable test again?
        for child in GraphSpectrum.WHICHSUBCLASS:
            res = child.readDataFromFile(self, attributes, **kwargs)
            if res is None or res != False:
                return True
        return False


class GraphSpectrumSpectraSuite(GraphSpectrum):
    """ Reads a HR2000 file (optical fiber spectrophotometer). """
    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        if fileExt == '.txt' and line1 == 'SpectraSuite Datei':
            return True

    def readDataFromFile(self, attributes, **kwargs):
        filenam_, fileext = ospath.splitext(self.filename)  # , fileExt
        # TODO:  improve header parsing
        self.append(CurveSpectrum(np.transpose(np.genfromtxt(self.filename, skip_header=17, delimiter='\t', invalid_raise=False)), attributes))
        # BG = 1140 # seems not generally valid
        BG = 0
        self[-1].setY(self[-1].y() - BG)
        self[-1].update({'label': filenam_.split('/')[-1]})
        self.headers.update({'collabels': ['Wavelength [nm]', 'Intensity [counts]']})
        self.update({'xlabel': self.formatAxisLabel(GraphSpectrum.AXISLABELS[0]),
                     'ylabel': self.formatAxisLabel(GraphSpectrum.AXISLABELS[1])})


class GraphSpectrumUVVIS(GraphSpectrum):
    """ Reads a ascii file from Shimadzu UVVIS 3600 """
    AXISLABELS = [['Wavelength', '', 'nm'], GraphSpectrum.AXISLABELS[1]]

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        if (fileExt == '.txt'
                and line1.startswith('"') and line1.endswith('"')
                and line2.startswith('"Wavelength nm."	"')
                and line2.endswith('%"')):
            return True

    def readDataFromFile(self, attributes, **kwargs):
        GraphIO.readDataFromFileGeneric(self, attributes)
        sub = self[-1].attr('"Wavelength nm."')
        lbl = self[-1].attr('label')
        ylabel = GraphSpectrumUVVIS.AXISLABELS[1]
        if sub == 'R%':
            sub = 'CurveSpectrumReflectance'
            ylabel = ['Reflectance', '', '%']
            if lbl.endswith(' R R%'):
                lbl = lbl[:-3]+'%'
        elif sub == 'T%':
            sub = 'CurveSpectrumTransmittance'
            ylabel = ['Transmittance', '', '%']
            if lbl.endswith(' T T%'):
                lbl = lbl[:-3]+'%'
        else:
            sub = ''
        self[-1].update({'_spectrumSubclass': sub, '_spectrumunit': '%'})
        self.castCurve('Curve Spectrum', len(self)-1, silentSuccess=True)
        self[-1].update({'label': lbl, '"Wavelength nm."': ''})
        self.update({'xlabel': self.formatAxisLabel(GraphSpectrumUVVIS.AXISLABELS[0]),
                     'ylabel': self.formatAxisLabel(ylabel)})


class GraphSpectrumTRPLsetup(GraphSpectrum):
    """ TRPL setup PL spectrum """
    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        if fileExt == '.dat' and line1[:29] == 'Excitation Wavelength[nm]	crv':
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        len0 = self.length()
        GraphIO.readDataFromFileGeneric(self, attributes)
        self.castCurve('Curve Spectrum', len0, silentSuccess=True)
        self[len0].update({'label': self[len0].attr('label').replace(' crv[0] [Cnts.]', '')})
        self.update({'xlabel': self.formatAxisLabel(GraphSpectrum.AXISLABELS[0]),
                     'ylabel': self.formatAxisLabel(GraphSpectrum.AXISLABELS[1])})
        self.update({'subplots_adjust': [0.2, 0.15]})


class GraphSpectrumPerkinElmerASC(GraphSpectrum):
    """
    Opens a kind of Perkin Elmer file.
    This parser is not exhaustive and does not necessary comply with the
    file format specifications. Much more information is contained in the file
    """

    AXISLABELS = [GraphSpectrum.AXISLABELS[0], ['Intensity', '', '%']]

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        if fileExt == '.asc' and line3.endswith('.asc'):
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        # reads header and count headers lines
        nlines = 0
        headers = []
        with open(self.filename) as fp:
            for line in fp:
                line = line.strip()
                nlines += 1
                # parse a few possibly useful headers lines
                if len(line) > 0 and ' ' in line:
                    if line[0] not in ['0', '1', '2', '3', '4', '5', '6', '7',
                                       '8', '9']:
                        headers.append(line)
                if line == '%R':
                    attributes.update({'_spectrumSubclass': 'CurveSpectrumReflectance'})
                if line == '%T':
                    attributes.update({'_spectrumSubclass': 'CurveSpectrumTransmittance'})
                if line == '#DATA':
                    break
        data = np.transpose(np.genfromtxt(self.filename, skip_header=nlines,
                                          delimiter='\t', invalid_raise=False))
        self.append(CurveSpectrum(data, attributes))
        for h in range(len(headers)):
            self[-1].update({'headers'+str(h): headers[h]})
        self.update({'xlabel': self.formatAxisLabel(GraphSpectrumPerkinElmerASC.AXISLABELS[0]),
                     'ylabel': self.formatAxisLabel(GraphSpectrumPerkinElmerASC.AXISLABELS[1])})
