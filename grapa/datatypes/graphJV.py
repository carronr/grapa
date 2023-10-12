# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:15:27 2018

@author: Romain
"""

import os
import numpy as np

from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.datatypes.curveJV import CurveJV


class GraphJV(Graph):

    FILEIO_GRAPHTYPE = 'J-V curve'
    FILEIO_GRAPHTYPE_TIV = 'TIV curve'
    FILEIO_GRAPHTYPE_IV_HLS = 'I-V curve (H-L soaking)'

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line2='', line3='', **kwargs):
        if fileExt == '.txt' and line1[-11:] == '_ParamDescr':
            return True  # JV
        if (fileExt == '.txt' and line1.strip().startswith('Sample name:')
                and line2.strip().startswith('Cell name:')
                and line3.strip().startswith('Cell area [cm')):
            return True  # TIV
        if (fileExt == '.csv' and len(line1) > 40
                and line1.strip()[:21] == '"Time";"Temperature ['
                and line1.strip()[-38:] == '";"Illumination [mV]";"Humidity [%RH]"'):
            return True  # J-V from Heat-Light soaking setup
        return False

    def readDataFromFile(self, attributes, **kwargs):
        line1 = kwargs['line1'] if 'line1' in kwargs else ''
        fileName = kwargs['fileName'] if 'fileName' in kwargs else ''
        if (len(line1) > 40 and line1.strip()[:21] == '"Time";"Temperature ['
                and line1.strip()[-38:] == '";"Illumination [mV]";"Humidity [%RH]"'):
            GraphJV.readDataFromFileIV_HLS(self, attributes)
        elif (line1.strip().startswith('Sample name:')
                and (not fileName.startswith('I-V_'))):
            GraphJV.readDataFromFileTIV(self, attributes)
        else:
            GraphJV.readDataFromFileJV(self, attributes)
        # set 'sample' attribute
        if self[-1].attr('sample') == '':
            self[-1].update({'sample': self[-1].attr('label')})

    def readDataFromFileJV(self, attributes):
        fileName, fileext = os.path.splitext(self.filename)  # , fileExt
        # extract data analysis from acquisition software - requires a priori
        # knowledge of file structure
        # especially want to know cell area before creation of the CurveJV
        # object
        JVsoft = np.genfromtxt(self.filename, skip_header=1, delimiter='\t',
                               usecols=[3], invalid_raise=False, dtype=str)
        key = ['Acquis soft Voc', 'Acquis soft Jsc', 'Acquis soft FF',
               'Acquis soft Eff', 'Acquis soft Cell area', 'Acquis soft Vmpp',
               'Acquis soft Jmpp', 'Acquis soft Pmpp', 'Acquis soft Rp',
               'Acquis soft Rs', 'Acquis soft datetime',
               'Acquis soft Temperature', 'Acquis soft Illumination factor']
        for i in range(len(key)):
            val = JVsoft[i] # .decode('utf-8')
            try:
                val = float(val)
            except ValueError:
                pass
            attributes.update({key[i]: val})
        if 'label' in attributes:
            attributes['label'] = attributes['label'].replace('I-V ','').replace('_', ' ')
        # create Curve object
        data = np.genfromtxt(self.filename, skip_header=1, delimiter='\t',
                             usecols=[0, 1], invalid_raise=False)
        self.append(CurveJV(np.transpose(data), attributes,
                            units=['V', 'mAcm-2'], ifCalc=True,
                            silent=self.silent))
        self.update({'collabels': ['Voltage [V]', 'Current density [mA cm-2]']})
        self.update({'xlabel': self.formatAxisLabel(['Bias voltage', 'V', 'V']),
                     'ylabel': self.formatAxisLabel(['Current density', 'J', 'mA cm$^{-2}$'])})
        # some cosmetic information
        self.update({'axhline': [0, {'linewidth': 0.5}],
                     'axvline': [0, {'linewidth': 0.5}]})

    def readDataFromFileTIV(self, attributes):
        GraphIO.readDataFromFileGeneric(self, attributes)
        # check if we can actually read some TIV data, otherwise this might be
        # a processed I-V_ database (summary of cells properties)
        if self.length() == 0:
            GraphIO.readDataFromFileTxt(self, attributes)
            return
        # back to TIV data reading
        self.update({'collabels': '',
                     'xlabel': self.formatAxisLabel(['Bias voltage', 'V', 'V']),
                     'ylabel': self.formatAxisLabel(['Current density', 'J', 'mA cm$^{-2}$'])})
        # delete columns of temperature in newer versions
        if len(self) == 3:
            if self[-1].attr('voltage (v)') == 'T sample (K)':
                self.deleteCurve(-1)
            if self[-1].attr('voltage (v)') == 'T stage (K)':
                self.deleteCurve(-1)
        self[0].update({'voltage (v)': ''})  # artifact from file reading
        filebasename, fileExt = os.path.splitext(os.path.basename(self.filename))
        self[0].update({'label': filebasename.replace('_', ' ')})
        # rename attributes to match these of JV setup
        key = ['sample name', 'cell name', 'voc (v)', 'jsc (ma/cm2)', 'ff (%)', 'eff (%)', 'cell area [cm2]', 'vmpp (v)', 'jmpp (ma/cm2)', 'pmpp (mw/cm2)', 'rp (ohmcm2)', 'rs (ohmcm2)', 'temperature [k]', 'Acquis soft Illumination factor'] #, 'Acquis soft datatime'
        new = ['sample', 'cell', 'Acquis soft Voc', 'Acquis soft Jsc', 'Acquis soft FF', 'Acquis soft Eff', 'Acquis soft Cell area', 'Acquis soft Vmpp', 'Acquis soft Jmpp', 'Acquis soft Pmpp', 'Acquis soft Rp', 'Acquis soft Rs', 'Acquis soft Temperature', 'Acquis soft Illumination factor'] #,  'Acquis soft datatime'
        c = self[0]
        for i in range(len(key)):
            c.update({new[i]: c.attr(key[i])})
            c.update({key[i]: ''})
        # we believe this information is trustable
        c.update({'temperature': c.attr('Acquis soft Temperature')})
        c.update({'measId': str(c.attr('temperature'))})
        # recreate curve as a CurveJV
        self.data[0] = CurveJV(c.data, attributes=c.getAttributes(),
                               silent=True)
        self.update({'meastype': GraphJV.FILEIO_GRAPHTYPE_TIV})

    def readDataFromFileIV_HLS(self, attributes):
        le = len(self)
        GraphIO.readDataFromFileGeneric(self, attributes, delimiter=';')
        self[le].update({'area': 1.0})
        if max(abs(self[le].x())) > 10:  # want units in [V], not [mV]
            self[le].setX(self[le].x()*0.001)
        self.update({'xlabel': self.attr('xlabel').replace('[mv]', '[mV]')})
        self.update({'xlabel': self.formatAxisLabel(self.attr('xlabel')),
                     'ylabel': self.formatAxisLabel(self.attr('ylabel'))})
        self.castCurve('Curve JV', le, silentSuccess=True)
        self.update({'meastype': GraphJV.FILEIO_GRAPHTYPE_IV_HLS})
