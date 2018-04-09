# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np
import copy
from scipy.optimize import fsolve


from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.datatypes.curveSIMS import CurveSIMS

from grapa.mathModule import roundSignificant, is_number


class GraphSIMS(Graph):
    """
    msmtId is an id based on the initial datafile name, in order to be able to
    process simultaneously several SIMS measurements
    class must be written like instance of self is actually a Graph object,
    not a GraphSIMS. Cannot call a method of GraphSIMS as method of self.
    """

    KEYWORDS = {'ggi':    [['^71Ga+'],['^71Ga+', '^113In+']],
                'cgi':    [['Cu+'],   ['^71Ga+', '^113In+']],
                'gi':     [['^71Ga+', '^113In+'],[]],
                'cusn':   [['Cu+'],   ['Sn+']],
                'cuzn':   [['Cu+'],   ['Zn+']],
                'znsn':   [['Zn+'],   ['Sn+']],
                'cuznsn': [['Cu+'],   ['Zn+', 'Sn+']],
                'cgt':    [['Cu+'],   ['^70Ge+', 'Sn+']],
                'ggt':    [['^72Ge+'],['^72Ge+', 'Sn+']]
                }
    
    TOHIDE = ['F+', 'Mg+', 'Al+', '^41K+', 'Fe+', '^65Cu+', '^66Zn+', 'Ga+',
              'Se+', '^94Mo+', 'In+', '^118Sn+', '^119Sn+',
              '^110Cd+', '^Cd112+', '^113Cd+', '^92Mo+', '^96Mo+', 'Cs+']

    YIELDS = {'^71Ga+': 1, '^113In+': 5.5, 'Cu+': 300,
              '^70Ge+': 4500, 'Sn+': 480, 'Zn+': 6000}

              
    FILEIO_GRAPHTYPE = 'SIMS data'
    
    AXISLABELS = [['Sputter time', 't', 's'], ['Intensity', '', 'counts']]

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1='', line3='', **kwargs):
        if (fileExt == '.txt'  and
            (line1[0:7] == '\ttotal\t' or line3[0:16] == 'Sputter Time (s)')):
            return True
        return False
    
        
    def readDataFromFile(self, attributes, **kwargs):
        # open using fileGeneric
        GraphIO.readDataFromFileGeneric(self, attributes)
        if self.length() == 0:
            GraphIO.readDataFromFileGeneric(self, attributes, ifReplaceCommaByPoint=True)
        self.update({'typeplot': 'semilogy'})
        ylabel = self.curve(-1).getAttribute('sputter time (s)')
        if ylabel == '':
            ylabel = GraphSIMS.AXISLABELS[1]
        if self.curve(-1).getAttribute('sputter time (s)') != '':
            self.update({'xlabel': self.formatAxisLabel(GraphSIMS.AXISLABELS[0])})
            self.update({'ylabel': self.formatAxisLabel(ylabel)})
        # set correct labels
        for c in range(self.length()):
            self.curve(c).update({'label': self.curve(c).getAttribute('total')})
            self.castCurve('CurveSIMS', c, silentSuccess=True)
        # prints default keywords
        print('SIMS available ratio keywords:', ', '.join(key for key in GraphSIMS.KEYWORDS))
        # set default SIMS relative yields
        msmtId = self.curve(-1).getAttribute('_SIMSmsmt')
        GraphSIMS.setYieldCoefs(self, msmtId, ifAuto=True)
        # add temporary Ga+In curve
        ok = GraphSIMS.appendReplaceCurveRatio(self, msmtId, 'gi', 'Ga+In')
        if ok:
            # find edges of Ga+In curve
            c = GraphSIMS.getCurve(self, msmtId, 'Ga+In', silent=True)
            ROI = c.findLayerBorders(returnIdx=False) if c is not None else self.curve(-1).x(index=[0, self.curve(-1).shape(1)])
            GraphSIMS.setLayerBoundaries(self, msmtId, ROI)
        # by default, hide some curves - add others, for kestertes?
        hidden = []
        toHide = GraphSIMS.TOHIDE
        for h in toHide:
            c = GraphSIMS.getCurve(self, msmtId, h, silent=True)
            if c is not None:
                c.update({'linestyle': 'none'})
                hidden.append(h)
        if len(hidden) > 0:
            print('SIMS curves hidden automatically:', ', '.join(hidden)+'.')

                    
                    
    
    def getLayerBoundaries(self, msmtId):
        for c in range(self.length()):
            if self.curve(c).getAttribute('_SIMSmsmt') == msmtId:
                return self.curve(c).getAttribute('_SIMSLayerBoundaries')
        print('ERROR graphSIMS getLayerBoundaries cannot find curve with desired msmtId (',msmtId,')')
        return False
    
    def setLayerBoundaries(self, msmtId, ROI):
        for c in range(self.length()):
            if self.curve(c).getAttribute('_SIMSmsmt') == msmtId:
                self.curve(c).update({'_SIMSLayerBoundaries': [min(ROI), max(ROI)]})
        self.update({'axvline': [min(ROI), max(ROI)]})
        return GraphSIMS.getLayerBoundaries(self, msmtId)

    def setLayerBoundariesDepthParameters(self, msmtId, ROI, depth):
        ROI = GraphSIMS.setLayerBoundaries(self, msmtId, ROI)
        offset, mult = None, None
        for c in range(self.length()):
            if self.curve(c).getAttribute('_SIMSmsmt') == msmtId:
                if offset is None:
                    ROI = np.argmin(np.abs(self.curve(c).x() - ROI[0])), np.argmin(np.abs(self.curve(c).x() - ROI[1]))
                    x = self.curve(c).x(index=ROI)
                    offset = - x[0]
                    mult = depth / (x[1]-x[0])
                self.curve(c).setDepthParameters(offset, mult)
        if offset is None:
            print('ERROR GraphSIMS setLayerBoundariesDepthParameters cannot find curve with desired msmtId (',msmtId,')')
        return True
    def setLayerBoundariesDepthParametersGUI(self, depth, ROI, msmtId=None):
        return GraphSIMS.setLayerBoundariesDepthParameters(self, msmtId, ROI, depth)

        
                
        

    def getCurve(self, msmtId, element, silent=False, returnIdx=False):
        for c in range(self.length()):
            if self.curve(c).getAttribute('_SIMSelement') == element and self.curve(c).getAttribute('_SIMSmsmt') == msmtId:
                if returnIdx:
                    return c
                return self.curve(c)
        if not silent:
            print ('ERROR getCurve: cannot find curve associated with element', element, 'and id',msmtId,'.')
            return False
        return None


    # function setting custom or defaults SIMS yields
    def setYieldCoefs(self, msmtId, elementYield=None, ifAuto=False):
        if elementYield is None:
            elementYield = {}
        if ifAuto:
            default = copy.deepcopy(GraphSIMS.YIELDS)
            default.update(elementYield)
            elementYield = default
        for key in elementYield:
            c = GraphSIMS.getCurve(self, msmtId, key, silent=True)
            if c is not None:
                c.update({'_SIMSYieldCoef': elementYield[key]})
                print('Graph SIMS setYields', key, elementYield[key])

    
    def appendReplaceCurveRatio(self, msmtId, ratioCurves, curveId, attributes={}, savgol=None):
        """
        savgol: 2-element list width and degree for Savistky-Golay smoothening
        """
        if savgol is not None: # savistky golay smoothening of the Curves
            from scipy.signal import savgol_filter
            savgol = list(savgol)
            if len(savgol) < 2 or not is_number(savgol[0]) or not is_number(savgol[1]):
                savgol = None
                print('Warning GraphSIMS.appendReplaceCurveRatio: savgol must be a 2-elements list of ints for savistky-golay width and degree. Received', savgol)
            if savgol is not None: # tests on savlues
                savgol[0] = max(1, int((savgol[0]-1)/2) * 2 + 1, 1) # must be an odd number
                savgol[1] = min(max(int(savgol[1]), 1), savgol[0]-1)
            if savgol == [1,1]:
                savgol = None
            if savgol is not None:
                print('GraphSIMS: smooth curve data using Savistky-Golay window', savgol[0], 'degree', savgol[1])
        ratioCurves = GraphSIMS.getRatioProcessRatiocurves(self, ratioCurves)
        ret = True
        ratio = None
        x = None
        filename, layerBoundaries, depthOffset, depthMult = '', '', '', ''
        for i in range(len(ratioCurves)):
            for j in range(len(ratioCurves[i])):
                curve = GraphSIMS.getCurve(self, msmtId, ratioCurves[i][j], silent=True)
                if curve is None:
                    print('WARNING SIMS appendReplaceCurveRatio Curve not found: ', ratioCurves[i][j],'.')
                    ret = False
                    return False
                if ratio is None:
                    tmp = copy.deepcopy(curve.y())
                    ratio = np.array([tmp * 0.0, tmp * 0.0])
                try:
                    tmp = curve.y() * curve.getAttribute('_SIMSYieldCoef', default=1)
                    if savgol is not None:
                        tmp = savgol_filter(tmp, *savgol, deriv=0)
                    ratio[i,:] += tmp
                except ValueError as e:
                    print('ERROR GraphSIMS appendReplaceCurveRatio: curve not same size.')
                    print(ratioCurves[i][j], 'shape curve.y', curve.y().shape, 'into array', ratio[i,:].shape)
                    print(e)
                    print(curve.y())
                if x is None:
                    x = curve.x()
                if filename == '':
                    filename = curve.getAttribute('filename')
                if layerBoundaries == '':
                    layerBoundaries = curve.getAttribute('_SIMSlayerBoundaries')
                if depthOffset == '':
                    depthOffset = curve.getAttribute('_simsdepth_offset')
                if depthMult == '':
                    depthMult = curve.getAttribute('_simsdepth_mult')
        for i in range(len(ratioCurves)):
            if len(ratioCurves[i]) == 0:
                ratio[i,:] = 1
        attr = {'label': curveId, 'filename': filename, '_simslayerboundaries': layerBoundaries, '_simsdepth_offset': depthOffset, '_simsdepth_mult': depthMult}
        attr.update(attributes)
        new  = CurveSIMS([x, ratio[0]/ratio[1]], attributes=attr)
        c = GraphSIMS.getCurve(self, msmtId, curveId, silent=True, returnIdx=True)
        if c is None:
            self.append(new)
        else:
            self.replaceCurve(new, c)
        return ret
    def appendReplaceCurveRatioGUI(self, ratioCurves, curveId, msmtId=None):
        return GraphSIMS.appendReplaceCurveRatio(self, msmtId, ratioCurves, curveId)
    def appendReplaceCurveRatioGUISmt(self, ratioCurves, curveId, SGw, SGd, msmtId=None):
        return GraphSIMS.appendReplaceCurveRatio(self, msmtId, ratioCurves, curveId, savgol=[SGw, SGd])
        
 


    def getRatioProcessRatiocurves(self, ratioCurves):
        # handle keywords for ratiocurves
        if isinstance(ratioCurves, str):
            ratioCurves=ratioCurves.lower()
            if ratioCurves not in GraphSIMS.KEYWORDS:
                print('WARNING: SIMS setYieldsFromExternalData unknown key (',ratioCurves,'). Used GGI instead.')
                ratioCurves = 'ggi'
            for key in GraphSIMS.KEYWORDS:
                if ratioCurves == key:
                    ratioCurves = GraphSIMS.KEYWORDS[key]
                    print('Replaced keyword', key, 'with ratio', str(GraphSIMS.KEYWORDS[key])+'.')
                    break
        return ratioCurves
    
    def getRatio(self, msmtId, ROI, ratioCurves, ifReturnAvg=False, ifReturnCoefs=False):
        # output: [ratioOfAveragesOverROI, AverageOverROIofLocalRatio]
        ratioCurves = GraphSIMS.getRatioProcessRatiocurves(self, ratioCurves)
        # compute ratio
        avgOfRatio = np.array([ROI * 0.0, ROI * 0.0])
        ratioOfAvg = np.array([0.0, 0.0])
        retAvg = np.array([[0.0]*len(r) for r in ratioCurves])
        retCoefs = np.array([[1.0]*len(r) for r in ratioCurves])
        for i in range(len(ratioCurves)):
            for j in range(len(ratioCurves[i])):
                key = ratioCurves[i][j]
                curve = GraphSIMS.getCurve(self, msmtId, key, silent=True)
                if curve is not None:
                    retCoefs[i][j] = curve.getAttribute('_SIMSYieldCoef', default=1)
                    avgOfRatio[i,:] += (curve.y(index=ROI) * retCoefs[i][j])
                    retAvg[i][j] = np.average((curve.y(index=ROI))) * retCoefs[i][j]
                    ratioOfAvg[i] += retAvg[i][j]
                else:
                    print('ERROR: SIMS setYieldsFromExternalData cannot find curve',key,'.')
            if len(ratioCurves[i]) == 0:
                avgOfRatio[i,:] = ROI * 0.0 + 1
                ratioOfAvg[i] = 1
        RatioOfAvg = ratioOfAvg[0] / ratioOfAvg[1]
        AvgOfRatio = np.average(avgOfRatio[0][avgOfRatio[1]!=0] / avgOfRatio[1][avgOfRatio[1]!=0])
        out = [RatioOfAvg, AvgOfRatio, retAvg] if ifReturnAvg else [RatioOfAvg, AvgOfRatio]
        if ifReturnCoefs:
            out = out + [retCoefs]
        return out
    
    def targetRatioSetYield(self, msmtId, ROI, ratioCurves, element, target, silent=False):
        ratioInit = ratioCurves
        ratioCurves = GraphSIMS.getRatioProcessRatiocurves(self, ratioCurves)
        ratios = GraphSIMS.getRatio(self, msmtId, ROI, ratioCurves, ifReturnAvg=True)
        tunable = np.array([[False]*len(r) for r in ratioCurves])
        for i in range(len(ratioCurves)):
            for j in range(len(ratioCurves[i])):
                if ratioCurves[i][j] == element:
                    tunable[i][j] = True
        def func(coef, avgs, tunable, target):
            out0 = np.sum([avgs[0][j]*coef if tunable[0][j] else avgs[0][j] for j in range(len(avgs[0]))])
            out1 = np.sum([avgs[1][j]*coef if tunable[1][j] else avgs[1][j] for j in range(len(avgs[1]))])
            return out0 / out1 - target
        res = fsolve(func, 1, args=(ratios[2], tunable, target))
        if isinstance(res, np.ndarray):
            res = res[0]
        curve = GraphSIMS.getCurve(self, msmtId, element, silent=True)
        if curve is None:
            return False
        old = curve.getAttribute('_SIMSYieldCoef')
        curve.update({'_SIMSYieldCoef': res*old})
        if not silent:
            print('SIMS adjust yield to reach', ratioInit,'ratio', target,': ', curve.getAttribute('_SIMSelement'), 'yield', curve.getAttribute('_SIMSYieldCoef'), '(old value',old,')')
        return GraphSIMS.getRatioGUI(self, ROI, ratioInit, msmtId=msmtId, ifROIisIdx=True)
    def targetRatioSetYieldGUI(self, ROI, ratioCurves, element, target, msmtId=None):
        ROI = GraphSIMS.ROI_GUI2idx(self, msmtId, ROI)
        return GraphSIMS.targetRatioSetYield(self, msmtId, ROI, ratioCurves, element, target)


    def getRatioGUI(self, ROI, ratioCurves, msmtId=None, ifROIisIdx=False):
        if msmtId is None:
            print('ERROR GraphSIMS getRatioGUI: msmtId was not provided.')
            return False
        ratioName = str(ratioCurves)
        # GUI function: ROI is assumed is be function of sputtering time
        if not ifROIisIdx:
            ROI = GraphSIMS.ROI_GUI2idx(self, msmtId, ROI)
        GGIs = GraphSIMS.getRatio(self, msmtId, ROI, ratioCurves, ifReturnCoefs=True)
        out = ratioName + ': ' + '{:1.4f}'.format(GGIs[0]) + ' (avg of local ratios: ' + '{:1.4f}'.format(GGIs[1]) + '). Coefficients: ' + str(roundSignificant(GGIs[2][0],4)) + ', ' + str(roundSignificant(GGIs[2][1],4)) + '.'
        return out

    def ROI_GUI2idx(self, msmtId, ROI):
        c = None
        for curve in range(self.length()):
            if self.curve(curve).getAttribute('_SIMSmsmt') == msmtId:
                c = curve
                break
        if c is None:
            print('GraphSIMS getRatioGUI cannot find curve with correct msmtId (',msmtId,').')
        if ROI == '':
            return np.arange(0, self.curve(c).shape(1)-1)
        return np.arange(np.argmin(np.abs(self.curve(c).x() - min(ROI))), np.argmin(np.abs(self.curve(c).x() - max(ROI))))
        
        

