# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 19:47:25 2016

@author: car
Copyright (c) 2018, Empa, Romain Carron
"""

import numpy as np
import warnings


from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.curve import Curve
from grapa.mathModule import is_number, roundgraphlim

from grapa.datatypes.curveJV import CurveJV




class GraphJVDarkIllum(Graph):

    def __init__(self, fileDark, fileIllum, area=np.nan, temperature=273.15+25, complement={}, silent=True, ifPlot=False, **newGraphKwargs):
        #print ('GraphJVDarkIllum area', area)
# TODO: give Graph files in constructor. Currently only handle filename (str)
        self.idxDark = -1
        self.idxIllum= -1
        # intervert fileIllum and fileDark if able to detect inversion
        swap = -1
        if isinstance(fileIllum, str) and fileIllum.find('dark') > -1:
            swap = 1
        elif isinstance(fileIllum, Graph) and fileIllum.getAttribute('filename').find('dark') > -1:
            swap = 1
        if isinstance(fileDark, str) and fileDark.find('dark') > -1:
            swap = 0
        elif isinstance(fileDark, Graph) and fileDark.getAttribute('filename').find('dark') > -1:
            swap = 0
        if swap == 1:
            swap = fileDark
            fileDark = fileIllum
            fileIllum = swap

        # prepare additional attributes
        complement.update({'temperature': temperature})
        if is_number(area):
            complementArea = {'area': area}
        if 'area' in complement:
            print('WARNING GraphJVDarkIllum: area defined in argument complement, beware that area correction might not function properly!')

        # start opening files, supposedly the dark
        Graph.__init__(self, fileDark, complement=complement, silent=silent, **newGraphKwargs)
        # fit first file
        if self.length() > 0:
            self.curve(-1).update(complementArea)
            self.idxDark = 0
            c = self.curve(-1)
            c.simpleLabel(forceCalc=True)
            # compute fit, create new curve for it
            c.update({'color': 'b'})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                newCurve = c.CurveJVFromFit(silent=silent)
            if not isinstance(newCurve, Curve): # if fit failed
                newCurve = Curve([[0], [np.nan]], {'diodeFit': [np.nan]*5, 'mpp': [np.nan]*3}, silent=True)
            self.append(newCurve)
        else:
            pass

        # open illuminated file, supposedly the illum
        graph = Graph(fileIllum, complement=complement, silent=silent, **newGraphKwargs)
        # fit second file
        if graph.length() > 0:
            graph.curve(-1).update(complementArea)
            self.idxIllum = self.length()
            self.merge(graph)
            c = self.curve(-1)
            c.simpleLabel(forceCalc=True)
            # compute fit, create new curve for it
            c.update({'color': 'r'})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                newCurve = c.CurveJVFromFit(silent=silent)
            if not isinstance(newCurve, Curve): # if fit failed
                newCurve = CurveJV([[0], [np.nan]], {'diodeFit': [np.nan]*5, 'mpp': [np.nan]*3}, silent=True)
            self.append(newCurve)
        else:
            pass
        
        # apparent photocurrent: difference between dark and illum
        if self.length() > 2: # for this the opening of the 2 file must have been successful
            Jsc = self.curve(self.idxIllum).getAttribute('Jsc')
            idx = [self.idxIllum, self.idxDark]
            c = [self.curve(idx[0]), self.curve(idx[1])]
            V = np.sort(np.append(c[0].x(), c[1].x()))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                J = c[0].interpJ(V) - c[1].interpJ(V)
            self.append(CurveJV([V, J], {'color': 'g', 'label': 'Apparent photocurrent', 'Jsc': Jsc, 'area': self.curve(self.idxIllum).area()}, ifCalc=False, silent=True))
            self.curve(-1).update({'linestyle': 'none'}) # by default this curve is not shown
        else:
            pass
 




    def printShort(self, header=False, onlyIllum=False, onlyDark=False):
        if header:
            return self.curve(0).printShort(header=True)
        if onlyDark and self.idxDark >= 0:
            if len(self.curve(self.idxDark).x()) > 0:
                return self.curve(self.idxDark).printShort()
        if onlyIllum and self.idxIllum >= 0:
            if len(self.curve(self.idxIllum).x()) > 0:
                return self.curve(self.idxIllum).printShort()
        out = ''
        if self.idxDark >= 0:
            out = out + self.curve(self.idxDark).printShort()
        if self.idxIllum >= 0:
            out = out + self.curve(self.idxIllum).printShort()
        return out


    def plot(self, filesave='', ylim=None, figAx=None, ifSave=True, ifExport=True, alter=None, pltClose=False):
        if figAx is not None:
            pltClose = False
        if pltClose:
            import matplotlib.pyplot as plt
        # want to handle alter ourself. argument still placed to ensure call call be conducted
        d = self.idxDark
        i = self.idxIllum
        if ylim is None:
            mM0 = np.array([np.min(self.curve(d).J()), np.max(self.curve(d).J())] if d >= 0 else [np.inf, -np.inf])
            mM2 = np.array([np.min(self.curve(i).J()), np.max(self.curve(i).J())] if i >= 0 else [np.inf, -np.inf])
            if self.getAttribute('ylim', None) is None:
                ylim = [min(mM0[0], mM2[0]), max(mM0[1], mM2[1])]
#               print ('GraphDarkIllum plot: ylim set to', ylim,'(0',mM0,mM2,')')
                ylim = list(roundgraphlim(ylim))
#               print ('   ylim', ylim, type(ylim))
            mM0 = np.array([np.min(self.curve(d).y(alter='log10abs')), np.max(self.curve(d).y(alter='log10abs'))] if d >= 0 else [np.inf, -np.inf])
            mM2 = np.array([np.min(self.curve(i).y(alter='log10abs')), np.max(self.curve(i).y(alter='log10abs'))] if i >= 0 else [np.inf, -np.inf])
            ylimLog = [np.floor(min(mM0[0], mM2[0])), np.ceil(max(mM0[1], mM2[1]))]
            ylimLog = list(np.power(10, np.array(ylimLog)))
#            print ('    ylimlog', ylimLog,'(0',mM0,mM2,')')
        #ylimInit = self.getAttribute('ylim')
        self.plotStd   (filesave, ifSave=ifSave, ifExport=ifExport, figAx=figAx, ylim=ylim)
        if pltClose:
            plt.close()
        if self.length() > 2:
            self.plotDiff  (filesave, ifSave=ifSave, ifExport=ifExport, figAx=figAx, ylim=ylim)
            if pltClose:
                plt.close()
        self.plotLogAbs(filesave, ifSave=ifSave, ifExport=ifExport, figAx=figAx, ylim=ylimLog)
        if pltClose:
            plt.close()
        #self.update({'ylim': ylimInit})


    def plotStd(self, filesave='', ylim=None, figAx=None, ifSave=True, ifExport=True):
        # normal plot
        temp = {}
        if ylim is not None:
            temp = self.deleteAttr(['ylim', 'alter'])
            self.update({'ylim': ylim})
        self.curve(1).update({'linestyle': 'None'})
        if self.length() > 2:
            self.curve(3).update({'linestyle': 'None'})
        Graph.plot(self, filesave=GraphIO.filesave_default(self,filesave) +'lin', figAx=figAx, ifSave=ifSave, ifExport=ifExport)
        self.curve(1).update({'linestyle': ''})
        if self.length() > 2:
            self.curve(3).update({'linestyle': ''})
        # restore initial attributes
        self.update(temp)


    def plotLogAbs(self, filesave='', ylim=None, figAx=None, ifSave=True, ifExport=True):
        # unset xlim, ylim indications (could also uset the delete() function like for axhline and axvline)
        # change ylabel of the graph
        temp = self.deleteAttr(['ylim', 'xlim', 'ylabel', 'axhline', 'axvline', 'alter'])
        self.update({'ylabel': 'Log current density', 'alter': 'log10abs'})
        if ylim is not None:
            self.update({'ylim': ylim})
        Graph.plot(self, filesave=GraphIO.filesave_default(self, filesave)+'log', figAx=figAx, ifSave=ifSave, ifExport=ifExport)
        self.update(temp)


    def plotDiff(self, filesave='', ylim=None, figAx=None, ifSave=True, ifExport=True):
        temp = self.deleteAttr(['axhline', 'alter'])
        if self.length() > 2:
            Jsc = self.curve(self.idxIllum).getAttribute('Jsc')
            if self.idxDark >= 0 and self.idxIllum >= 0:
                self.update({'axhline': [Jsc, -Jsc]})
            else:
                self.update({'axhline': [-Jsc]})
        linestIdx = [i for i in [1, 3, 4] if i < self.length()]
        linestyle = [self.curve(i).getAttribute('linestyle') for i in linestIdx]
        if self.length() > 2:
            self.curve(4).update({'linestyle': ''})
        if ylim is not None:
            saveAttr = self.deleteAttr(['ylim'])
            self.update({'ylim': ylim})

        Graph.plot(self, filesave=GraphIO.filesave_default(self, filesave) +'diff', figAx=figAx, ifSave=ifSave, ifExport=ifExport)

        # restore curve properties
        if ylim is not None:
            self.update(saveAttr)
        for i in range(len(linestIdx)):
            self.curve(linestIdx[i]).update({'linestyle': linestyle[i]})
        self.update(temp)
