# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:02:34 2020

@author: Romain Carron
Copyright (c) 2020, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""


import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import sys

path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if path not in sys.path:
    sys.path.append(path)

from grapa.graph import Graph
from grapa.mathModule import roundSignificantRange, roundSignificant, is_number

from grapa.datatypes.curveJscVoc import GraphJscVoc, CurveJscVoc




    

def script_processJscVoc(file, pltClose=True, newGraphKwargs={},
                         ROIJsclim=None, ROIVoclim=None, ROITlim=None):
    """
    
    TO IMPLEMENT PARAMETERS:
        Jo, A fit: JSC LIM, VOC LIM
        Voc vs T: K range below max
    """
    newGraphKwargs = copy.deepcopy(newGraphKwargs)
    newGraphKwargs.update({'silent': True})
    
    print('Script process Jsc-Voc')
    
    graph = Graph(file, **newGraphKwargs)
    
    print('WARNING: Cell area assumed to be', graph.curve(0).getArea(), 'cm-2.')
    
    lbl = graph.curve(0).getAttribute('label')
    if len(lbl) > 1:
        graph.update({'title': lbl})

    labelT = graph.formatAxisLabel(['Temperature', 'T', 'K'])
    labelA = graph.formatAxisLabel(['Diode ideality factor', 'A', ''])
    label1000AT = graph.formatAxisLabel(['1000 / (A*Temperature)', '', 'K$^{-1}$'])
    labellnJ0 = graph.formatAxisLabel(['ln(J$_0$)', '', 'mAcm$^{-2}$'])
    labelVoc = graph.formatAxisLabel(['Voc', '', 'V'])
    
    presets = {}
    presets.update({'default': {'ylim': '', 'xlim': '', 'alter': '',
                                'typeplot': 'semilogy',
                                'xlabel': graph.getAttribute('xlabel'),
                                'ylabel': graph.getAttribute('ylabel')}})
    presets.update({'ideality': {'ylim': '', 'xlim': '', 'alter': '',
                                'typeplot': '',
                                'xlabel': labelT,
                                'ylabel': labelA}})
    presets.update({'J0vsAT': {'ylim': '', 'xlim': '',  'typeplot': '',
                               'alter': ['CurveArrhenius.x_1000overK', 'CurveArrhenius.y_Arrhenius'],
                                'xlabel': label1000AT,
                                'ylabel': labellnJ0}})
    presets.update({'VocvsT': {'ylim': '', 'xlim': [0, ''],  'typeplot': '',
                               'alter': '',
                                'xlabel': labelT,
                                'ylabel': labelVoc}})
    # save
    folder = os.path.dirname(file)
    filesave = os.path.join(folder, 'JscVoc_'+graph.getAttribute('title').replace(' ','_')+'_') # graphIO.filesave_default(self)
    print('filesave', filesave)
    plotargs = {} # {'ifExport': False, 'ifSave': False}
    grap2 = copy.deepcopy(graph)
    # default graph: with fits
    graph.update(presets['default'])
    Voclim = roundSignificantRange([0,   max(graph.curve(0).x())*1.01], 3)
    Jsclim = roundSignificantRange([CurveJscVoc.CST_Jsclim0, max(graph.curve(0).y())*1.1],  2)
    if ROIJsclim is not None:
        if isinstance(ROIJsclim, list):
            Jsclim = ROIJsclim
        elif is_number(ROIJsclim):
            Jsclim[0] = ROIJsclim
    if ROIVoclim is not None:
        if isinstance(ROIVoclim, list):
            Voclim = ROIVoclim
        elif is_number(ROIVoclim):
            Voclim[0] = ROIVoclim
    print('Fit J0, A: Voc limits', ', '.join(['{:.3f}'.format(v) for v in Voclim]), 'V')
    print('Fit J0, A: Jsc limits', ', '.join(['{:.2e}'.format(v) for v in Jsclim]), 'mAcm-2')
    for curve in GraphJscVoc.CurveJscVoc_fitNJ0(graph, Voclim, Jsclim, 3, True,
                                                curve=graph.curve(0), silent=True):
        graph.append(curve)
    graph.plot(filesave=filesave+'fits', **plotargs)
    if pltClose:
        plt.close()
    # Graph ideality factor vs T
    grap2.update(presets['ideality'])
    grap2.append(graph.curve(-3))
    for c in range(grap2.length()):
        grap2.curve(c).swapShowHide()
    grap2.plot(filesave=filesave+'IdealityvsT', **plotargs)
    if pltClose:
        plt.close()
    # Graph ideality factor vs T
    grap2.update(presets['J0vsAT'])
    grap2.deleteCurve(-1)
    curve = graph.curve(-1)
    grap2.append(curve)
    curve.swapShowHide()
    ROI = list(roundSignificant([min(curve.x_1000overK())*0.95,
                                  1.05*max(curve.x_1000overK())], 4))
    grap2.append(graph.curve(-1).CurveArrhenius_fit(ROI, silent=True))
    Ea = grap2.curve(-1).getAttribute('_popt')[0]
    grap2.curve(-1).update({'label': 'E$_a$ '+'{:1.3f}'.format(Ea)+' eV'})
    grap2.plot(filesave=filesave+'J0vsAT', **plotargs)
    if pltClose:
        plt.close()
    # Voc vs T
    grap2.update(presets['VocvsT'])
    grap2.deleteCurve(-1) # previous fit
    grap2.deleteCurve(-1) # ln(Jo) vs A*T
    curve = grap2.curve(0)
    Tlim = [0.99*np.min(graph.curve(1).y()), 1.01*np.max(graph.curve(1).y())]
    Tlim[0] = max(Tlim[0], Tlim[1] - 80) # restrict fit to highest 100K
    if ROITlim is not None:
        if isinstance(ROITlim, list):
            Tlim = ROITlim
        elif is_number(ROITlim):
            Tlim[0] = ROITlim
    print('Voc vs T: fit range restricted to', ', '.join(['{:.1f}'.format(T) for T in Tlim]), 'K')
    res = GraphJscVoc.splitIllumination(grap2, 3.0, True, Tlim, True, curve=curve, silent=True)
    grap2.append(res, idx=2)
    grap2.plot(filesave=filesave+'VocvsT', **plotargs)
    if pltClose:
        plt.close()
    
    graph.curve(-1).swapShowHide()
    graph.curve(-3).swapShowHide()
    print('WARNING: fits limits are chosen automatically. It is YOUR responsibility to check for the goodness of fit, and manually decide for the limits!')
    print('End of process Jsc-Voc.')
    return graph
    

    
if __name__ == "__main__":

    file = './../examples/JscVoc/JscVoc_SAMPLE_c2_Values.txt'
    graph = script_processJscVoc(file, pltClose=False)

    plt.show()
    
