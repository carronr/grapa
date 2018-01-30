# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:57:53 2017

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
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
from grapa.graphIO import GraphIO
from grapa.curve import Curve
from grapa.colorscale import Colorscale

from grapa.datatypes.curveCf import GraphCf
from grapa.datatypes.curveCV import GraphCV, CurveCV
from grapa.datatypes.curveArrhenius import CurveArrheniusExtrapolToZero



def maskUndesiredLegends(graph, legend):
    """
    mask undesired legends in Graph object
    legend: possible values: 'no', 'minmax', 'all'
    """
    if legend == 'all':
        pass
    elif legend == 'none':
        for c in range(0, graph.length()):
            graph.curve(c).update({'label': ''})
    else: # legend == 'minmax':
        for c in range(1, graph.length()-1):
            graph.curve(c).update({'label': ''})

def setXlim(graph, keyword='tight'):
    if keyword != 'tight':
        print('processCV Cf setXlim unknown keyword')
    xlim = [np.inf, -np.inf]
    for c in graph.iterCurves():
        xlim = [min(xlim[0], min(c.x())), max(xlim[1], max(c.x()))]
    graph.update({'xlim': xlim})

    

def script_processCV(folder, legend='minmax', ROIfit=None, ROIsmart=None, pltClose=True, newGraphKwargs={}):
    """
    """
    newGraphKwargs = copy.deepcopy(newGraphKwargs)
    newGraphKwargs.update({'silent': True})
    
    print('Script process C-V')
    if ROIfit is None:
        ROIfit = CurveCV.CST_MottSchottky_Vlim_def
    if ROIsmart is None:
        ROIsmart = CurveCV.CST_MottSchottky_Vlim_adaptative
    
    graph      = Graph('', **newGraphKwargs)
    graphPhase = Graph('', **newGraphKwargs)
    graphVbi   = Graph('', **newGraphKwargs)
    # list possible files
    listdir = []
    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)):
            fileName, fileExt = os.path.splitext(file)
            fileExt = fileExt.lower()
            line1, line2, line3 = GraphIO.readDataFileLine123(os.path.join(folder, file))
            if GraphCV.isFileReadable(fileName, fileExt, line1=line1, line2=line2, line3=line3):
                listdir.append(file)
    if len(listdir) == 0:
        print('Found no suitable file in folder', folder)
        return Graph('', **newGraphKwargs)
    listdir.sort()
    # open all data files
    for file in listdir:
        print(file)
        graphTmp = Graph(os.path.join(folder, file), complement={'_CVLoadPhase': True}, **newGraphKwargs)
        if graph.length() == 0:
            graph = graphTmp
            while graph.length() > 1:
                graph.deleteCurve(1)
        else:
            graph.append(graphTmp.curve(0))
        if graphTmp.length() > 1:
            graphPhase.append(graphTmp.curve(1))
    graph.colorize(Colorscale(np.array([[1,0.43,0], [0,0,1]]), invert=True)) # ThW admittance colorscale
    graphPhase.colorize(Colorscale(np.array([[1,0.43,0], [0,0,1]]), invert=True)) # ThW admittance colorscale
    lbl = graph.curve(-1).getAttribute('label').replace(' K','K').split(' ')
    if len(lbl) > 1:
        graph.update({'title': ' '.join(lbl[:-1])})
        graph.replaceLabels(' '.join(lbl[:-1]), '')
        graphPhase.update({'title': ' '.join(lbl[:-1])})
        graphPhase.replaceLabels(' '.join(lbl[:-1]), '')
    graphPhase.update({'xlabel': graph.getAttribute('xlabel'),
                       'ylabel': graph.formatAxisLabel('Impedance angle [°]')})
    # mask undesired legends
    maskUndesiredLegends(graph, legend)
    maskUndesiredLegends(graphPhase, legend)
    # set xlim
    setXlim(graph, 'tight')
    setXlim(graphPhase, 'tight')
    
    labels = graph.getAttribute('xlabel'),  graph.getAttribute('ylabel')
    presets = {}
    presets.update({'default': {'ylim': [0, np.nan], 'alter': '',
                                'typeplot': '',
                                'xlabel': labels[0], 'ylabel': labels[1]}})
    presets.update({'MS': {'ylim': '', 'alter': ['', 'CurveCV.y_ym2'],
                           'typeplot': '',
                           'ylabel': graph.formatAxisLabel(['1 / C^2', '', 'nF$^{-2}$ cm$^{4}$']),
                           'xlabel': labels[0]}})
    presets.update({'NV': {'ylim': [0, np.nan],
                           'typeplot': '',
                           'alter': ['', 'CurveCV.y_CV_Napparent'],
                           'ylabel': graph.formatAxisLabel(['Apparent doping density', 'N_{CV}', 'm$^{-3}$']),
                           'xlabel': labels[0]}})
    presets.update({'NVlog': copy.deepcopy(presets['NV'])})
    presets['NVlog'].update({'ylim': '', 'typeplot': 'semilogy'})
    presets.update({'Nx': {'ylim': [0, np.nan],
                           'alter': ['CurveCV.x_CVdepth_nm', 'CurveCV.y_CV_Napparent'],
                           'typeplot': '',
                           'ylabel': graph.formatAxisLabel(['Apparent doping density', 'N_{CV}', 'm$^{-3}$']),
                           'xlabel': graph.formatAxisLabel(['Apparent depth', 'd', 'nm'])}})
    presets.update({'Nxlog': copy.deepcopy(presets['Nx'])})
    presets['Nxlog'].update({'ylim': '', 'typeplot': 'semilogy'})

    # save
    filesave = os.path.join(folder, graph.getAttribute('title').replace(' ','_')+'_') # graphIO.filesave_default(self)
    plotargs = {} # {'ifExport': False, 'ifSave': False}
    #plotargs = {}
    # default graph: C vs log(f)
    graph.update(presets['default'])
    graph.plot(filesave=filesave+'CV', **plotargs)
    if pltClose:
        plt.close()
    # graph 2: Mott Schottky
    graph.update(presets['MS'])
    graph.plot(filesave=filesave+'MottSchottky', **plotargs)
    if pltClose:
        plt.close()
    # graph 3: N vs V
    graph.update(presets['NV'])
    graph.plot(filesave=filesave+'NV', **plotargs)
    if pltClose:
        plt.close()
    graph.update(presets['NVlog'])
    graph.plot(filesave=filesave+'NVlog', **plotargs)
    if pltClose:
        plt.close()
    # graph 4: N vs depth
    graph.update(presets['Nx'])
    graph.plot(filesave=filesave+'Ndepth', **plotargs)
    if pltClose:
        plt.close()
    graph.update(presets['Nxlog'])
    graph.plot(filesave=filesave+'Ndepthlog', **plotargs)
    if pltClose:
        plt.close()
    # Fit Mott-Schottky curves
    Ncvminmax0 = [np.inf, -np.inf]
    Ncvminmax1 = [np.inf, -np.inf]
    graphVbi.append(Curve([[], []], CurveArrheniusExtrapolToZero.attr))
    graphVbi.append(Curve([[], []], {'linestyle': 'none'}))
    graphVbi.append(Curve([[], []], CurveArrheniusExtrapolToZero.attr))
    graphVbi.append(Curve([[], []], {'linestyle': 'none'}))
    graphSmart = Graph('', **newGraphKwargs)
    numCurves = graph.length()
    for curve in range(numCurves):
        c = graph.curve(curve)
        c.update({'linewidth': 0.25})
        new   = c.CurveCV_fitVbiN(ROIfit, silent=True)
        smart = c.CurveCV_fitVbiN(c.smartVlim_MottSchottky(Vlim=ROIsmart), silent=True)
        if isinstance(new, CurveCV):
            graph.append(new)
            graph.curve(-1).update({'linespec': '--', 'color': c.getAttribute('color')})
            Vbi, Ncv = new.getAttribute('_popt')[0], new.getAttribute('_popt')[1]
            graphVbi.curve(0).appendPoints([c.getAttribute('temperature [k]')], [Vbi])
            graphVbi.curve(1).appendPoints([c.getAttribute('temperature [k]')], [Ncv])
            Ncvminmax0 = [min(Ncvminmax0[0], Ncv), max(Ncvminmax0[1], Ncv)]
        if isinstance(smart, CurveCV):
            graphSmart.append(smart)
            graphSmart.curve(-1).update({'linespec': '--', 'color': c.getAttribute('color')})
            Vbi, Ncv = smart.getAttribute('_popt')[0], smart.getAttribute('_popt')[1]
            graphVbi.curve(2).appendPoints([c.getAttribute('temperature [k]')], [Vbi])
            graphVbi.curve(3).appendPoints([c.getAttribute('temperature [k]')], [Ncv])
            Ncvminmax1 = [min(Ncvminmax1[0], Ncv), max(Ncvminmax1[1], Ncv)]
    graphVbi.update({'legendtitle': 'Mott-Schottky fit'})
    graphVbi.curve(0).update({'linespec': 'o', 'color': 'k', 'label': 'Built-in voltage (same Vlim)', 'markeredgewidth': 0, })
    graphVbi.curve(1).update({'linespec': 'x', 'color': 'k', 'label': 'Ncv (same Vlim)'})
    graphVbi.curve(2).update({'linespec': 'o', 'color': 'r', 'label': 'Built-in voltage (adaptative Vlim)', 'markeredgewidth': 0})
    graphVbi.curve(3).update({'linespec': 'x', 'color': 'r', 'label': 'Ncv (adaptative Vlim)'})
    graphVbi.update({'xlabel': graphVbi.formatAxisLabel(['Temperature', 'T', 'K']),
                     'ylabel': graphVbi.formatAxisLabel(['Built-in voltage', 'V_{bi}', 'V'])})
    graphVbi.plot(filesave=filesave+'VbiT', **plotargs)
    if pltClose:
        plt.close()
    for curve in graphVbi.iterCurves():
        curve.swapShowHide()
    graphVbi.update({'xlabel': graphVbi.formatAxisLabel(['Temperature', 'T', 'K']),
                     'ylabel': presets['NV']['ylabel'], 'typeplot': 'semilogy'})
    graphVbi.plot(filesave=filesave+'NcvT', **plotargs)
    if pltClose:
        plt.close()
    # Mott schottky, then N vs V, with fit curves - same Vlim
    graph.update(presets['MS'])
    graph.update({'legendtitle': 'Mott-Schottky fit (same Vlim)'})
    graph.update({'axvline': ROIfit})
    graph.plot(filesave=filesave+'MottSchottkySameVlim', **plotargs)
    if pltClose:
        plt.close()
    graph.update(presets['NVlog'])
    graph.update({'ylim': [0.75*Ncvminmax0[0], 2.2*Ncvminmax0[1]]})
    graph.plot(filesave=filesave+'NVlogSameVlim', **plotargs)
    if pltClose:
        plt.close()
    graph.update({'axvline': '', 'legendtitle': ''})
    # Mottschottky, then N vs V, with fit lines - adaptative ROI
    for i in range(graph.length()-1, numCurves-1, -1):
        graph.deleteCurve(i)
    graph.merge(graphSmart)
    graph.update(presets['MS'])
    graph.update({'legendtitle': 'Mott-Schottky fit (adaptative Vlim)'})
    graph.plot(filesave=filesave+'MottSchottkyAdaptative', **plotargs)
    if pltClose:
        plt.close()
    graph.update(presets['NVlog'])
    graph.update({'ylim': [0.75*Ncvminmax1[0], 2.2*Ncvminmax1[1]]})
    graph.plot(filesave=filesave+'NVlogAdaptative', **plotargs)
    if pltClose:
        plt.close()
    graph.update({'legendtitle': '', 'ylim': [0.5*Ncvminmax1[0], 5*Ncvminmax1[1]]})

    # graph phase
    graphPhase.update({'alter': '', 'ylim': [0, 90]})
    f = graphPhase.curve(0).x()
#    graphPhase.append(Curve([[min(f), max(f), max(f), min(f)], [0, 0, 20, 20]], {'type': 'fill', 'facecolor': [1,0,0,0.5], 'linewidth': 0}))
    graphPhase.append(Curve([[min(f), max(f)], [20, 20]], {'color': [1,0,0], 'linewidth': 2, 'linespec': '--'}))
    graphPhase.plot(filesave=filesave+'phase', **plotargs)
    if pltClose:
        plt.close()
    
    print('End of process C-V.')
    return graph
    
    
def script_processCf(folder, legend='minmax', pltClose=True, newGraphKwargs={}):
    """
    legend: possible values: 'no', 'minmax', 'all'
    """
    print('Script process C-f')
    newGraphKwargs = copy.deepcopy(newGraphKwargs)
    newGraphKwargs.update({'silent': True})
    graph = Graph('', **newGraphKwargs)
    graphPhase = Graph('', **newGraphKwargs)
    # list possible files
    listdir = []
    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)):
            fileName, fileExt = os.path.splitext(file)
            fileExt = fileExt.lower()
            line1, line2, line3 = GraphIO.readDataFileLine123(os.path.join(folder, file))
            if GraphCf.isFileReadable(fileName, fileExt, line1=line1, line2=line2, line3=line3):
                listdir.append(file)
    if len(listdir) == 0:
        print('Found no suitable file in folder', folder)
    listdir.sort()
    # open all data files
    for file in listdir:
        print(file)
        graphTmp = Graph(os.path.join(folder, file), complement={'_CfLoadPhase': True}, **newGraphKwargs)
        if graph.length() == 0:
            graph = graphTmp
            while graph.length() > 1:
                graph.deleteCurve(1)
        else:
            graph.append(graphTmp.curve(0))
        if graphTmp.length() > 1:
            graphPhase.append(graphTmp.curve(1))
    graph.colorize(Colorscale(np.array([[1,0.43,0], [0,0,1]]), invert=True)) # ThW admittance colorscale
    graphPhase.colorize(Colorscale(np.array([[1,0.43,0], [0,0,1]]), invert=True)) # ThW admittance colorscale
    lbl = graph.curve(-1).getAttribute('label').replace(' K','K').split(' ')
    if len(lbl) > 1:
        graph.update({'title': ' '.join(lbl[:-1])})
        graph.replaceLabels(' '.join(lbl[:-1]), '')
        graphPhase.update({'title': ' '.join(lbl[:-1])})
        graphPhase.replaceLabels(' '.join(lbl[:-1]), '')
    graphPhase.update({'xlabel': graph.getAttribute('xlabel'),
                       'ylabel': graphPhase.formatAxisLabel('Impedance angle [°]')})
    # mask undesired legends
    maskUndesiredLegends(graph, legend)
    maskUndesiredLegends(graphPhase, legend)
    # set xlim
    # set xlim
    setXlim(graph, 'tight')
    setXlim(graphPhase, 'tight')
    
    # save
    filesave = os.path.join(folder, graph.getAttribute('title').replace(' ','_')+'_') # graphIO.filesave_default(self)
    plotargs = {} # {'ifExport': False, 'ifSave': False}
    # default graph: C vs log(f)
    graph.update({'ylim': [0, np.nan]})
    graph.plot(filesave=filesave+'Clogf', **plotargs)
    if pltClose:
        plt.close()
    # graph 2: derivative
    graph.update({'alter': ['', 'CurveCf.y_mdCdlnf'], 'typeplot': 'semilogx'})
    graph.update({'ylim': [0, np.nan], 'ylabel': 'Derivative d Capacitance / d ln(f)'})
    graph.plot(filesave=filesave+'deriv', **plotargs)
    if pltClose:
        plt.close()
    # graph 3: derivative, zoomed-in
    graph.update({'alter': ['', 'CurveCf.y_mdCdlnf'], 'typeplot': 'semilogx'})
    graph.update({'ylim': [0, 1.5*max(graph.curve(0).y(alter='CurveCf.y_mdCdlnf'))]})
    graph.plot(filesave=filesave+'derivZoom', **plotargs)
    if pltClose:
        plt.close()
    # graph 4: phase
    graphPhase.update({'alter': '', 'typeplot': 'semilogx', 'ylim': [0,90]})
    f = graphPhase.curve(0).x()
    graphPhase.append(Curve([[min(f), max(f)], [20, 20]], {'color': [1,0,0], 'linewidth': 2, 'linespec': '--'}))
    graphPhase.plot(filesave=filesave+'phase', **plotargs)
    if pltClose:
        plt.close()
    
    print('End of process C-f.')
    return graph

    
if __name__ == "__main__":

    folder = './../examples/Cf/'
    graph = script_processCf(folder, pltClose=False)


    folder = './../examples/CV/'
    #graph = script_processCV(folder, ROIfit=[0.15,0.3], pltClose=False)

    plt.show()
    