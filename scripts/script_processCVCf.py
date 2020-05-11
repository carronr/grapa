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
from grapa.curve_image import Curve_Image
from grapa.mathModule import roundSignificant

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
            graph.curve(c).update({'labelhide': 1})
    else: # legend == 'minmax':
        for c in range(1, graph.length()-1):
            graph.curve(c).update({'labelhide': 1})

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
                           'ylabel': graph.formatAxisLabel(['Apparent doping density', 'N_{CV}', 'cm$^{-3}$']),
                           'xlabel': labels[0]}})
    presets.update({'NVlog': copy.deepcopy(presets['NV'])})
    presets['NVlog'].update({'ylim': '', 'typeplot': 'semilogy'})
    presets.update({'Nx': {'ylim': [0, np.nan],
                           'alter': ['CurveCV.x_CVdepth_nm', 'CurveCV.y_CV_Napparent'],
                           'typeplot': '',
                           'ylabel': graph.formatAxisLabel(['Apparent doping density', 'N_{CV}', 'cm$^{-3}$']),
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
    graphNyqui = Graph('', **newGraphKwargs)
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
        graphTmp = Graph(os.path.join(folder, file), complement={'_CfLoadPhase': True, '_CfLoadNyquist': True}, **newGraphKwargs)
        if graph.length() == 0:
            graph.update(graphTmp.graphInfo)
        graph.append(graphTmp.curve(0)) # append C-f
        if graphTmp.length() > 1:
            for c in graphTmp.iterCurves():
                if c.getAttribute('_CfPhase', False):
                    graphPhase.append(c)
                if c.getAttribute('_CfNyquist', False):
                    graphNyqui.append(c)
    graph.colorize     (Colorscale(np.array([[1,0.43,0], [0,0,1]]), invert=True)) # ThW admittance colorscale
    graphPhase.colorize(Colorscale(np.array([[1,0.43,0], [0,0,1]]), invert=True)) # ThW admittance colorscale
    graphNyqui.colorize(Colorscale(np.array([[1,0.43,0], [0,0,1]]), invert=True)) # ThW admittance colorscale
    lbl = graph.curve(-1).getAttribute('label').replace(' K','K').split(' ')
    if len(lbl) > 1:
        graph.update     ({'title': ' '.join(lbl[:-1])})
        graphPhase.update({'title': ' '.join(lbl[:-1])})
        graphNyqui.update({'title': ' '.join(lbl[:-1])})
        graph.replaceLabels     (' '.join(lbl[:-1]), '')
        graphPhase.replaceLabels(' '.join(lbl[:-1]), '')
        graphNyqui.replaceLabels(' '.join(lbl[:-1]), '')
    graphPhase.update({'xlabel': graph.getAttribute('xlabel'),
                       'ylabel': graphPhase.formatAxisLabel('Impedance angle [°]')})
    graphNyqui.update({'xlabel': graph.getAttribute('xlabel'),
                       'ylabel': graphNyqui.formatAxisLabel('Impedance angle [°]')})
    graphNyqui.update({'xlabel': graphNyqui.formatAxisLabel(GraphCf.AXISLABELSNYQUIST[0]),
                       'ylabel': graphNyqui.formatAxisLabel(GraphCf.AXISLABELSNYQUIST[1])})
    # mask undesired legends
    maskUndesiredLegends(graph, legend)
    maskUndesiredLegends(graphPhase, legend)
    maskUndesiredLegends(graphNyqui, legend)
    # set xlim
    setXlim(graph, 'tight')
    setXlim(graphPhase, 'tight')
    setXlim(graphNyqui, 'tight')
    
    # save
    filesave = os.path.join(folder, graph.getAttribute('title').replace(' ','_')+'_') # graphIO.filesave_default(self)
    plotargs = {} # {'ifExport': False, 'ifSave': False}
    graphattr = {}
    for attr in ['alter', 'typeplot', 'xlim', 'ylim', 'xlabel', 'ylabel']:
        graphattr.update({attr: graph.getAttribute(attr)})
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
    # graph 5: Nyquist
    limmax = -np.inf
    for c in graphNyqui.iterCurves():
        limmax = max(limmax, max(c.x()), max(c.y()))
    graphNyqui.update({'alter': '', 'xlim': [0, limmax], 'ylim': [0, limmax],
                       'subplots_adjust': [0.15, 0.15, 0.95, 0.95],
                       'figsize': [5,5]})
    graphNyqui.plot(filesave=filesave+'Nyquist', **plotargs)
    if pltClose:
        plt.close()
    # graph 6: image derivative, T vs f
    graphImage = Graph()
    flag = True
    x = [0] + list(graph.curve(0).x())
    Tmin, Tmax = np.inf, -np.inf
    for c in graph.iterCurves():
        T = c.getAttribute('temperature', None)
        if T is None:
            T = c.getAttribute('temperature [k]', None)
        if T is None:
            print('Script Cf, image, cannot identify Temperature', c.getAttributes())
            flag = False
        else:
            Tmin, Tmax = min(Tmin, T),  max(Tmax, T)
        y = [T] + list(c.y(alter='CurveCf.y_mdCdlnf'))
        if len(x) != len(y):
            print('WARNING data curve', T, 'not consistent number of points throughout the input files!', c.getAttribute('filename'))
            continue
        graphImage.append(Curve_Image([x, y], {}))
    if flag:
        # levels -> int value does not seem to work (matplotlib version?)
        m, M = np.inf, -np.inf
        for c in graphImage.iterCurves():
            m = min(m, np.min(c.y()[1:]))
            M = max(M, np.max(c.y()[1:]))
        space = roundSignificant(M / 12, 1)
        levels = [m] + list(np.arange(0, M*1.5, space))
        for l in range(len(levels)-2, -1, -1):
            if levels[l] > M:
                del levels[-1]
            else:
                break
        if len(levels) > 1 and levels[0] > levels[1]:
            del levels[0]
        # graph size
        Tdelta = (Tmax - Tmin) / 50
        if Tdelta > 0:
            fmin, fmax = np.min(graph.curve(0).x()), np.max(graph.curve(0).x())
            fdelta = (np.log10(fmax) - np.log10(fmin))
            subadj = [0.8, 0.5, 0.8+fdelta, 0.5+Tdelta, 'abs']
            graphImage.update({'subplots_adjust': subadj,
                               'figsize': [2+fdelta, 1+Tdelta]})
        else:
            print('script Cf, image, Warning T delta <=0, cannot properly scale image.')
        graphImage.update({'typeplot': 'semilogx',
                           'xlabel': graph.getAttribute('xlabel'),
                           'ylabel': graphImage.formatAxisLabel(['Temperature', '', 'K'])})
        graphImage.curve(0).update({'datafile_xy1rowcol': 1,
                                    'cmap': 'afmhot',
                                    'type': 'contourf',
                                    'colorbar': {'label': graph.getAttribute('ylabel'),#'-dC / dln(f)',
                                                 'adjust': [1.05, 0, 0.05, 1, 'ax']},
                                    'vmin': 0,
                                    'levels': levels})
        # 'cmap': [[0.91, 0.25, 1], [1.09, 0.75, 1], 'hls']
        graphImage.plot(filesave=filesave+'image', **plotargs)
        if pltClose:
            plt.close()
    # graph 7: f vs apparent doping
    graph.update({'alter': ['CurveCV.x_CVdepth_nm', 'x'], 'typeplot': 'semilogy'})
    graph.update({'xlabel': graph.formatAxisLabel(['Apparent depth', 'd', 'nm']),
                  'ylabel': graph.formatAxisLabel(['Frequency', 'f', 'Hz']),
                  'xlim':'', 'ylim':''})
    graph.plot(filesave=filesave+'apparentDepth', **plotargs)
    if pltClose:
        plt.close()
    
    graph.update(graphattr) # restore initial graph
    print('Tip for next step: pick inflection points for different T, then the fit activation energy.')
    print('End of process C-f.')
    return graph

    
if __name__ == "__main__":

    folder = './../examples/Cf/'
    graph = script_processCf(folder, pltClose=False)


    folder = './../examples/CV/'
    #graph = script_processCV(folder, ROIfit=[0.15,0.3], pltClose=False)

    plt.show()
    
