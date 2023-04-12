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
import warnings

path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if path not in sys.path:
    sys.path.append(path)

from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.curve import Curve
from grapa.colorscale import Colorscale
from grapa.curve_image import Curve_Image
from grapa.mathModule import roundSignificant

from grapa.datatypes.graphCf import GraphCf
from grapa.datatypes.graphCV import GraphCV
from grapa.datatypes.curveCV import  CurveCV
from grapa.datatypes.curveArrhenius import CurveArrheniusExtrapolToZero



def maskUndesiredLegends(graph, legend):
    """
    mask undesired legends in Graph object
    legend: possible values: 'no', 'minmax', 'all'
    """
    if legend == 'all':
        pass
    elif legend == 'none':
        for c in range(0, len(graph)):
            graph[c].update({'labelhide': 1})
    else:  # legend == 'minmax':
        for c in range(1, len(graph)-1):
            graph[c].update({'labelhide': 1})


def setXlim(graph, keyword='tight'):
    if keyword != 'tight':
        print('processCV Cf setXlim unknown keyword')
    xlim = [np.inf, -np.inf]
    for c in graph.iterCurves():
        xlim = [min(xlim[0], min(c.x())), max(xlim[1], max(c.x()))]
    graph.update({'xlim': xlim})



def script_processCV(folder, legend='auto', ROIfit=None, ROIsmart=None, pltClose=True, newGraphKwargs={}):
    """
    """
    DEFAULT_T = 300

    WARNINGS = []

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
#        with warnings.catch_warnings():
        with warnings.catch_warnings(record=True) as w:
            graphTmp = Graph(os.path.join(folder, file), complement={'_CVLoadPhase': True}, **newGraphKwargs)
        if len(w) > 0:
            for _ in w:
                WARNINGS.append(str(_.message))
        if len(graph) == 0:
            graph = graphTmp
            while len(graph) > 1:
                graph.deleteCurve(1)
        else:
            graph.append(graphTmp[0])
        if len(graphTmp) > 1:
            graphPhase.append(graphTmp[1])
        else:
            graphPhase.append(Curve([[0], [0]], {}))
    graph.colorize(Colorscale(np.array([[1, 0.43, 0], [0, 0, 1]]), invert=True))  # ThW admittance colorscale
    graphPhase.colorize(Colorscale(np.array([[1, 0.43, 0], [0, 0, 1]]), invert=True))  # ThW admittance colorscale
    lbl = graph[-1].attr('label').replace(' K','K').split(' ')
    lb0 = graph[0].attr('label').replace(' K','K').split(' ')
    joined = ' '.join(lbl[:-1])
    joine0 = ' '.join(lb0[:-1])
    legendifauto = 'all' if joined != joine0 else 'minmax'
    if len(lbl) > 1:
        graph.update(     {'title': joined})
        graphPhase.update({'title': joined})
        if legendifauto == '':  # later, will also remove most text of legend
            graph.replaceLabels(joined, '')
            graphPhase.replaceLabels(joined, '')
    graphPhase.update({'xlabel': graph.attr('xlabel'),
                       'ylabel': graph.formatAxisLabel('Impedance angle [°]')})
    # mask undesired legends
    if legend == 'auto':
        legend = legendifauto
    maskUndesiredLegends(graph, legend)
    maskUndesiredLegends(graphPhase, legend)
    # set xlim
    setXlim(graph, 'tight')
    setXlim(graphPhase, 'tight')

    labels = graph.attr('xlabel'),  graph.attr('ylabel')
    presets = {}
    presets.update({'default': {'ylim': [0, np.nan], 'alter': '',
                                'typeplot': '',
                                'xlabel': labels[0], 'ylabel': labels[1],
                                'subplots_adjust': [0.16, 0.15]}})
    presets.update({'MS': {'ylim': '', 'alter': ['', 'CurveCV.y_ym2'],
                           'typeplot': '',
                           'ylabel': graph.formatAxisLabel(['1 / C$^2$', '', 'nF$^{-2}$ cm$^{4}$']),
                           'xlabel': labels[0]}})
    presets.update({'NV': {'ylim': [0, np.nan],
                           'typeplot': '',
                           'alter': ['', 'CurveCV.y_CV_Napparent'],
                           'ylabel': graph.formatAxisLabel(['Apparent doping density', 'N_{CV}', 'cm$^{-3}$']),
                           'xlabel': labels[0]}})
    presets.update({'NVlog': copy.deepcopy(presets['NV'])})
    presets['NVlog'].update({'ylim': '', 'typeplot': 'semilogy'})
    presets.update({'Nx': {'ylim': [0, np.nan], 'xlim': '',
                           'alter': ['CurveCV.x_CVdepth_nm', 'CurveCV.y_CV_Napparent'],
                           'typeplot': '',
                           'ylabel': graph.formatAxisLabel(['Apparent doping density', 'N_{CV}', 'cm$^{-3}$']),
                           'xlabel': graph.formatAxisLabel(['Apparent depth', 'd', 'nm'])}})
    presets.update({'Nxlog': copy.deepcopy(presets['Nx'])})
    presets['Nxlog'].update({'ylim': '', 'typeplot': 'semilogy'})

    graphVbi.update({'subplots_adjust': [0.16, 0.15]})
    graphPhase.update({'subplots_adjust': [0.16, 0.15]})

    # save
    filesave = os.path.join(folder, graph.attr('title').replace(' ','_')+'_')  # graphIO.filesave_default(self)
    plotargs = {}  # {'ifExport': False, 'ifSave': False}
    # plotargs = {}
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
    # add doping V=0
    curves = []
    for curve in graph:
        # check that temperature is set
        if curve.attr('temperature [k]') == '':
            msg = ('WARNING processCV file '
                   + os.path.basename(curve.attr('filename'))
                   +' temperature [k] not found, set to '+str(DEFAULT_T)+' K.')
            print(msg)
            WARNINGS.append(msg)
            curve.update({'temperature [k]': DEFAULT_T})
        # do required stuff
        tmp = curve.CurveCV_0V()
        if tmp is not False:
            tmp.update({'label': tmp.attr('label')+' Ncv @ 0V'})
            curves.append(tmp)
    graph.append(curves)
    graph.update(presets['Nx'])
    graph.plot(filesave=filesave+'Ndepth', **plotargs)
    if pltClose:
        plt.close()
    graph.update(presets['Nxlog'])
    graph.plot(filesave=filesave+'Ndepthlog', **plotargs)
    if pltClose:
        plt.close()
    # save V=0 doping values
    N0V_T, N0V_N = [], []
    for curve in curves:
        N = curve.y(alter=presets['Nxlog']['alter'][1])
        N0V_T.append(curve.attr('temperature [k]'))
        N0V_N.append(N[2])
    N0V = Curve([N0V_T, N0V_N], {'label': 'N$_\mathrm{CV}$ (0V)', 'linestyle': 'none', 'linespec':'s', 'markeredgewidth':0, 'color':'b'})
    for c in range(len(graph)-1, -1, -1): # remove unnecessary curves
        if graph[c] in curves:
            graph.deleteCurve(c)

    # Fit Mott-Schottky curves
    Ncvminmax0 = [np.inf, -np.inf]
    Ncvminmax1 = [np.inf, -np.inf]
    graphVbi.append(Curve([[], []], CurveArrheniusExtrapolToZero.attr))
    graphVbi.append(Curve([[], []], {'linestyle': 'none'}))
    graphVbi.append(Curve([[], []], CurveArrheniusExtrapolToZero.attr))
    graphVbi.append(Curve([[], []], {'linestyle': 'none'}))
    graphVbi.append(N0V)
    graphSmart = Graph('', **newGraphKwargs)
    numCurves = graph.length()
    for curve in range(numCurves): # number of Curves in graph will change
        c = graph.curve(curve)
        c.update({'linewidth': 0.25})
        new   = c.CurveCV_fitVbiN(ROIfit, silent=True)
        smart = c.CurveCV_fitVbiN(c.smartVlim_MottSchottky(Vlim=ROIsmart), silent=True)
        if isinstance(new, CurveCV):
            graph.append(new)
            graph[-1].update({'linespec': '--', 'color': c.attr('color')})
            Vbi, Ncv = new.attr('_popt')[0], new.attr('_popt')[1]
            graphVbi[0].appendPoints([c.attr('temperature [k]')], [Vbi])
            graphVbi[1].appendPoints([c.attr('temperature [k]')], [Ncv])
            Ncvminmax0 = [min(Ncvminmax0[0], Ncv), max(Ncvminmax0[1], Ncv)]
        if isinstance(smart, CurveCV):
            graphSmart.append(smart)
            graphSmart[-1].update({'linespec': '--', 'color': c.attr('color')})
            Vbi, Ncv = smart.attr('_popt')[0], smart.attr('_popt')[1]
            graphVbi[2].appendPoints([c.attr('temperature [k]')], [Vbi])
            graphVbi[3].appendPoints([c.attr('temperature [k]')], [Ncv])
            Ncvminmax1 = [min(Ncvminmax1[0], Ncv), max(Ncvminmax1[1], Ncv)]
    graphVbi.update({'legendtitle': 'Mott-Schottky fit'})
    graphVbi[0].update({'linespec': 'o', 'color': 'k', 'label': 'Built-in voltage (same Vlim)', 'markeredgewidth': 0})
    graphVbi[1].update({'linespec': 'x', 'color': 'k', 'label': 'N$_\mathrm{CV}$ (same Vlim)'})
    graphVbi[2].update({'linespec': 'o', 'color': 'r', 'label': 'Built-in voltage (adaptative Vlim)', 'markeredgewidth': 0})
    graphVbi[3].update({'linespec': 'x', 'color': 'r', 'label': 'N$_\mathrm{CV}$ (adaptative Vlim)'})
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

    if len(WARNINGS) > 0:
        print('Enf of process C-V. Got some warnings along the way... see above or summary')
        for msg in WARNINGS:
            print(msg)
    else:
        print('End of process C-V, successful.')
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
                if c.attr('_CfPhase', False):
                    graphPhase.append(c)
                if c.attr('_CfNyquist', False):
                    graphNyqui.append(c)
    graph.colorize     (Colorscale(np.array([[1,0.43,0], [0,0,1]]), invert=True)) # ThW admittance colorscale
    graphPhase.colorize(Colorscale(np.array([[1,0.43,0], [0,0,1]]), invert=True)) # ThW admittance colorscale
    graphNyqui.colorize(Colorscale(np.array([[1,0.43,0], [0,0,1]]), invert=True)) # ThW admittance colorscale
    lbl = graph.curve(-1).attr('label').replace(' K','K').split(' ')
    if len(lbl) > 1:
        graph.update     ({'title': ' '.join(lbl[:-1])})
        graphPhase.update({'title': ' '.join(lbl[:-1])})
        graphNyqui.update({'title': ' '.join(lbl[:-1])})
        graph.replaceLabels     (' '.join(lbl[:-1]), '')
        graphPhase.replaceLabels(' '.join(lbl[:-1]), '')
        graphNyqui.replaceLabels(' '.join(lbl[:-1]), '')
    graphPhase.update({'xlabel': graph.attr('xlabel'),
                       'ylabel': graphPhase.formatAxisLabel('Impedance angle [°]')})
    graphNyqui.update({'xlabel': graph.attr('xlabel'),
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
    filesave = os.path.join(folder, graph.attr('title').replace(' ','_')+'_') # graphIO.filesave_default(self)
    plotargs = {} # {'ifExport': False, 'ifSave': False}
    graphattr = {}
    for attr in ['alter', 'typeplot', 'xlim', 'ylim', 'xlabel', 'ylabel']:
        graphattr.update({attr: graph.attr(attr)})
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
        T = c.attr('temperature', None)
        if T is None:
            T = c.attr('temperature [k]', None)
        if T is None:
            print('Script Cf, image, cannot identify Temperature',
                  c.getAttributes())
            flag = False
        else:
            Tmin, Tmax = min(Tmin, T),  max(Tmax, T)
        y = [T] + list(c.y(alter='CurveCf.y_mdCdlnf'))
        if len(x) != len(y):
            print('WARNING data curve', T, 'not consistent number of points',
                  'throughout the input files!', c.attr('filename'))
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
                           'xlabel': graph.attr('xlabel'),
                           'ylabel': graphImage.formatAxisLabel(['Temperature', '', 'K'])})
        graphImage.curve(0).update({'datafile_xy1rowcol': 1,
                                    'cmap': 'afmhot',
                                    'type': 'contourf',
                                    'colorbar': {'label': graph.attr('ylabel'),  #'-dC / dln(f)',
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
                  'xlim': '', 'ylim': ''})
    graph.plot(filesave=filesave+'apparentDepth', **plotargs)
    if pltClose:
        plt.close()

    graph.update(graphattr)  # restore initial graph
    print('Tip for next step: pick inflection points for different T, then the fit activation energy.')
    print('End of process C-f.')
    return graph


if __name__ == "__main__":

    folder = './../examples/Cf/'
    #graph = script_processCf(folder, pltClose=False)


    folder = './../examples/CV/'
    graph = script_processCV(folder, ROIfit=[0.15,0.3], pltClose=True)

    plt.show()
