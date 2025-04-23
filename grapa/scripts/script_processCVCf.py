# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:57:53 2017

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import sys
import copy
import warnings
import numpy as np
import matplotlib.pyplot as plt

path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if path not in sys.path:
    sys.path.append(path)

from grapa.graph import Graph
from grapa.utils.graphIO import file_read_first3lines
from grapa.curve import Curve
from grapa.colorscale import Colorscale
from grapa.curve_image import Curve_Image
from grapa.mathModule import roundSignificant

from grapa.datatypes.graphCf import GraphCf
from grapa.datatypes.graphCV import GraphCV
from grapa.datatypes.curveCV import CurveCV
from grapa.datatypes.curveArrhenius import CurveArrheniusExtrapolToZero, CurveArrhenius


def maskUndesiredLegends(graph, legend):
    """
    mask undesired legends in Graph object
    legend: possible values: 'no', 'minmax', 'all'
    """
    if legend == "all":
        pass
    elif legend == "none":
        for c in range(0, len(graph)):
            graph[c].update({"labelhide": 1})
    else:  # legend == 'minmax':
        for c in range(1, len(graph) - 1):
            graph[c].update({"labelhide": 1})


def setXlim(graph, keyword="tight"):
    if keyword != "tight":
        print("processCV Cf setXlim unknown keyword")
    xlim = [np.inf, -np.inf]
    for curve in graph:
        xlim = [min(xlim[0], min(curve.x())), max(xlim[1], max(curve.x()))]
    graph.update({"xlim": xlim})


def script_processCV(
    folder, legend="auto", ROIfit=None, ROIsmart=None, pltClose=True, newGraphKwargs={}
):
    """ """
    DEFAULT_T = 300

    WARNINGS = []

    newGraphKwargs = copy.deepcopy(newGraphKwargs)
    newGraphKwargs.update({"silent": True})

    print("Script process C-V")
    if ROIfit is None:
        ROIfit = CurveCV.CST_MottSchottky_Vlim_def
    if ROIsmart is None:
        ROIsmart = CurveCV.CST_MottSchottky_Vlim_adaptative

    graph = Graph("", **newGraphKwargs)
    graphPhase = Graph("", **newGraphKwargs)
    graphVbi = Graph("", **newGraphKwargs)
    # list possible files
    listdir = []
    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)):
            fileName, fileExt = os.path.splitext(file)
            fileExt = fileExt.lower()
            line1, line2, line3 = file_read_first3lines(os.path.join(folder, file))
            if GraphCV.isFileReadable(
                fileName, fileExt, line1=line1, line2=line2, line3=line3
            ):
                listdir.append(file)
    if len(listdir) == 0:
        print("Found no suitable file in folder", folder)
        return Graph("", **newGraphKwargs)
    listdir.sort()
    # open all data files
    for file in listdir:
        print(file)
        #        with warnings.catch_warnings():
        with warnings.catch_warnings(record=True) as w:
            graphTmp = Graph(
                os.path.join(folder, file),
                complement={"_CVLoadPhase": True},
                **newGraphKwargs
            )
        if len(w) > 0:
            for _ in w:
                WARNINGS.append(str(_.message))
        if len(graph) == 0:
            graph = graphTmp
            while len(graph) > 1:
                graph.curve_delete(1)
        else:
            graph.append(graphTmp[0])
        if len(graphTmp) > 1:
            graphPhase.append(graphTmp[1])
        else:
            graphPhase.append(Curve([[0], [0]], {}))
    graph.colorize(
        Colorscale(np.array([[1, 0.43, 0], [0, 0, 1]]), invert=True)
    )  # ThW admittance colorscale
    graphPhase.colorize(
        Colorscale(np.array([[1, 0.43, 0], [0, 0, 1]]), invert=True)
    )  # ThW admittance colorscale
    lbl = graph[-1].attr("label").replace(" K", "K").split(" ")
    lb0 = graph[0].attr("label").replace(" K", "K").split(" ")
    joined = " ".join(lbl[:-1])
    joine0 = " ".join(lb0[:-1])
    legendifauto = "all" if joined != joine0 else "minmax"
    if len(lbl) > 1:
        graph.update({"title": joined})
        graphPhase.update({"title": joined})
        if legendifauto == "":  # later, will also remove most text of legend
            graph.replace_labels(joined, "")
            graphPhase.replace_labels(joined, "")
    graphPhase.update(
        {
            "xlabel": graph.attr("xlabel"),
            "ylabel": graph.formatAxisLabel("Impedance phase [°]"),
        }
    )
    # mask undesired legends
    if legend == "auto":
        legend = legendifauto
    maskUndesiredLegends(graph, legend)
    maskUndesiredLegends(graphPhase, legend)
    # set xlim
    setXlim(graph, "tight")
    setXlim(graphPhase, "tight")

    labels = graph.attr("xlabel"), graph.attr("ylabel")
    presets = {}
    presets.update(
        {
            "default": {
                "ylim": [0, np.nan],
                "alter": "",
                "typeplot": "",
                "xlabel": labels[0],
                "ylabel": labels[1],
                "subplots_adjust": [0.16, 0.15],
            }
        }
    )
    presets.update(
        {
            "MS": {
                "ylim": "",
                "alter": ["", "CurveCV.y_ym2"],
                "typeplot": "",
                "ylabel": graph.formatAxisLabel(
                    ["1 / C$^2$", "", "nF$^{-2}$ cm$^{4}$"]
                ),
                "xlabel": labels[0],
            }
        }
    )
    presets.update(
        {
            "NV": {
                "ylim": [0, np.nan],
                "typeplot": "",
                "alter": ["", "CurveCV.y_CV_Napparent"],
                "ylabel": graph.formatAxisLabel(
                    ["Apparent doping density", "N_{CV}", "cm$^{-3}$"]
                ),
                "xlabel": labels[0],
            }
        }
    )
    presets.update({"NVlog": copy.deepcopy(presets["NV"])})
    presets["NVlog"].update({"ylim": "", "typeplot": "semilogy"})
    presets.update(
        {
            "Nx": {
                "ylim": [0, np.nan],
                "xlim": "",
                "alter": ["CurveCV.x_CVdepth_nm", "CurveCV.y_CV_Napparent"],
                "typeplot": "",
                "ylabel": graph.formatAxisLabel(
                    ["Apparent doping density", "N_{CV}", "cm$^{-3}$"]
                ),
                "xlabel": graph.formatAxisLabel(["Apparent depth", "d", "nm"]),
            }
        }
    )
    presets.update({"Nxlog": copy.deepcopy(presets["Nx"])})
    presets["Nxlog"].update({"ylim": "", "typeplot": "semilogy"})

    graphVbi.update({"subplots_adjust": [0.16, 0.15]})
    graphPhase.update({"subplots_adjust": [0.16, 0.15]})

    # save
    filesave = os.path.join(
        folder, graph.attr("title").replace(" ", "_") + "_"
    )  # graphIO.filesave_default(self)
    plotargs = {}  # {'ifExport': False, 'ifSave': False}
    # plotargs = {}
    # default graph: C vs log(f)
    graph.update(presets["default"])
    graph.plot(filesave=filesave + "CV", **plotargs)
    if pltClose:
        plt.close()
    # graph 2: Mott Schottky
    graph.update(presets["MS"])
    graph.plot(filesave=filesave + "MottSchottky", **plotargs)
    if pltClose:
        plt.close()
    # graph 3: N vs V
    graph.update(presets["NV"])
    graph.plot(filesave=filesave + "NV", **plotargs)
    if pltClose:
        plt.close()
    graph.update(presets["NVlog"])
    graph.plot(filesave=filesave + "NVlog", **plotargs)
    if pltClose:
        plt.close()

    # graph 4: N vs depth
    # add doping V=0
    curves = []
    for curve in graph:
        # check that temperature is set
        if curve.attr("temperature [k]") == "":
            msg = (
                "WARNING processCV file "
                + os.path.basename(curve.attr("filename"))
                + " temperature [k] not found, set to "
                + str(DEFAULT_T)
                + " K."
            )
            print(msg)
            WARNINGS.append(msg)
            curve.update({"temperature [k]": DEFAULT_T})
        # do required stuff
        tmp = curve.CurveCV_0V()
        if tmp is not False:
            tmp.update({"label": tmp.attr("label") + " Ncv @ 0V"})
            curves.append(tmp)
    graph.append(curves)
    graph.update(presets["Nx"])
    graph.plot(filesave=filesave + "Ndepth", **plotargs)
    if pltClose:
        plt.close()
    graph.update(presets["Nxlog"])
    graph.plot(filesave=filesave + "Ndepthlog", **plotargs)
    if pltClose:
        plt.close()
    # save V=0 doping values
    N0V_T, N0V_N = [], []
    for curve in curves:
        N = curve.y(alter=presets["Nxlog"]["alter"][1])
        N0V_T.append(curve.attr("temperature [k]"))
        N0V_N.append(N[2])
    N0V = Curve(
        [N0V_T, N0V_N],
        {
            "label": "N$_\\mathrm{CV}$ (0V)",
            "linestyle": "none",
            "linespec": "s",
            "markeredgewidth": 0,
            "color": "b",
        },
    )
    for c in range(len(graph) - 1, -1, -1):  # remove unnecessary curves
        if graph[c] in curves:
            graph.curve_delete(c)

    # Fit Mott-Schottky curves
    Ncvminmax0 = [np.inf, -np.inf]
    Ncvminmax1 = [np.inf, -np.inf]
    graphVbi.append(Curve([[], []], CurveArrheniusExtrapolToZero.attr))
    graphVbi.append(Curve([[], []], {"linestyle": "none"}))
    graphVbi.append(Curve([[], []], CurveArrheniusExtrapolToZero.attr))
    graphVbi.append(Curve([[], []], {"linestyle": "none"}))
    graphVbi.append(N0V)
    graphSmart = Graph("", **newGraphKwargs)
    numCurves = len(graph)
    for curve in range(numCurves):  # number of Curves in graph will change
        c = graph[curve]
        c.update({"linewidth": 0.25})
        new = c.CurveCV_fitVbiN(ROIfit, silent=True)
        smart = c.CurveCV_fitVbiN(c.smartVlim_MottSchottky(Vlim=ROIsmart), silent=True)
        if isinstance(new, CurveCV):
            graph.append(new)
            graph[-1].update({"linespec": "--", "color": c.attr("color")})
            Vbi, Ncv = new.attr("_popt")[0], new.attr("_popt")[1]
            graphVbi[0].appendPoints([c.attr("temperature [k]")], [Vbi])
            graphVbi[1].appendPoints([c.attr("temperature [k]")], [Ncv])
            Ncvminmax0 = [min(Ncvminmax0[0], Ncv), max(Ncvminmax0[1], Ncv)]
        if isinstance(smart, CurveCV):
            graphSmart.append(smart)
            graphSmart[-1].update({"linespec": "--", "color": c.attr("color")})
            Vbi, Ncv = smart.attr("_popt")[0], smart.attr("_popt")[1]
            graphVbi[2].appendPoints([c.attr("temperature [k]")], [Vbi])
            graphVbi[3].appendPoints([c.attr("temperature [k]")], [Ncv])
            Ncvminmax1 = [min(Ncvminmax1[0], Ncv), max(Ncvminmax1[1], Ncv)]
    graphVbi.update({"legendtitle": "Mott-Schottky fit"})
    graphVbi[0].update(
        {
            "linespec": "o",
            "color": "k",
            "label": "Built-in voltage (same Vlim)",
            "markeredgewidth": 0,
        }
    )
    graphVbi[1].update(
        {"linespec": "x", "color": "k", "label": "N$_\\mathrm{CV}$ (same Vlim)"}
    )
    graphVbi[2].update(
        {
            "linespec": "o",
            "color": "r",
            "label": "Built-in voltage (adaptative Vlim)",
            "markeredgewidth": 0,
        }
    )
    graphVbi[3].update(
        {"linespec": "x", "color": "r", "label": "N$_\\mathrm{CV}$ (adaptative Vlim)"}
    )
    graphVbi.update(
        {
            "xlabel": graphVbi.formatAxisLabel(["Temperature", "T", "K"]),
            "ylabel": graphVbi.formatAxisLabel(["Built-in voltage", "V_{bi}", "V"]),
        }
    )
    graphVbi.plot(filesave=filesave + "VbiT", **plotargs)
    if pltClose:
        plt.close()
    for curve in graphVbi:
        curve.visible(not curve.visible())
    graphVbi.update(
        {
            "xlabel": graphVbi.formatAxisLabel(["Temperature", "T", "K"]),
            "ylabel": presets["NV"]["ylabel"],
            "typeplot": "semilogy",
        }
    )
    graphVbi.plot(filesave=filesave + "NcvT", **plotargs)
    if pltClose:
        plt.close()
    # Mott schottky, then N vs V, with fit curves - same Vlim
    graph.update(presets["MS"])
    graph.update({"legendtitle": "Mott-Schottky fit (same Vlim)"})
    graph.update({"axvline": ROIfit})
    graph.plot(filesave=filesave + "MottSchottkySameVlim", **plotargs)
    if pltClose:
        plt.close()
    graph.update(presets["NVlog"])
    graph.update({"ylim": [0.75 * Ncvminmax0[0], 2.2 * Ncvminmax0[1]]})
    graph.plot(filesave=filesave + "NVlogSameVlim", **plotargs)
    if pltClose:
        plt.close()
    graph.update({"axvline": "", "legendtitle": ""})
    # Mott-Schottky, then N vs V, with fit lines - adaptative ROI
    for i in range(len(graph) - 1, numCurves - 1, -1):
        graph.curve_delete(i)
    graph.merge(graphSmart)
    graph.update(presets["MS"])
    graph.update({"legendtitle": "Mott-Schottky fit (adaptative Vlim)"})
    graph.plot(filesave=filesave + "MottSchottkyAdaptative", **plotargs)
    if pltClose:
        plt.close()
    graph.update(presets["NVlog"])
    graph.update({"ylim": [0.75 * Ncvminmax1[0], 2.2 * Ncvminmax1[1]]})
    graph.plot(filesave=filesave + "NVlogAdaptative", **plotargs)
    if pltClose:
        plt.close()
    graph.update({"legendtitle": "", "ylim": [0.5 * Ncvminmax1[0], 5 * Ncvminmax1[1]]})

    # graph phase
    graphPhase.update({"alter": "", "ylim": [0, 90]})
    f = graphPhase.curve(0).x()
    #    graphPhase.append(Curve([[min(f), max(f), max(f), min(f)], [0, 0, 20, 20]],{'type': 'fill', 'facecolor': [1,0,0,0.5], 'linewidth': 0}))
    graphPhase.append(
        Curve(
            [[min(f), max(f)], [20, 20]],
            {"color": [1, 0, 0], "linewidth": 2, "linespec": "--"},
        )
    )
    graphPhase.plot(filesave=filesave + "phase", **plotargs)
    if pltClose:
        plt.close()

    if len(WARNINGS) > 0:
        msg = "Enf of process C-V. Got warnings along the way. See above or summary"
        print(msg)
        for msg in WARNINGS:
            print(msg)
    else:
        print("End of process C-V, successful.")
    return graph


def script_processCf(folder, legend="minmax", pltClose=True, newGraphKwargs={}):
    """
    legend: possible values: 'no', 'minmax', 'all'
    """
    print("Script process C-f")
    newGraphKwargs = copy.deepcopy(newGraphKwargs)
    newGraphKwargs.update({"silent": True})
    graph = Graph("", **newGraphKwargs)
    graphPhase = Graph("", **newGraphKwargs)
    graphNyqui = Graph("", **newGraphKwargs)
    graphBode = Graph("", **newGraphKwargs)
    # list possible files
    listdir = []
    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)):
            fileName, fileExt = os.path.splitext(file)
            fileExt = fileExt.lower()
            line1, line2, line3 = file_read_first3lines(os.path.join(folder, file))
            if GraphCf.isFileReadable(
                fileName, fileExt, line1=line1, line2=line2, line3=line3
            ):
                listdir.append(file)
    if len(listdir) == 0:
        print("Found no suitable file in folder", folder)
        return None
    listdir.sort()
    # open all data files
    for file in listdir:
        print(file)
        graphTmp = Graph(
            os.path.join(folder, file),
            complement={"_CfLoadPhase": True, "_CfLoadNyquist": True},
            **newGraphKwargs
        )
        if len(graph) == 0:
            graph.update(graphTmp.graphinfo)
        graph.append(graphTmp[0])  # append C-f

        if len(graphTmp) > 1:
            dataz, dataphase = None, None
            for curve in graphTmp:
                if curve.attr("_CfPhase", False):
                    graphPhase.append(curve)
                    dataphase = curve.x(), curve.y()
                    graphBode.append(
                        Curve([curve.x(), curve.y()], curve.get_attributes())
                    )
                    graphBode[-1].update({"ax_twinx": 1, "linespec": "--"})
                if curve.attr("_CfNyquist", False):
                    graphNyqui.append(curve)
                    dataz = curve.x(), curve.y()
            if dataz is not None and dataphase is not None:
                graphBode.append(
                    Curve(
                        [dataphase[0], np.sqrt(dataz[0] ** 2 + dataz[1] ** 2)],
                        graphTmp[0].get_attributes(),
                    )
                )
                graphBode[-1].update({"label": ""})
    # colorize curves
    colorscaleThW = Colorscale(np.array([[1, 0.43, 0], [0, 0, 1]]), invert=True)
    graph.colorize(colorscaleThW)
    graphPhase.colorize(colorscaleThW)
    graphNyqui.colorize(colorscaleThW)
    graphBode.colorize(colorscaleThW, sameIfEmptyLabel=True)
    # title, labels
    lbl = graph.curve(-1).attr("label").replace(" K", "K").split(" ")
    if len(lbl) > 1:
        # title
        graph.update({"title": " ".join(lbl[:-1])})
        graphPhase.update({"title": " ".join(lbl[:-1])})
        graphNyqui.update({"title": " ".join(lbl[:-1])})
        graphBode.update({"title": " ".join(lbl[:-1])})
        # labels
        graph.replace_labels(" ".join(lbl[:-1]), "")
        graphPhase.replace_labels(" ".join(lbl[:-1]), "")
    for curve in graphNyqui:
        curve.update({"label": "{:.0f} K".format(curve.attr("temperature [k]"))})
    for curve in graphBode:
        phaseorz = "phase" if curve.attr("_cfphase") else "|z|"
        temperature = curve.attr("temperature [k]")
        curve.update({"label": "{:.0f} K {}".format(temperature, phaseorz)})
    # mask undesired legends
    maskUndesiredLegends(graph, legend)
    maskUndesiredLegends(graphPhase, legend)
    maskUndesiredLegends(graphNyqui, legend)
    maskUndesiredLegends(graphBode, legend)
    # set xlim
    setXlim(graph, "tight")
    setXlim(graphPhase, "tight")
    setXlim(graphNyqui, "tight")
    setXlim(graphBode, "tight")

    # save
    filesave = os.path.join(
        folder, graph.attr("title").replace(" ", "_") + "_"
    )  # graphIO.filesave_default(self)
    plotargs = {}  # {'ifExport': False, 'ifSave': False}
    graphattr = {}
    for attr in ["alter", "typeplot", "xlim", "ylim", "xlabel", "ylabel"]:
        graphattr.update({attr: graph.attr(attr)})

    # default graph: C vs log(f)
    graph.update({"ylim": [0, np.nan]})
    graph.plot(filesave=filesave + "Clogf", **plotargs)
    if pltClose:
        plt.close()

    # graph 2: derivative
    graph.update({"alter": ["", "CurveCf.y_mdCdlnf"], "typeplot": "semilogx"})
    graph.update(
        {"ylim": [0, np.nan], "ylabel": "Derivative - f dC/df"}
    )  # same math as d Capacitance / d ln(f)
    graph.plot(filesave=filesave + "deriv", **plotargs)
    if pltClose:
        plt.close()

    # graph 3: derivative, zoomed-in
    graph.update({"alter": ["", "CurveCf.y_mdCdlnf"], "typeplot": "semilogx"})
    graph.update({"ylim": [0, 1.5 * max(graph.curve(0).y(alter="CurveCf.y_mdCdlnf"))]})
    graph.plot(filesave=filesave + "derivZoom", **plotargs)
    if pltClose:
        plt.close()

    # graph 4: phase
    graphPhase.update(
        {
            "xlabel": graph.attr("xlabel"),
            "ylabel": graphPhase.formatAxisLabel("Impedance phase [°]"),
            "alter": "",
            "typeplot": "semilogx",
            "ylim": [0, 90],
        }
    )
    f = graphPhase.curve(0).x()
    graphPhase.append(
        Curve(
            [[min(f), max(f)], [20, 20]],
            {"color": [1, 0, 0], "linewidth": 2, "linespec": "--"},
        )
    )
    graphPhase.plot(filesave=filesave + "phase", **plotargs)
    if pltClose:
        plt.close()

    # graph 5: Nyquist
    graphNyqui.update(
        {
            "alter": "",
            "xlabel": graphNyqui.formatAxisLabel(GraphCf.AXISLABELSNYQUIST[0]),
            "ylabel": graphNyqui.formatAxisLabel(GraphCf.AXISLABELSNYQUIST[1]),
        }
    )
    # Nyquist limits
    limmaxx, limmaxy = -np.inf, -np.inf
    for curve in graphNyqui:
        limmaxx = max(limmaxx, max(curve.x()))
        limmaxy = max(limmaxy, max(curve.y()))
    limmax = max(limmaxx, limmaxy)
    upd = {  # default: square plot
        "xlim": [0, limmax],
        "ylim": [0, limmax],
        "subplots_adjust": [0.15, 0.15, 0.95, 0.95],
        "figsize": [5, 5],
    }
    if limmaxy <= 0.5 * limmaxx:  # if data fit into a 2:1 rectangle
        upd.update(
            {
                "ylim": [0, limmax * 0.5],
                "subplots_adjust": [0.15, 0.15, 0.95, 0.95],
                "figsize": [8, 4],
            }
        )
    graphNyqui.update(upd)
    graphNyqui.plot(filesave=filesave + "Nyquist", **plotargs)
    if pltClose:
        plt.close()

    # graph 6: image derivative, omega (aka f) vs T (or rather arrhenius)
    graphImage = Graph()
    x = [1] + list(graph.curve(0).x() * 2 * np.pi)  # convert frequency into omega
    matrix = [x]
    tempmin, tempmax = np.inf, -np.inf
    for curve in graph:
        temperature = curve.attr("temperature", None)
        if temperature is None:
            temperature = curve.attr("temperature [k]", None)
        if temperature is None:
            msg = "Script Cf, image, cannot identify Temperature. File ignored. {}"
            print(msg.format(curve.get_attributes()))
            continue

        tempmin, tempmax = min(tempmin, temperature), max(tempmax, temperature)
        y = [temperature] + list(curve.y(alter="CurveCf.y_mdCdlnf"))
        if len(x) != len(y):
            msg = (
                "WARNING data curve {} not consistent number of points throughout "
                "the input files! File ignored. {}"
            )
            print(msg.format(temperature, curve.attr("filename")))
            continue  # do not work with the data
        matrix.append(y)

    matrix = np.array(matrix)
    if matrix.shape[0] > 1:  # enough datapoints
        # format data
        for line in range(1, matrix.shape[1]):
            freq = matrix[0, line] / 2 / np.pi
            graphImage.append(
                Curve_Image([matrix[:, 0], matrix[:, line]], {"frequency [Hz]": freq})
            )
        # levels -> int value does not seem to work (matplotlib version?)
        m, M = np.inf, -np.inf
        for curve in graphImage:
            m = min(m, np.min(curve.y()[1:]))
            M = max(M, np.max(curve.y()[1:]))
        space = roundSignificant(M / 15, 1)
        levels = list(np.arange(0, M * 1.5, space))
        for i in range(len(levels) - 2, -1, -1):
            if levels[i] > M:
                del levels[-1]
            else:
                break
        if len(levels) > 1 and levels[0] > levels[1]:
            del levels[0]
        # graph size - identical pixel unit size for all image in terms of Delta T
        # and Delta ln(f)
        omegamin, omegamax = np.min(matrix[0, 1:]), np.max(matrix[0, 1:])
        tempdelta = (1000 / tempmin - 1000 / tempmax) * 1
        if tempdelta > 0:
            fdelta = (np.log10(omegamax) - np.log10(omegamin)) * 1
            subadj = [0.8, 0.5, 0.8 + tempdelta, 0.5 + fdelta, "abs"]
            graphImage.update(
                {"subplots_adjust": subadj, "figsize": [3 + tempdelta, 1 + fdelta]}
            )
        else:
            print("script Cf, image, Warning T delta <=0, cannot properly scale image.")
        # ticks
        xtickslabels = [[], []]
        values_base = [400, 350, 300, 250, 200, 170, 150, 130, 115, 100, 80, 60, 50, 40]
        for value in values_base:
            xtickslabels[0].append(1000 / value)
            xtickslabels[1].append(value)
        for value in np.arange(np.min(values_base), np.max(values_base), 10):
            if value not in xtickslabels[1]:
                xtickslabels[0].append(1000 / value)
                xtickslabels[1].append("")
        # display image as desired
        attrs = {
            "typeplot": "semilogy",
            "alter": ["CurveArrhenius.x_1000overK", ""],
            "ylim": [omegamin, omegamax],
            "twinx_ylim": [omegamin / 2 / np.pi, omegamax / 2 / np.pi],
            "ylabel": graphImage.formatAxisLabel(
                ["Angular frequency $\\omega$", "", "s$^{-1}$"]
            ),
            "twinx_ylabel": graph.attr("xlabel"),
            "xlabel": graphImage.formatAxisLabel(["Temperature", "T", "K"]),
            "twiny_xlabel": graphImage.formatAxisLabel(
                ["1000/Temperature", "", "1000/K"]
            ),
            "xlim": [tempmax, tempmin],
            "twiny_xlim": [tempmax, tempmin],
            "xtickslabels": xtickslabels,
        }
        graphImage.update(attrs)
        graphImage[0].update(
            {
                "datafile_xy1rowcol": 1,
                "cmap": "magma_r",
                "type": "contourf",
                "colorbar": {
                    "label": graph.attr("ylabel"),  # '-dC / dln(f)',
                    "adjust": [1.18, 0, 0.05, 1, "ax"],
                },
                "levels": levels,
                "extend": "both",
            }
        )
        # For secondary axes
        attx = {"ax_twinx": 1, "labelhide": 1, "label": "Ax twinx", "type": "semilogy"}
        atty = {"ax_twiny": 1, "labelhide": 1, "label": "Ax twiny"}
        graphImage.append(Curve([[1], [1]], attx))
        graphImage.append(Curve([[1], [1]], atty))
        # Fit curve to play with
        image_fit_data = [[100, 150, 200, 250, 300, 350], [1, 1, 1, 1, 1, 1]]
        image_fit_attr = {
            "_arrhenius_variant": "Cfdefault",
            "_fitfunc": "func_Arrhenius",
            "_popt": [0.15, 1e8],
            "_fitroi": [3, 10],
            "color": "k",
            "curve": "Curve Arrhenius",
            "label": "Fit curve to toy with",
            "labelhide": 1,
            "linestyle": "none",
        }
        graphImage.append(CurveArrhenius(image_fit_data, image_fit_attr))
        graphImage[-1].updateFitParam(*graphImage[-1].attr("_popt"))
        # plot
        graphImage.plot(filesave=filesave + "image", **plotargs)
        # if pltClose:
        #    plt.close()

    # graph 7: f vs apparent doping
    graph.update(
        {
            "alter": ["CurveCV.x_CVdepth_nm", "x"],
            "typeplot": "semilogy",
            "xlabel": graph.formatAxisLabel(["Apparent depth", "d", "nm"]),
            "ylabel": graph.formatAxisLabel(["Frequency", "f", "Hz"]),
            "xlim": "",
            "ylim": "",
        }
    )
    graph.plot(filesave=filesave + "apparentDepth", **plotargs)
    if pltClose:
        plt.close()

    # Bode plot
    graphBode.update(
        {
            "xlabel": graph.attr("ylabel"),
            "ylabel": "Modulus |Z| [Ohm]",
            "typeplot": "loglog",
            "twinx_ylabel": graphNyqui.formatAxisLabel("Impedance phase [°]"),
            "twinx_ylim": [0, 90],
            "subplots_adjust": [0.15, 0.15],
        }
    )
    if len(graphBode) > 1:
        graphBode[0].update({"labelhide": 1})
        graphBode[1].update({"labelhide": ""})
    graphBode.plot(filesave=filesave + "Bode", **plotargs)
    if pltClose:
        plt.close()

    # restore initial graph
    graph.update(graphattr)
    if pltClose:
        plt.close()
    print(
        "Tip for next step: pick inflection points for different T, then the fit "
        "activation energy."
    )
    print("End of process C-f.")
    return graphImage


def execute_standalone():
    folder = "./../examples/Cf/"
    graph = script_processCf(folder, pltClose=False)

    folder = "./../examples/CV/"
    # graph = script_processCV(folder, ROIfit=[0.15, 0.3], pltClose=True)

    plt.show()


if __name__ == "__main__":
    execute_standalone()
