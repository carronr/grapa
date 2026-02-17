# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:02:34 2020

@author: Romain Carron
Copyright (c) 2026, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import sys
import copy
from typing import Optional, Dict, Any

import numpy as np

path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if path not in sys.path:
    sys.path.append(path)

from grapa.graph import Graph
from grapa.shared.maths import roundSignificantRange, roundSignificant, is_number
from grapa.shared.mpl_figure_factory import MplFigureFactory
from grapa.datatypes.curveJscVoc import GraphJscVoc, CurveJscVoc


def script_processJscVoc(
    file,
    pltClose=True,
    newGraphKwargs={},
    ROIJsclim=None,
    ROIVoclim=None,
    ROITlim=None,
    figure_factory: Optional[MplFigureFactory] = None,
):
    """
    TO IMPLEMENT PARAMETERS:
    Jo, A fit: JSC LIM, VOC LIM
    Voc vs T: K range below max
    """
    newGraphKwargs = copy.deepcopy(newGraphKwargs)
    newGraphKwargs.update({"silent": True})
    if figure_factory is None:
        figure_factory = MplFigureFactory()

    print("Script process Jsc-Voc")

    graph = Graph(file, **newGraphKwargs)

    print("WARNING: Cell area assumed to be", graph[0].getArea(), "cm-2.")

    lbl = graph[0].attr("label")
    if len(lbl) > 1:
        graph.update({"title": lbl})

    labelT = graph.format_axis_label(["Temperature", "T", "K"])
    labelA = graph.format_axis_label(["Diode ideality factor", "A", ""])
    label1000AT = graph.format_axis_label(["1000 / (A*Temperature)", "", "K$^{-1}$"])
    labellnJ0 = graph.format_axis_label(["ln(J$_0$)", "", "mAcm$^{-2}$"])
    labelVoc = graph.format_axis_label(["Voc", "", "V"])

    presets = {}
    presets.update(
        {
            "default": {
                "ylim": "",
                "xlim": "",
                "alter": "",
                "typeplot": "semilogy",
                "xlabel": graph.attr("xlabel"),
                "ylabel": graph.attr("ylabel"),
            }
        }
    )
    presets.update(
        {
            "ideality": {
                "ylim": "",
                "xlim": "",
                "alter": "",
                "typeplot": "",
                "xlabel": labelT,
                "ylabel": labelA,
            }
        }
    )
    presets.update(
        {
            "J0vsAT": {
                "ylim": "",
                "xlim": "",
                "typeplot": "",
                "alter": ["CurveArrhenius.x_1000overK", "CurveArrhenius.y_Arrhenius"],
                "xlabel": label1000AT,
                "ylabel": labellnJ0,
            }
        }
    )
    presets.update(
        {
            "VocvsT": {
                "ylim": "",
                "xlim": [0, ""],
                "typeplot": "",
                "alter": "",
                "xlabel": labelT,
                "ylabel": labelVoc,
            }
        }
    )
    # save
    folder = os.path.dirname(file)
    filesave = os.path.join(
        folder, "JscVoc_" + graph.attr("title").replace(" ", "_") + "_"
    )  # graphIO.filesave_default(self)
    print("filesave", filesave)
    plotargs: Dict[str, Any] = {"figure_factory": figure_factory}
    # also possible e.g. {'if_export': False, 'if_save': False}
    grap2 = copy.deepcopy(graph)
    # default graph: with fits
    graph.update(presets["default"])
    Voclim = roundSignificantRange([0, max(graph[0].x()) * 1.01], 3)
    Jsclim = roundSignificantRange(
        [CurveJscVoc.CST_Jsclim0, max(graph[0].y()) * 1.1], 2
    )
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
    print("Fit J0, A: Voc limits", ", ".join(["{:.3f}".format(v) for v in Voclim]), "V")
    print(
        "Fit J0, A: Jsc limits",
        ", ".join(["{:.2e}".format(v) for v in Jsclim]),
        "mAcm-2",
    )
    for curve in GraphJscVoc.CurveJscVoc_fitNJ0(
        graph, Voclim, Jsclim, 3, True, curve=graph[0], silent=True
    ):
        graph.append(curve)
    graph.plot(filesave=filesave + "fits", **plotargs)
    if pltClose:
        figure_factory.close()
    # Graph ideality factor vs T
    grap2.update(presets["ideality"])
    grap2.append(graph[-3])
    for c in range(len(grap2)):
        grap2[c].visible(not grap2[c].visible())
    grap2.plot(filesave=filesave + "IdealityvsT", **plotargs)
    if pltClose:
        figure_factory.close()
    # Graph ideality factor vs T
    grap2.update(presets["J0vsAT"])
    grap2.curve_delete(-1)
    curve = graph[-1]
    grap2.append(curve)
    curve.visible(not curve.visible())
    ROI = list(
        roundSignificant(
            [min(curve.x_1000overK()) * 0.95, 1.05 * max(curve.x_1000overK())], 4
        )
    )
    grap2.append(graph[-1].CurveArrhenius_fit(ROI, silent=True))
    Ea = grap2[-1].attr("_popt")[0]
    grap2[-1].update({"label": "E$_a$ " + "{:1.3f}".format(Ea) + " eV"})
    grap2.plot(filesave=filesave + "J0vsAT", **plotargs)
    if pltClose:
        figure_factory.close()
    # Voc vs T
    grap2.update(presets["VocvsT"])
    grap2.curve_delete(-1)  # previous fit
    grap2.curve_delete(-1)  # ln(Jo) vs A*T
    curve = grap2[0]
    Tlim = [0.99 * np.min(graph[1].y()), 1.01 * np.max(graph[1].y())]
    Tlim[0] = max(Tlim[0], Tlim[1] - 80)  # restrict fit to highest 100K
    if ROITlim is not None:
        if isinstance(ROITlim, list):
            Tlim = ROITlim
        elif is_number(ROITlim):
            Tlim[0] = ROITlim
    print(
        "Voc vs T: fit range restricted to",
        ", ".join(["{:.1f}".format(T) for T in Tlim]),
        "K",
    )
    res = GraphJscVoc.split_illumination(
        grap2, 3.0, True, Tlim, True, curve=curve, silent=True
    )
    grap2.append(res, idx=2)
    grap2.plot(filesave=filesave + "VocvsT", **plotargs)
    if pltClose:
        figure_factory.close()

    graph[-1].visible(False)
    graph[-3].visible(False)
    print(
        "WARNING: fits limits are chosen automatically. It is YOUR",
        "responsibility to check for the goodness of fit, and manually",
        "decide for the limits!",
    )
    print("End of process Jsc-Voc.")
    return graph


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    file_ = "./../examples/JscVoc/JscVoc_SAMPLE_c2_Values.txt"
    graph_ = script_processJscVoc(file_, pltClose=False)

    plt.show()
