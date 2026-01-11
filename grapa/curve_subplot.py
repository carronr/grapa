# -*- coding: utf-8 -*-
"""A subclass of Curve to deal with subplots

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import os
import logging
from typing import Optional

import numpy as np

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.shared.maths import roundSignificant
from grapa.shared.error_management import issue_warning, IncorrectInputError
from grapa.shared.funcgui import FuncGUI, CurveActionRequestToGui
from grapa.plot.plot_graph_aux import SubplotsAdjuster

logger = logging.getLogger(__name__)


def folder_initialdir(file: str, graph: Optional[Graph]) -> str:
    """Returns folder of subplotfile, intended as initialdir"""
    if os.path.isabs(file):
        return os.path.dirname(file)

    folder = ""
    if graph is not None:
        folder = os.path.dirname(str(graph.filename))
        subplotfolder = os.path.dirname(file)
        if subplotfolder != "":
            post = os.path.normpath(os.path.join(folder, subplotfolder))
            folder = os.path.relpath(post, folder)
            return post
    return folder


class Curve_Subplot(Curve):
    """
    Curve_Subplot provides GUI support to handle subplots within a Graph.
    The main graph will be subdivised in an array of subplots.
    """

    CURVE = "subplot"

    SPUKEYS = ["xlim", "ylim", "xlabel", "ylabel"]

    def __init__(self, *args, **kwargs):
        Curve.__init__(self, *args, **kwargs)
        # define default values for important parameters
        self.update({"Curve": Curve_Subplot.CURVE})

        def check_attr(key, default, typ_):
            val = self.attr(key)
            if val == "":
                self.update({key: default})
            elif not isinstance(val, typ_):
                try:
                    new = typ_(val)
                    self.update({key: new})
                except ValueError as e:
                    msg = "Curve_Subplot init check_attr: {}. ValueError {}."
                    issue_warning(logger, msg.format(val, e), exc_info=True)

        check_attr("subplotfile", " ", str)
        check_attr("subplotrowspan", 1, int)
        check_attr("subplotcolspan", 1, int)
        check_attr("subplotupdate", {}, dict)
        if self.attr("subplotfile") == " ":
            if self.attr("insetfile") not in ["", " "]:
                self.update({"subplotfile": self.attr("insetfile")})
                self.update({"insetfile": ""})

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        graph: Optional[Graph] = None
        if "graph" in kwargs:
            graph = kwargs["graph"]
            del kwargs["graph"]

        out += _funclistgui_subplotfile(self, graph)
        out += _funclistgui_open(self, graph)
        out += _funclistgui_colrowspan(self)
        out += _funclistgui_subplotupdate(self)
        out += _funclistgui_subplotupdate_shortcuts(self)

        if graph is not None and isinstance(graph, Graph):
            out.append([None, "Graph options", [], []])
            out += _funclistgui_ncols_transpose_id(self, graph)
            out += _funclistgui_widthheightratios(self, graph)
            out += _funclistgui_subplotadjust(self, graph)
            out += _funclistgui_update_spa_figsize_abs(self, graph)

        out.append([self.print_help, "Help!", [], []])  # one line per function
        return out

    def update_spa_figsize_abs(
        self,
        panelsize,
        margin,
        ncols=2,
        nrows=2,
        graph: Optional[Graph] = None,  # actually not optional. Because of GUI.
        **_kwargs
    ):
        """
        Updates the subplots_adjustupdate and figsize, based on margins and panelsize
        CAUTION: modifies object graph
        """
        if graph is None:
            msg = "curve_subplot update_spa_figsize_abs, must provide argument graph."
            raise IncorrectInputError(msg)

        if not isinstance(panelsize, list):
            panelsize = list(panelsize)
        if not isinstance(margin, list):
            margin = list(margin)
        try:
            ncols = int(ncols)
            nrows = int(nrows)
            panelsize = [float(val) for val in panelsize]
            margin = [float(val) for val in margin]
        except (TypeError, ValueError) as e:
            print("Exception {}: {}".format(type(e), e))
            return False
        while len(margin) < 6:
            margin.append(1)
        while len(panelsize) < 6:
            panelsize.append(4)
        figsize = [
            margin[0] + margin[2] + (ncols - 1) * margin[4] + ncols * panelsize[0],
            margin[1] + margin[3] + (nrows - 1) * margin[5] + nrows * panelsize[1],
        ]
        subplots_adjust = [
            margin[0] / figsize[0],
            margin[1] / figsize[1],
            1 - margin[2] / figsize[0],
            1 - margin[3] / figsize[1],
            margin[4] / panelsize[0],
            margin[5] / panelsize[1],
        ]
        # print({"figsize": figsize, "subplots_adjust": subplots_adjust})
        graph.update(
            {
                "figsize": figsize,
                "subplots_adjust": subplots_adjust,
                "subplotsncols": ncols,
            }
        )
        if int(graph.attr("subplotstranspose", 0)):
            graph.update({"subplotsncols": nrows})
        return True

    def update_spu(self, *args, **_kwargs):
        """updates the subplotupdate, carrying information to the subplot"""
        spu = self.attr("subplotupdate")
        if not isinstance(spu, dict):
            spu = {}
        for i, key in enumerate(self.SPUKEYS):
            spu.update({key: args[i]})
        self.update({"subplotupdate": spu})
        return True

    def print_help(self):
        print("*** *** ***")
        print(
            "Class Curve_Subplot facilitates the creation and customization of",
            "subplots inside a Graph.",
        )
        print("The main graph will be subdivised in an array of subplots.")
        string = (
            "Important parameters:\n"
            + "- subplotfile: saved Graph file which must be shown in subplot. The next"
            + "Curves will also be shown in the created axis, until the next Curve with"
            + "same Curve_Subplot type.\n"
            + "- subplotcolspan: on how many column the subplot will be plotted.\n"
            + "- subplotrowspan: on how many rows the subplot will be plotted.\n"
            + "- subplotupdate: a dict which will be applied to the subplot"
            + "Graph. Provides basic support for run-time customization of"
            + "the inset graph."
        )
        print(string)
        print("Examples: {'fontsize': 8}, or {'xlabel': 'An updated label'}")
        print()
        return True


def _funclistgui_subplotfile(curve: Curve, graph: Optional[Graph]):
    """create Curve Actions lines"""
    subplotfile = curve.attr("subplotfile")
    initialdir = folder_initialdir(subplotfile, graph)
    line = FuncGUI(curve.update_values_keys, "Set")
    line.set_hiddenvars({"keys": ["subplotfile"]})
    line.append_pickfile("file subplot", subplotfile, initialdir)
    return [line]


def _funclistgui_open(curve: Curve, graph: Optional[Graph]):
    """open subplot"""
    if not curve.has_attr("subplotfile"):
        return []

    path = curve.attr("subplotfile")
    if graph is not None:
        path = graph.filenamewithpath(path)
    path_escap = os.path.normpath(path).encode("unicode_escape").decode("ascii")
    line = FuncGUI(CurveActionRequestToGui.OPEN_FILE, "Open")
    line.append("file", path_escap, options={"state": "readonly"})
    return [line]


def _funclistgui_colrowspan(curve: Curve):
    # colspan, rowspan
    values = ["", "1", "2", "3", "4", "1 number"]
    line = FuncGUI(curve.update_values_keys, "Set")
    line.set_hiddenvars({"keys": ["subplotcolspan", "subplotrowspan"]})
    line.appendcbb("colspan", curve.attr("subplotcolspan"), values, bind="beforespace")
    line.appendcbb("rowspan", curve.attr("subplotrowspan"), values, bind="beforespace")
    return [line]


def _funclistgui_subplotupdate(curve: Curve):
    """shortcuts for subplotupdate"""
    line = FuncGUI(curve.update_values_keys, "Set", {"keys": ["subplotupdate"]})
    line.append("update subplot", curve.attr("subplotupdate"))
    return [line]


def _funclistgui_subplotupdate_shortcuts(curve: Curve_Subplot):
    """shortcuts for subplotupdate: xlim, ylim, xlabel, ylabel"""
    spu = curve.attr("subplotupdate")
    if not isinstance(spu, dict):
        spu = {}
    spuvals = [spu[key] if key in spu else "" for key in curve.SPUKEYS]
    spucust = [{}, {}, {}, {}]
    for i in [2, 3]:
        if len(spuvals[i]) > 5:
            spucust[i] = {"width": min(15, len(spuvals[i]))}
    line = FuncGUI(curve.update_spu, "Set")
    for label, value, options in zip(curve.SPUKEYS, spuvals, spucust):
        line.append(label, value, options=options)
    return [line]


def _funclistgui_ncols_transpose_id(_curve: Curve, graph: Graph):
    ncols = int(graph.attr("subplotsncols", 2))
    line = FuncGUI(graph.update_values_keys, "Set")
    line.set_hiddenvars({"keys": ["subplotsncols", "subplotstranspose", "subplotsid"]})
    line.appendcbb("n cols", ncols, ["", "1", "2", "3", "4", "1 number"])
    line.appendcbb(
        "transpose",
        str(graph.attr("subplotstranspose")),
        ["", "0 False", "1 True"],
        bind="beforespace",
    )
    line.append("show id? or (" + "\u0394x,\u0394y)", str(graph.attr("subplotsid")))
    return [line]


def _funclistgui_widthheightratios(_curve: Curve, graph: Graph):
    """width, height ratios"""
    line = FuncGUI(graph.update_values_keys, "Set")
    line.set_hiddenvars({"keys": ["subplotswidth_ratios", "subplotsheight_ratios"]})
    line.append("width ratios (i.e. [1,2,1])", graph.attr("subplotswidth_ratios"))
    line.append("height_ratios", graph.attr("subplotsheight_ratios"))
    return [line]


def _funclistgui_subplotadjust(_curve: Curve, graph: Graph):
    """subplots_adjust"""
    spa_attr = graph.attr("subplots_adjust")
    line = FuncGUI(graph.update_values_keys, "Set", {"keys": ["subplots_adjust"]})
    line.append("subplots_adjust [left, bottom, right, top, wspace, hspace]", spa_attr)
    return [line]


def _funclistgui_update_spa_figsize_abs(curve: Curve_Subplot, graph: Graph):
    spa_attr = graph.attr("subplots_adjust")
    # plot dimensions absolute numbers (inch)
    nsub = [
        1 for curve in graph if isinstance(curve, Curve_Subplot) and curve.visible()
    ]
    nsub = np.sum(nsub)
    # ncols  # already defined
    ncols = int(graph.attr("subplotsncols", 2))
    nrows = np.ceil(nsub / ncols)
    if int(graph.attr("subplotstranspose", 0)):
        ncols, nrows = nrows, ncols
    fs = graph.attr("figsize")
    if not isinstance(fs, (list, tuple)) or len(fs) < 2:
        fs = Graph.FIGSIZE_DEFAULT
    # spa_attr  # already defined
    spa_default = SubplotsAdjuster.default()
    spa = SubplotsAdjuster.merge(spa_default, spa_attr, fs)
    margin = [
        fs[0] * spa["left"],
        fs[1] * spa["bottom"],
        fs[0] * (1 - spa["right"]),
        fs[1] * (1 - spa["top"]),
    ]
    plotarea = [fs[0] - margin[0] - margin[2], fs[1] - margin[1] - margin[3]]
    panelsize = [
        plotarea[0] / (ncols + (ncols - 1) * spa["wspace"]),
        plotarea[1] / (nrows + (nrows - 1) * spa["hspace"]),
    ]
    margin += [panelsize[0] * spa["wspace"], panelsize[1] * spa["hspace"]]
    margin = roundSignificant(margin, 6)
    panelsize = roundSignificant(panelsize, 6)
    line = FuncGUI(curve.update_spa_figsize_abs, "Set")
    line.set_hiddenvars({"graph": graph})
    line.append(
        "figure dimensions from panel size [inch]", panelsize, options={"width": 18}
    )
    line.append("", None, widgetclass="Frame")
    line.append("         margin on each side [inch]", margin, options={"width": 25})
    line.append("ncols", int(ncols), keyword="ncols")
    line.append("nrows", int(nrows), keyword="nrows")
    return [line]
