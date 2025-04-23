# -*- coding: utf-8 -*-
"""A subclass of Curve to deal with subplots

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import numpy as np

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.mathModule import roundSignificant
from grapa.utils.plot_graph_aux import SubplotsAdjuster
from grapa.utils.funcgui import FuncGUI


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
                self.update({key: typ_(val)})

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
        graph = None
        if "graph" in kwargs:
            graph = kwargs["graph"]
            del kwargs["graph"]
        # shortcuts for subplotupdate
        spu = self.attr("subplotupdate")
        if not isinstance(spu, dict):
            spu = {}
        spuvals = [spu[key] if key in spu else "" for key in self.SPUKEYS]
        spucust = [{}, {}, {}, {}]
        for i in [2, 3]:
            if len(spuvals[i]) > 5:
                spucust[i] = {"width": min(15, len(spuvals[i]))}
        # create Curve Actions lines
        out.append(
            [
                self.updateValuesDictkeys,
                "Set",
                ["file subplot"],
                [self.attr("subplotfile")],
                {"keys": ["subplotfile"]},
            ]
        )
        # colspan, rowspan
        dictspan = {
            "field": "Combobox",
            "bind": "beforespace",
            "values": ["", "1", "2", "3", "4", "1 number"],
        }
        out.append(
            [
                self.updateValuesDictkeys,
                "Set",
                ["colspan", "rowspan"],
                [self.attr("subplotcolspan"), self.attr("subplotrowspan")],
                {"keys": ["subplotcolspan", "subplotrowspan"]},
                [dictspan, dictspan],
            ]
        )
        # subplotupdate
        out.append(
            [
                self.updateValuesDictkeys,
                "Set",
                ["update subplot"],
                [self.attr("subplotupdate")],
                {"keys": ["subplotupdate"]},
            ]
        )
        # xlim, ylim, xlabel, ylabel
        out.append([self.update_spu, "Set", self.SPUKEYS, spuvals, {}, spucust])
        # subplotsncols, transpose, id
        if graph is not None and isinstance(graph, Graph):
            out.append([None, "Graph options", [], []])
            ncols = int(graph.attr("subplotsncols", 2))
            out.append(
                [
                    graph.updateValuesDictkeys,
                    "Set",
                    ["n cols", "transpose", "show id? or (" + "\u0394x,\u0394y)"],
                    [
                        ncols,
                        str(graph.attr("subplotstranspose")),
                        str(graph.attr("subplotsid")),
                    ],
                    {"keys": ["subplotsncols", "subplotstranspose", "subplotsid"]},
                    [
                        {
                            "field": "Combobox",
                            "values": ["", "1", "2", "3", "4", "1 number"],
                        },
                        {
                            "field": "Combobox",
                            "values": ["", "0 False", "1 True"],
                            "bind": "beforespace",
                        },
                        {},
                    ],
                ]
            )
            # width, height ratios
            out.append(
                [
                    graph.updateValuesDictkeys,
                    "Set",
                    ["width ratios (i.e. [1,2,1])", "height_ratios"],
                    [
                        graph.attr("subplotswidth_ratios"),
                        graph.attr("subplotsheight_ratios"),
                    ],
                    {"keys": ["subplotswidth_ratios", "subplotsheight_ratios"]},
                ]
            )

            # subplots_adjust
            spa_attr = graph.attr("subplots_adjust")
            out.append(
                [
                    graph.updateValuesDictkeys,
                    "Set",
                    ["subplots_adjust [left, bottom, right, top, wspace, hspace]"],
                    [spa_attr],
                    {"keys": ["subplots_adjust"]},
                ]
            )

            # plot dimensions absolute numbers (inch)
            nsub = [
                1
                for curve in graph
                if isinstance(curve, Curve_Subplot) and curve.visible()
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
            line = FuncGUI(self.update_spa_figsize_abs, "Set")
            line.set_hiddenvars({"graph": graph})
            line.append(
                "figure dimensions from panel size [inch]",
                panelsize,
                options={"width": 18},
            )
            line.append("", None, widgetclass="Frame")
            line.append(
                "         margin on each side [inch]", margin, options={"width": 25}
            )
            line.append("ncols", int(ncols), keyword="ncols")
            line.append("nrows", int(nrows), keyword="nrows")
            out.append(line)

        out.append([self.print_help, "Help!", [], []])  # one line per function
        return out

    def update_spa_figsize_abs(
        self, panelsize, margin, ncols=2, nrows=2, graph=None, **_kwargs
    ):
        """
        Updates the subplots_adjustupdate and figsize, based on margins and panelsize
        CAUTION: modifies object graph
        """
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
        for i in range(len(self.SPUKEYS)):
            spu.update({self.SPUKEYS[i]: args[i]})
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
        print(string),
        print("Examples: {'fontsize': 8}, or {'xlabel': 'An updated label'}")
        print()
        return True
