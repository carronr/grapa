# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:03:50 2017

@author: Romain Carron
Copyright (c) 2023, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

from grapa.graph import Graph
from grapa.curve import Curve


class Curve_Subplot(Curve):
    """
    The purpose is this class is to provide GUI support to handle subplots
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
        out.append(
            [
                self.updateValuesDictkeys,
                "Set",
                ["update subplot"],
                [self.attr("subplotupdate")],
                {"keys": ["subplotupdate"]},
            ]
        )
        out.append([self.update_spu, "Set", self.SPUKEYS, spuvals, {}, spucust])
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
            out.append(
                [
                    graph.updateValuesDictkeys,
                    "Set",
                    ["subplots_adjust [left, bottom, right, top, wspace, hspace]"],
                    [graph.attr("subplots_adjust")],
                    {"keys": ["subplots_adjust"]},
                ]
            )
        out.append([self.printHelp, "Help!", [], []])  # one line per function
        return out

    def update_spu(self, *args, **kwargs):
        """updates the subplotupdate, carrying information to the subplot"""
        spu = self.attr("subplotupdate")
        if not isinstance(spu, dict):
            spu = {}
        for i in range(len(self.SPUKEYS)):
            spu.update({self.SPUKEYS[i]: args[i]})
        self.update({"subplotupdate": spu})
        return True

    @staticmethod
    def printHelp():
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
