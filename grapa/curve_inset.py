# -*- coding: utf-8 -*-
"""A subclass of Curve to deal with plot insert

@author: Romain Carron
Copyright (c) 2026, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import os
from typing import Optional

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.curve_subplot import folder_initialdir
from grapa.shared.funcgui import FuncGUI, CurveActionRequestToGui


class Curve_Inset(Curve):
    """
    The purpose is this class is to provid GUI support to create insets
    """

    CURVE = "inset"

    def __init__(self, *args, **kwargs):
        Curve.__init__(self, *args, **kwargs)
        # define default values for important parameters
        if self.attr("insetfile") == "":
            self.update({"insetfile": ""})
        if self.attr("insetcoords") == "":
            self.update({"insetcoords": [0.3, 0.2, 0.4, 0.3]})
        if self.attr("insetupdate") == "":
            self.update({"insetupdate": {}})

        if self.attr("insetfile") == " ":
            if self.attr("subplotfile") not in ["", " "]:
                self.update({"insetfile": self.attr("subplotfile")})
                self.update({"subplotfile": ""})

        self.update({"Curve": Curve_Inset.CURVE})

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)

        graph: Optional[Graph] = None
        if "graph" in kwargs:
            graph = kwargs["graph"]
            del kwargs["graph"]

        insetfile = self.attr("insetfile")
        initialdir = folder_initialdir(insetfile, graph)
        line = FuncGUI(self.update_values_keys, "Set")
        line.set_hiddenvars({"keys": ["insetfile"]})
        line.append_pickfile("file inset", insetfile, initialdir)
        out.append(line)

        # open subplot
        if self.has_attr("insetfile"):
            path = self.attr("insetfile")
            if graph is not None:
                path = graph.filenamewithpath(path)
            path_escaped = (
                os.path.normpath(path).encode("unicode_escape").decode("ascii")
            )
            line = FuncGUI(CurveActionRequestToGui.OPEN_FILE, "Open")
            line.append("file", path_escaped, options={"state": "readonly"})
            out.append(line)

        out.append(
            [
                self.update_values_keys,
                "Set",
                ["coords, in figure fraction [left, bottom, width, height]"],
                [self.attr("insetcoords")],
                {"keys": ["insetcoords"]},
            ]
        )
        out.append(
            [
                self.update_values_keys,
                "Set",
                ["update inset"],
                [self.attr("insetupdate")],
                {"keys": ["insetupdate"]},
            ]
        )
        out.append([self.print_help, "Help!", [], []])  # one line per function
        return out

    def print_help(self):
        print("*** *** ***")
        print(
            "Class Curve_Inset facilitates the creation and customization",
            "of insets inside a Graph.",
        )
        print("Important parameters:")
        print(
            "- insetfile: a path to a saved Graph, either absolute or",
            "relative to the main Graph.",
        )
        print(
            "  if set to " " (whitespace) or if no Curve are found in the",
            "given file, the next Curves of the main graph will be",
            "displayed in the inset.",
        )
        print(
            "- insetcoords: coordinates of the inset axis, relative to the",
            "main graph. Prototype: [left, bottom, width, height]",
        )
        print(
            "- insetupdate: a dict which will be applied to the inset",
            "graph. Provides basic support for run-time customization of",
            "the inset graph.",
        )
        print("  Examples: {'fontsize':8}, or {'xlabel': 'An updated label'}")
        return True
