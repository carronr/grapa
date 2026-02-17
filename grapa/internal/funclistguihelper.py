# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2026, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import logging
from typing import TYPE_CHECKING

from grapa.shared.error_management import issue_warning
from grapa.shared.funcgui import FuncGUI
from grapa.shared.keywords_loader import keywords_curve
from grapa.plot.plot_graph_aux import Plotter

if TYPE_CHECKING:
    from grapa.graph import Graph
    from grapa.curve import Curve


logger = logging.getLogger(__name__)


class FuncListGUIHelper:
    """
    Not really a class, rather a collection of functions that generate FuncGUI items
    to provide to the GUI.
    Related to notably curve typeplot, but not only
    """

    @classmethod
    def typeplot(cls, curve, **kwargs) -> list:
        """Suggests to the user possible actions depending on curve typeplot."""
        out = []
        typeplot = curve.attr("type")
        try:
            graph = kwargs["graph"]
            c = kwargs["graph_i"]
        except KeyError:
            msg = (
                "FuncListGUIHelper.typeplot: missing kwarg 'graph' or 'graph_i',"
                " output not complete. kwargs: {}."
            )
            issue_warning(logger, msg.format(kwargs))
            return out

        try:
            if typeplot == "errorbar":  # helper for type errorbar
                out += cls._errorbar(curve, graph, c)
            elif typeplot == "scatter":  # helper for type scatter
                out += cls._scatter(curve, graph, c)
            elif typeplot.startswith("fill"):  # helper for type fill
                out += cls._fill(curve, graph, c)
            elif typeplot.startswith("boxplot"):  # helper for type boxplot
                out += cls._boxplot(curve, graph, c)
            elif typeplot.startswith("violinplot"):  # helper for type violinplot
                out += cls._violinplot(curve, graph, c)
            elif typeplot == "bar":  # helper for type bar
                out += cls._bar_yetcannotnamefunctionbar(curve, graph, c)
            elif typeplot == "barh":  # helper for type bar
                out += cls._barh(curve, graph, c)
        except Exception as e:
            logger.error("FuncListGUIHelper.typeplot", exc_info=True)
            raise e
        return out

    @classmethod
    def _errorbar(cls, curve, graph, c):
        out = []
        types = []
        if c + 1 < len(graph):
            types.append(graph[c + 1].attr("type"))
        if c + 2 < len(graph):
            types.append(graph[c + 2].attr("type"))
        xyerr_or = ""
        if len(types) > 0:
            choices = ["", "errorbar_xerr", "errorbar_yerr"]
            labels = ["next Curves type: (" + str(c + 1) + ")"]
            labels += ["(" + str(i) + ")" for i in range(c + 2, c + 1 + len(types))]
            line = FuncGUI(curve.update_scatter_next_curves, "Save")
            line.set_hiddenvars({"graph": graph, "graph_i": c})
            for i, labelvalue in enumerate(labels):
                line.appendcbb(labelvalue, types[i], choices)
            out.append(line)
            xyerr_or = "OR "
        # xerr, yerr values
        line = FuncGUI(curve.update_values_keys, "Save")
        line.set_hiddenvars({"keys": ["xerr", "yerr"]})
        line.append(xyerr_or + "error values x", curve.attr("xerr"))
        line.append("y", curve.attr("yerr"))
        out.append(line)
        # capsize, ecolor
        at = ["capsize", "ecolor"]
        line = FuncGUI(curve.update_values_keys, "Save", hiddenvars={"keys": at})
        for attr in at:
            line.append(attr, curve.attr(attr))
        out.append(line)
        return out

    @classmethod
    def _scatter(cls, curve, graph, c):
        out = []
        types = []
        if c + 1 < len(graph):
            types.append(graph[c + 1].attr("type"))
        if c + 2 < len(graph):
            types.append(graph[c + 2].attr("type"))
        if len(types) > 0:
            choices = ["", "scatter_c", "scatter_s"]
            labels = ["next Curves type: (" + str(c + 1) + ")"] + [
                "(" + str(i) + ")" for i in range(c + 2, c + 1 + len(types))
            ]
            line = FuncGUI(curve.update_scatter_next_curves, "Save")
            line.set_hiddenvars({"graph": graph, "graph_i": c})
            for i, labelvalue in enumerate(labels):
                line.appendcbb(labelvalue, types[i], choices)
            out.append(line)
        # cmap, vminmax
        keys = ["cmap", "vminmax"]
        line = FuncGUI(curve.update_values_keys, "Save", hiddenvars={"keys": keys})
        for key in keys:
            line.append(key, curve.attr(key))
        out.append(line)
        return out

    @classmethod
    def _fill(cls, curve, _graph, _c):
        out = []
        at = ["fill", "hatch", "fill_padto0"]
        at2 = list(at)
        at2[2] = "pad to 0"
        val = [curve.attr(a) for a in at]
        line = FuncGUI(curve.update_values_keys, "Save", hiddenvars={"keys": at})
        line.appendcbb(at2[0], val[0], ["True", "False"])
        line.appendcbb(at2[1], val[1], ["", ".", "+", "/", r"\\"], options={"width": 7})
        line.appendcbb(at2[2], val[2], ["", "True", "False"])
        out.append(line)
        return out

    @classmethod
    def _boxplot(cls, curve: "Curve", graph: "Graph", _c):
        out = []
        # basic parameters
        at = ["boxplot_position", "color"]
        at2 = ["position", "color"]
        line = FuncGUI(curve.update_values_keys, "Save", hiddenvars={"keys": at})
        for i, value in enumerate(at):
            line.append(at2[i], curve.attr(value))
        out.append(line)
        # additional parameters
        at = ["widths", "notch", "vert", "showfliers"]
        hv = {
            "keys": at,
            "graph": graph,
            "also_attr": ["type"],
            "also_vals": ["boxplot"],
        }
        txt = "Applies to all data in boxplot"
        line = FuncGUI(curve.update_values_keys_graph_condition, "Save", hiddenvars=hv)
        line.set_tooltiptext(txt)
        line.append(at[0], curve.attr(at[0]))
        line.appendcbb(at[1], curve.attr(at[1]), ["True", "False"])
        line.appendcbb(at[2], curve.attr(at[2]), ["True", "False"])
        line.appendcbb(at[3], curve.attr(at[3]), ["True", "False"])
        out.append(line)
        # another batch of additional parameters
        at = ["patch_artist"]
        hv = {
            "keys": at,
            "graph": graph,
            "also_attr": ["type"],
            "also_vals": ["boxplot"],
        }
        txt = "Applies to all data in boxplot."
        txt += "\n- patch_artist: to color box. Best if color specifies transparency."
        line = FuncGUI(curve.update_values_keys_graph_condition, "Save", hiddenvars=hv)
        line.set_tooltiptext(txt)
        line.appendcbb(at[0], curve.attr(at[0]), ["", "True", "False"])
        out.append(line)
        # seaborn stripplot and swamplot
        out += Plotter.funcListGUI(curve)
        return out

    @classmethod
    def _violinplot(cls, curve: "Curve", _graph, _c):
        out = []
        at = ["boxplot_position", "color"]
        at2 = ["position", "color"]
        line = FuncGUI(curve.update_values_keys, "Save", hiddenvars={"keys": at})
        for i, value in enumerate(at):
            line.append(at2[i], curve.attr(value))
        out.append(line)
        # seaborn stripplot and swamplot
        out += Plotter.funcListGUI(curve)
        return out

    @staticmethod
    def _set_xytickslabels_from_listlabels(
        graph, listlabels, curve: "Curve" = None, key="xtickslabels", **_kwargs
    ):
        listlabels = list(listlabels)
        tickslabels = [list(curve.x()), listlabels]
        graph.update({key: tickslabels})
        return True

    @classmethod
    def _bar_yetcannotnamefunctionbar(cls, curve, graph, _c):
        out = []
        # quick modifications to most useful parameters
        at = ["width", "align", "bottom"]
        bottomlist = ["", "bar_first", "bar_previous", "bar_next", "bar_last"]
        line = FuncGUI(curve.update_values_keys, "Save", hiddenvars={"keys": at})
        line.append(at[0], curve.attr(at[0]))
        line.appendcbb(at[1], curve.attr(at[1]), ["", "center", "edge"])
        line.appendcbb(at[2], curve.attr(at[2]), bottomlist)
        msg = (
            'Tip: with align "edge", use also negative width value\nbottom: '
            "value, list, or keyword"
        )
        line.set_tooltiptext(msg)
        out.append(line)
        # xtickslabels from values
        tickslabels = graph.attr("xtickslabels")
        if isinstance(tickslabels, list) and len(tickslabels) > 1:
            values = list(tickslabels[1])
        else:
            values = curve.attr("_bar_labels")
        if len(values) == 0:
            values = [str(v) for v in list(curve.x())]
        line = FuncGUI(cls._set_xytickslabels_from_listlabels, "Save")
        line.set_hiddenvars({"curve": curve, "key": "xtickslabels"})
        line.append("xticks from list of labels", values)
        line.set_tooltiptext(
            "Values taken from property xtickslabels, "
            "or _bar_labels. Length must match Curve length."
        )
        out.append(line)
        return out

    @classmethod
    def _barh(cls, curve: "Curve", graph: "Graph", _c):
        out = []
        # quick modifications to most useful parameters
        at = ["height", "align", "left"]
        leftlist = ["", "bar_first", "bar_previous", "bar_next", "bar_last"]
        line = FuncGUI(curve.update_values_keys, "Save", hiddenvars={"keys": at})
        line.append(at[0], curve.attr(at[0]))
        line.appendcbb(at[1], curve.attr(at[1]), ["", "center", "edge"])
        line.appendcbb(at[2], curve.attr(at[2]), leftlist)
        line.set_tooltiptext("Tip: with align 'edge', use also negative height value")
        out.append(line)
        # xtickslabels from values
        tickslabels = graph.attr("ytickslabels")
        if isinstance(tickslabels, list) and len(tickslabels) > 1:
            values = list(tickslabels[1])
        else:
            values = str(curve.attr("_bar_labels"))
        if len(values) == 0:
            values = [str(v) for v in list(curve.x())]
        line = FuncGUI(cls._set_xytickslabels_from_listlabels, "Save")
        line.set_hiddenvars({"curve": curve, "key": "ytickslabels"})
        line.append("yticks from list of labels", values)
        line.set_tooltiptext(
            "Values taken from property ytickslabels, "
            "or _bar_labels. Length must match Curve length."
        )
        out.append(line)
        return out

    @staticmethod
    def offset_muloffset(curve: "Curve", **_kwargs) -> list:
        """Suggests to the user to modify offset and muloffset keywords
        if the curve has these attributes."""
        out = []
        if curve.has_attr("offset") or curve.has_attr("muloffset"):
            kw_curve = keywords_curve()
            at = ["offset", "muloffset"]
            line = FuncGUI(curve.update_values_keys, "Modify screen offsets")
            line.set_hiddenvars({"keys": at})
            line.set_funcdocstring_alt("Modify keywords {} and {}.".format(*at))
            for key in at:
                vals = []
                if key in kw_curve["keys"]:
                    i = kw_curve["keys"].index(key)
                    vals = [str(v) for v in kw_curve["guiexamples"][i]]
                line.appendcbb(key, curve.attr(key), vals)
            out.append(line)
        return out
