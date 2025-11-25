# -*- coding: utf-8 -*-
"""
Class handling optical spectra, with notably nm to eV conversion and background
subtraction.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import logging

import numpy as np

from grapa.curve import Curve
from grapa.graph import Graph
from grapa.mathModule import is_number
from grapa.utils.funcgui import FuncGUI
from grapa.utils.error_management import issue_warning

logger = logging.getLogger(__name__)


class CurveMath(Curve):
    """
    Class handling optical spectra, with notably nm to eV conversion and background
    subtraction.
    """

    CURVE = "Curve Math"

    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        self.update({"Curve": CurveMath.CURVE})

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        le = len(self.x())
        list_curves = [""]
        list_curves_samelen = []
        default_samelen = ""
        graph = kwargs["graph"] if "graph" in kwargs else None
        if graph is not None:
            for c, curve in enumerate(graph):
                lbl_nice = str(c) + " " + str(curve.attr("label"))[0:12]
                list_curves.append(lbl_nice)
                if len(curve.x()) == le:
                    list_curves_samelen.append(lbl_nice)
                if self == curve:
                    default_samelen = str(c)

        fieldprops = {
            "field": "Combobox",
            "width": 12,
            "values": list_curves,
            "bind": "beforespace",
        }
        default = ""
        fieldprops_samelen = {
            "field": "Combobox",
            "width": 12,
            "values": list_curves_samelen,
            "bind": "beforespace",
        }
        xory = {"field": "Combobox", "width": 2, "values": ["x", "y"]}
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
        out.append(
            [
                self.add,
                "Add",
                ["{this Curve} + cst", "+ Curve_idx"],
                [0, default],
                {"graph": graph, "operator": "add"},
                [{}, fieldprops],
            ]
        )
        out.append(
            [
                self.add,
                "Sub",
                ["{this Curve} - cst", "- Curve_idx"],
                [0, default],
                {"graph": graph, "operator": "sub"},
                [{}, fieldprops],
            ]
        )
        out.append(
            [
                self.add,
                "Mul",
                ["{this Curve} * cst", "* Curve_idx"],
                [1, default],
                {"graph": graph, "operator": "mul"},
                [{}, fieldprops],
            ]
        )
        out.append(
            [
                self.add,
                "Div",
                ["{this Curve} / cst", "/ Curve_idx"],
                [1, default],
                {"graph": graph, "operator": "div"},
                [{}, fieldprops],
            ]
        )
        out.append([self.neg, "0 - {this Curve} ", [], []])
        out.append([self.inv, "1 / {this Curve} ", [], []])
        out.append([self.swap_xy, "x <-> y", [], []])

        line = FuncGUI(self.assemble_curve_xy, "Assemble new Curve", {"graph": graph})
        line.append("x:", default_samelen, options=fieldprops_samelen)
        line.append(".", "x", options=xory)
        line.append(", y:", default_samelen, options=fieldprops_samelen)
        line.append(".", "y", options=xory)
        out.append(line)
        # out.append([self.assembleCurveXY, "Assemble new Curve",
        #             ["x:", ".", ", y:", "."],
        #             [default_samelen, "x", default_samelen, "y"],
        #             {"graph": graph},
        #             [fieldprops_samelen, xory, fieldprops_samelen, xory]])
        out.append([self.print_help, "Help!", [], []])
        return out

    @classmethod
    def classNameGUI(cls):
        return cls.CURVE.replace("Curve ", "") + " operations"

    @classmethod
    def _append_to_math(cls, curve, txt):
        """Append a text to the 'math' attribute of a curve."""
        math = curve.attr("math")
        curve.update({"math": math + "; " + txt if math != "" else txt})

    def add(self, cst, curves, graph=None, operator="add"):
        """Add a constant or a (list of) curves to this curve."""

        def op(x, y, operator):
            if operator == "sub":
                return x - y
            if operator == "mul":
                return x * y
            if operator == "div":
                return x / y
            if operator == "pow":
                return x**y
            if operator != "add":
                msg = "CurveMath add: unexpected operator argument ({})."
                issue_warning(logger, msg.format(operator))
            return x + y

        strjoin_ = {
            "add": " + ",
            "sub": " - ",
            "mul": " * ",
            "div": " / ",
            "pow": " ** ",
        }
        strjoin = strjoin_[operator] if operator in strjoin_ else " + "
        # fabricate a copy of the curve
        out = self + 0
        lbl = self.attr("label")
        idx = np.nan
        for c, curve in enumerate(graph):
            if curve == self:
                idx = c
                break
        lst = ["{Curve " + str(int(idx)) + (": " + lbl if lbl != "" else "") + "}"]
        # constants
        if not isinstance(cst, (list, tuple)):
            cst = [cst]
        for c in cst:
            out = op(out, c, operator)
        lst += [str(c) for c in cst]
        # curves
        if graph is not None:
            if not isinstance(curves, (list, tuple)):
                curves = [curves]
            for c in curves:
                if is_number(c):
                    out = op(out, graph[int(c)], operator)
                    lbl = graph[int(c)].attr("label")
                    lst.append("{Curve " + str(int(c)))
                    if lbl != "":
                        lst[-1] = lst[-1] + ": " + str(lbl)
                    lst[-1] = lst[-1] + "}"
        txt = strjoin.join(lst)
        math = self.attr("math")
        out.update({"math": math + "; " + txt if math != "" else txt})
        return out

    def neg(self):
        """Return the negative of this curve, i.e. 0 - Curve."""
        out = 0 - self
        lbl = self.attr("label")
        txt = " - {Curve" + (": " + lbl if lbl != "" else "") + "}"
        self._append_to_math(out, txt)
        return out

    def inv(self):
        """Return the inverse of this curve, i.e. 1 / Curve."""
        out = 1 / self
        lbl = self.attr("label")
        txt = "1 / {Curve" + (": " + lbl if lbl != "" else "") + "}"
        self._append_to_math(out, txt)
        return out

    def swap_xy(self):
        """Return a copy of this curve with x and y data swapped."""
        out = 0 + self  # work on copy
        out.setX(self.y())
        out.setY(self.x())
        txt = "swap x<->y"
        self._append_to_math(out, txt)
        return out

    def assemble_curve_xy(
        self, idx_x: int, xory_x: str, idx_y: int, xory_y: str, graph: Graph
    ) -> Curve:
        """
        Assemble new Curve from data series of same length available in a Graph object.

        :param idx_x: index of Curve in Graph, from which to pick the x data series
        :param xory_x: is the x or y data series picked
        :param idx_y: index of Curve in Graph, from which to pick the y data series
        :param xory_y: is the x or y data series picked
        :param graph: Graph object to work on
        :return: a Curve with x and y data according to wishes
        """
        curve_x = graph[int(idx_x)]
        curve_y = graph[int(idx_y)]
        data_x = curve_x.x() if xory_x == "x" else curve_x.y()
        data_y = curve_y.x() if xory_y == "x" else curve_y.y()
        if len(data_x) != len(data_y):
            print(
                "ERROR CurveMath assembleCurveXY: selected data series do not have"
                + "same length! Abort."
            )
            print("input parameters:", idx_x, xory_x, idx_y, xory_y)
            return False
        attr = {}
        attr.update(curve_x.get_attributes())
        attr.update(curve_y.get_attributes())
        curve = Curve([data_x, data_y], attr)
        label = "{} {} vs {} {}".format(
            curve_y.attr("label"), xory_y, curve_x.attr("label"), xory_x
        )
        curve.update({"label": label})
        txt = "assembled x,y from curves {} {} {}, {} {} {}".format(
            int(idx_x),
            curve_x.attr("label"),
            xory_x,
            int(idx_y),
            curve_y.attr("label"),
            xory_y,
        )
        self._append_to_math(curve, txt)
        curve_type = curve.attr("curve", "")
        if curve_type != "":
            curve = curve.castCurve(curve_type)
        return curve

    def print_help(self):
        print("*** *** ***")
        print("CurveMath offers some mathematical transformation of the data.")
        print("Functions:")
        print(" - Add. Parameters:")
        print("   cst: a constant, or a list of constants,")
        print("   curves: the index of a Curve, or a list of Curves indices.")
        print(" - Sub. Substraction, similar as Add.")
        print(" - Mul. Multiplication, similar as Add.")
        print(" - Div. Division as Add.")
        print(" - Neg. 0 - Curve.")
        print(" - Inv. 1 / Curve.")
        print(
            " - Assemble a new Curve from data series of same length available in"
            + "the current Graph."
        )
        return True
