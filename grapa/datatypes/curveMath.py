# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 15:10:29 2017

@author: Romain Carron
Copyright (c) 2023, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

import numpy as np

from grapa.curve import Curve
from grapa.graph import Graph
from grapa.mathModule import is_number
from grapa.gui.GUIFuncGUI import FuncGUI


class CurveMath(Curve):
    """
    Class handling optical spectra, with notably nm to eV conversion and
    background substraction.
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
        listCurves = [""]
        listCurves_samelen = []
        default_samelen = ""
        graph = kwargs["graph"] if "graph" in kwargs else None
        if graph is not None:
            for i in range(len(graph)):
                lbl_nice = str(i) + " " + str(graph[i].attr("label"))[0:12]
                listCurves.append(lbl_nice)
                if len(graph[i].x()) == le:
                    listCurves_samelen.append(lbl_nice)
                if self == graph[i]:
                    default_samelen = str(i)

        fieldprops = {
            "field": "Combobox",
            "width": 12,
            "values": listCurves,
            "bind": "beforespace",
        }
        default = ""
        fieldprops_samelen = {
            "field": "Combobox",
            "width": 12,
            "values": listCurves_samelen,
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
        out.append([self.swapXY, "x <-> y", [], []])

        line = FuncGUI(self.assembleCurveXY, "Assemble new Curve", {"graph": graph})
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
        out.append([self.printHelp, "Help!", [], []])
        return out

    @classmethod
    def classNameGUI(cls):
        return cls.CURVE.replace("Curve ", "") + " operations"

    def add(self, cst, curves, graph=None, operator="add"):
        def op(x, y, operator):
            if operator == "sub":
                return x - y
            if operator == "mul":
                return x * y
            if operator == "div":
                return x / y
            if operator == "pow":
                return x ** y
            if operator != "add":
                print(
                    "WARNING CureMath.add: unexpected operator argument",
                    "(" + operator + ").",
                )
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
        for c in range(len(graph)):
            if graph[c] == self:
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
                    out = op(out, graph.curve(int(c)), operator)
                    lbl = graph.curve(int(c)).attr("label")
                    lst.append("{Curve " + str(int(c)))
                    if lbl != "":
                        lst[-1] = lst[-1] + ": " + str(lbl)
                    lst[-1] = lst[-1] + "}"
        txt = strjoin.join(lst)
        math = self.attr("math")
        out.update({"math": math + "; " + txt if math != "" else txt})
        return out

    def neg(self):
        out = 0 - self
        lbl = self.attr("label")
        txt = " - {Curve" + (": " + lbl if lbl != "" else "") + "}"
        math = self.attr("math")
        out.update({"math": math + "; " + txt if math != "" else txt})
        return out

    def inv(self):
        out = 1 / self
        lbl = self.attr("label")
        txt = "1 / {Curve" + (": " + lbl if lbl != "" else "") + "}"
        math = self.attr("math")
        out.update({"math": math + "; " + txt if math != "" else txt})
        return out

    def swapXY(self):
        out = 0 + self  # work on copy
        out.setX(self.y())
        out.setY(self.x())
        txt = "swap x<->y"
        math = self.attr("math")
        out.update({"math": math + "; " + txt if math != "" else txt})
        return out

    def assembleCurveXY(
            self, idx_x: int, xory_x: str, idx_y: int, xory_y: str, graph: Graph
    ) -> Curve:
        """
        Assemble new Curve from data series of same length available in a Graph object.
        :param idx_x: index of Curve in Graph, from which to pick the x data series
        :param xory_x: is the x or y data series picked
        :param idx_y:index of Curve in Graph, from which to pick the y data series
        :param xory_y: is the x or y data series picked
        :param graph: Graph object to work on
        :return: a Curve with x and y data according to wishes
        """
        curve_x = graph[int(idx_x)]
        curve_y = graph[int(idx_y)]
        data_x = curve_x.x() if xory_x == "x" else curve_x.y()
        data_y = curve_y.x() if xory_y == "x" else curve_y.y()
        if len(data_x) != len(data_y):
            print("ERROR CurveMath assembleCurveXY: selected data series do not have"
                  + "same length! Abort.")
            print("input parameters:", idx_x, xory_x, idx_y, xory_y)
            return False
        attr = {}
        attr.update(curve_x.getAttributes())
        attr.update(curve_y.getAttributes())
        curve = Curve([data_x, data_y], attr)
        label = "{} {} vs {} {}".format(curve_y.attr("label"), xory_y,
                                        curve_x.attr("label"), xory_x)
        curve.update({"label": label})
        math = curve.attr("math")
        txt = "assembled x,y from curves {} {} {}, {} {} {}".format(
            int(idx_x), curve_x.attr("label"), xory_x,
            int(idx_y), curve_y.attr("label"), xory_y,
        )
        curve.update({"math": math + "; " + txt if math != "" else txt})
        curve_type = curve.attr("curve", "")
        if curve_type != "":
            curve = curve.castCurve(curve_type)
        return curve

    def printHelp(self):
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
        print(" - Assemble a new Curve from data series of same length available in"
              + "the current Graph.")
        return True
