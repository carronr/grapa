﻿# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:46:13 2016

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np


from grapa.graph import Graph
from grapa.curve import Curve
from grapa.mathModule import is_number, roundSignificant, roundSignificantRange
from grapa.constants import CST
from grapa.utils.graphIO import GraphIO
from grapa.utils.funcgui import FuncListGUIHelper


class GraphJscVoc(Graph):
    FILEIO_GRAPHTYPE = "Jsc-Voc curve"

    AXISLABELS = [["Voc", "", "V"], ["Jsc", "", "mA cm$^{-2}$"]]

    @classmethod
    def isFileReadable(cls, filename, fileext, line1="", line2="", line3="", **_kwargs):
        if (
            fileext == ".txt"
            and filename[:7] == "JscVoc_"
            and line1[:14] == "Sample name: 	"
        ):
            return True  # can open this stuff
        return False

    def readDataFromFile(self, attributes, **kwargs):
        le = len(self)
        GraphIO.readDataFromFileGeneric(self, attributes, **kwargs)
        # expect 3 columns data
        self.castCurve(CurveJscVoc.CURVE, le, silentSuccess=True)
        # remove strange characters from attributes keys
        attr = self[le].get_attributes()
        dictUpd = {}
        for key in attr:
            if "²" in key or "Â²" in key:
                dictUpd.update(
                    {
                        key.replace("²", "2")
                        .replace("²", "2")
                        .replace("Â²", "2"): attr[key]
                    }
                )
                dictUpd.update({key: ""})
        self[le].update(dictUpd)
        # remove strange characters from attributes values
        for key in self[le].get_attributes():
            val = self[le].attr(key)
            if isinstance(val, str) and r"ÃÂ²" in val:
                self[le].update({key: val.replace(r"ÃÂ²", "2")})
        # cosmetics
        self[le].update(
            {"type": "scatter", "cmap": [[0, 0, 1], [1, 0.43, 0]], "markeredgewidth": 0}
        )
        lbl = (
            self.attr("label")
            .replace("JscVoc ", "")
            .replace("²", "2")
            .replace(" Values Jsc [mA/cm2]", "")
        )
        self[le].update({"label": lbl, "sample": lbl})
        self[le].data_units(unit_x="V", unit_y="mA cm-2")
        self[le + 1].update({"type": "scatter_c"})
        self[le + 1].data_units(unit_x="V", unit_y="K")
        # self.curve(le+1).visible(False) # hide T curve
        self.update(
            {
                "typeplot": "semilogy",
                "alter": ["", "abs"],
                "xlabel": self.formatAxisLabel(GraphJscVoc.AXISLABELS[0]),
                "ylabel": self.formatAxisLabel(GraphJscVoc.AXISLABELS[1]),
            }
        )

    def findCurveWithX(self, curve):
        """
        Find the Curve with same x data as the given Curve
        JscVoc: that Curve should store temperatures
        """
        for cu in range(len(self)):
            if self[cu] == curve:
                # start looping at position of the curve
                for c in range(cu + 1, len(self)):
                    if np.array_equiv(curve.x(), self[c].x()):
                        return self[c]
                # could not find, start looking from beginning
                for c in range(cu):
                    if np.array_equiv(curve.x(), self[c].x()):
                        return self[c]
                return False

    def split_temperatures(self, curve, threshold=3):
        """
        Splits the compiled data into different data (one for each T)
        curve: stores the Jsc-Voc pairs - will need to find the T
        threshold in °C
        """
        T = GraphJscVoc.findCurveWithX(self, curve)
        if T == False:
            print(
                "Error JscVoc: cannot find temperature Curve, must have same",
                "Voc list as Jsc-Voc Curve. Aborted.",
            )
            return [], []
        Voc = curve.x()
        Jsc = curve.y()
        Tem = T.y()
        datas = []
        temps = []
        j = 0
        data, temp = [], []
        for i in range(len(Voc)):
            # check if is new temperature
            if j != 0:
                if np.abs(Tem[i] - np.average(temp)) > threshold:
                    datas.append(data)
                    temps.append(np.average(temp))
                    data = []
                    temp = []
                    j = 0
            if j == 0:  # is new temperature
                data = [[Voc[i]], [Jsc[i]]]
                temp = [Tem[i]]
            else:  # is not new
                data[0].append(Voc[i])
                data[1].append(Jsc[i])
                temp.append(Tem[i])
            j += 1
        datas.append(data)
        temps.append(np.average(temp))
        return datas, temps

    def CurvesJscVocSplitTemperature(self, threshold=3, curve=None):
        """Splits the compiled data into different data (one for each T)

        :param threshold: in K
        :param curve: stores the Jsc-Voc pairs. If None: error. Weird prototype design
               to allow calls from GUI
        """
        if curve is None:
            print(
                "Error CurvesJscVocSplitTemperature, you must provide",
                'argument key "curve" with a Jsc-Voc Curve.',
            )
        datas, temps = GraphJscVoc.split_temperatures(self, curve, threshold=3)
        attr = curve.get_attributes()
        out = []
        for i in range(len(datas)):
            out.append(CurveJscVoc(datas[i], attr))
            out[-1].update(
                {
                    "temperature": temps[i],
                    "type": "",
                    "cmap": "",
                    "label": "{} {:.0f} K".format(out[-1].attr("label"), temps[i]),
                }
            )
            out[-1].data_units(unit_x="V", unit_y="mA cm-2")
        return out

    # handling of curve fitting
    def CurveJscVoc_fitNJ0(
        self,
        Voclim=None,
        Jsclim=None,
        threshold=3,
        graphsnJ0=True,
        curve=None,
        silent=False,
    ):
        """Fit the Jsc-Voc data, returns fitted Curves.
        If required, first splits data in different temperatures.

        :param Voclim: fit limits for Voc, in data units
        :param Jsclim: fit limits for Jsc, in data units
        :param threshold: tolerance to temperature, in K
        :param graphsnJ0: if True, also returns A vs T, J0 vs T and J0 vs A*T
        :param curve: stores the Jsc-Voc pairs. If None: error. Weird prototype design
               to allow calls from GUI
        :param silent: If False, prints more information
        """
        if curve is None:
            print(
                "Error CurveJscVoc_fitNJ0, you must provide argument key",
                '"curve" with a Jsc-Voc Curve.',
            )
            return False
        try:
            T = curve.T(default=np.nan, silent=True)
        except Exception:
            print(
                "Warning CurveJscVoc_fitNJ0, method T() not found. Curve",
                "type not suitable?",
            )
            T = np.nan
        # check if is single or multiple temperature
        if np.isnan(T):
            datas, temps = GraphJscVoc.split_temperatures(
                self, curve, threshold=threshold
            )
        else:
            datas, temps = [[curve.x(), curve.y()]], [T]
        # fit different data series (different illuminations at same T)
        out = []
        ns, J0s = [], []
        for i in range(len(datas)):
            n, J0 = curve.fit_nJ0(
                Voclim=Voclim, Jsclim=Jsclim, data=datas[i], T=temps[i]
            )
            attr = {
                "color": "k",
                "area": curve.getArea(),
                "_popt": [n, J0],
                "_fitFunc": "func_nJ0",
                "_Voclim": Voclim,
                "_Jsclim": Jsclim,
                "temperature": temps[i],
                "filename": "fit to "
                + curve.attr("filename").split("/")[-1].split("\\")[-1],
            }
            if not silent:
                msg = "Fit Jsc-Voc (T={}): ideality factor n={}, J0={:1.4e} [mA/cm2]."
                print(msg.format(temps[i], n, J0))
            x, y = curve.selectData(xlim=Voclim, ylim=Jsclim, data=datas[i])
            out.append(CurveJscVoc([x, curve.func_nJ0(x, n, J0, T=temps[i])], attr))
            ns.append(n)
            J0s.append(J0)
        if graphsnJ0:
            from grapa.datatypes.curveArrhenius import (
                CurveArrhenius,
                CurveArrheniusJscVocJ00,
            )

            # generate n vs T, J0 vs T
            ylabel_j0 = ["Saturation current J$_0$", "", "mA cm$^{-2}$"]
            lbl = curve.attr("label")
            if len(lbl) > 0:
                lbl += " "
            out.append(
                Curve(
                    [temps, ns],
                    {
                        "linestyle": "none",
                        "linespec": "o",
                        "label": lbl + "Ideality factor A vs T [K]",
                        Curve.KEY_AXISLABEL_X: ["Temperature", "T", "K"],
                        Curve.KEY_AXISLABEL_Y: ["Ideality factor", "A", ""],
                    },
                )
            )
            out.append(
                Curve(
                    [temps, J0s],
                    {
                        "linestyle": "none",
                        "linespec": "o",
                        "label": lbl + "J0 vs T [K]",
                        Curve.KEY_AXISLABEL_X: ["Temperature", "T", "K"],
                        Curve.KEY_AXISLABEL_Y: ylabel_j0,
                    },
                )
            )
            out.append(
                CurveArrhenius(
                    [np.array(temps) * np.array(ns), J0s], CurveArrheniusJscVocJ00.attr
                )
            )
            out[-1].update(
                {
                    "label": lbl + out[-1].attr("label"),
                    Curve.KEY_AXISLABEL_X: ["A * temperature", "", "K"],
                    Curve.KEY_AXISLABEL_Y: ylabel_j0,
                }
            )  # 'label': lbl+'J0 vs A*T'
            # out[-1].update({'type': 'scatter', 'markeredgewidth':0, 'markersize':100, 'cmap': 'inferno'})# 'label': lbl+'J0 vs A*T'
            # out.append(Curve([np.array(temps)*np.array(ns), np.array(temps)], {'linestyle': 'none', 'type': 'scatter_c'}))
        return out

    def split_illumination(
        self,
        threshold=3,
        ifFit=False,
        fitTlim=None,
        extend0=False,
        curve=None,
        silent=False,
    ):
        """
        Splits Jsc Voc data according to illumination intensity.
        Assumes data are stored as intensity series, grouped by temperatures
        """
        from grapa.datatypes.curveArrhenius import CurveArrhenius

        # retrieves data
        if curve is None:
            print(
                "Error splitIllumination, you must provide argument key",
                '"curve" with a Jsc-Voc Curve.',
            )
            return False
        # check if is single or multiple temperature
        curveT = GraphJscVoc.findCurveWithX(self, curve)
        if not isinstance(curveT, Curve):
            print(
                "Error splitIllumination, cannot find Curve with T",
                "information (must have same x data as the selected Jsc vs",
                "Voc Curve).",
            )
            return False
        Ts = curveT.y()
        x = curve.x()
        T = np.inf
        j = 0
        data = []
        for i in range(len(x)):
            if np.abs(T - Ts[i]) > threshold:
                j = 0
            if i == j:
                data.append([[], []])
            data[j][0].append(Ts[i])
            data[j][1].append(x[i])
            T = Ts[i]
            j += 1
        attr = {
            "area": curve.getArea(),
            "filename": "extracted from "
            + curve.attr("filename").split("/")[-1].split("\\")[-1],
        }
        try:
            from grapa.colorscale import Colorscale

            colorscale = Colorscale(
                np.array([[0.91, 0.25, 1], [1.09, 0.75, 1]]), space="hls"
            )
            colors = colorscale.values_to_color(np.array(range(len(data))) / len(data))
        except Exception:
            colors = [""] * len(data)
        xylbls = [
            self.formatAxisLabel(["Temperature", "T", "K"]),
            self.formatAxisLabel(["Voc", "", "V"]),
        ]
        res = []
        out = []
        for j in range(len(data) - 1, -1, -1):
            out.append(CurveArrhenius(data[j], attr))
            out[-1].update(
                {
                    "label": "Voc vs T, intensity " + str(j),
                    "labelhide": 1,
                    "linespec": "o",
                    "color": colors[j],
                    "_Arrhenius_variant": "ExtrapolationTo0",
                    "_Arrhenius_dataLabel": xylbls,
                    "_Arrhenius_dataLabelArrhenius": xylbls,
                    Curve.KEY_AXISLABEL_X: ["Temperature", "T", "K"],
                    Curve.KEY_AXISLABEL_Y: ["Voc", "", "V"],
                }
            )
            if ifFit:
                Tlim = fitTlim
                if fitTlim is None:
                    Tlim = [
                        0.9 * min(out[-1].x_1000overK()) * 0.9,
                        1.1 * max(out[-1].x_1000overK()),
                    ]
                out.append(out[-1].CurveArrhenius_fit(Tlim, silent=True))
                res.append([j, out[-1].attr("_popt")])
            if extend0:
                out[-1].resampleX("linspace", 0, np.max(out[-1].x_1000overK()), 51)
        if not silent:
            if len(res) > 0:
                print("Intensity level, Voc @ T=0 [V]")
                for r in res:
                    print(str(r[0]), r[1][1])
        if len(out) > 0:
            out[0].update({"labelhide": ""})
            out[-2].update({"labelhide": ""})
        return out


class CurveJscVoc(Curve):
    CURVE = "Curve JscVoc"

    AXISLABELS_X = {"": GraphJscVoc.AXISLABELS[0]}
    AXISLABELS_Y = {
        "": GraphJscVoc.AXISLABELS[1],
        Curve.ALTER_ABS: GraphJscVoc.AXISLABELS[1],
    }

    CST_Jsclim0 = 1.0  # mA/cm2

    Tdefault = CST.STC_T  # 273.15 + 25

    def __init__(self, data, attributes, silent=False):
        # delete area from attributes, to avoid normalization during init
        Curve.__init__(self, data, attributes, silent=silent)
        self.update({"Curve": CurveJscVoc.CURVE})  # for saved files further re-reading

    # RELATED TO GUI
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        graph = kwargs["graph"] if "graph" in kwargs else None
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...] (, [default1, default2, ...]) ]
        lbl = "Area [cm2] (old value " + "{:4.3f}".format(self.getArea()) + ")"
        out.append(
            [self.setArea, "Area correction", [lbl], ["{:6.5f}".format(self.getArea())]]
        )  # one line per function
        # fit function
        if self.attr("_fitFunc") == "" or self.attr("_popt", None) is None:
            Voclim = roundSignificantRange([0, max(self.x()) * 1.01], 3)
            Jsclim = roundSignificantRange([0.1, max(self.y()) * 1.1], 2)
            out.append(
                [
                    GraphJscVoc.CurveJscVoc_fitNJ0,
                    "Fit J0 & ideality",
                    ["Voclim", "Jsclim", "T max fluct.", "compile"],
                    [Voclim, Jsclim, 3, True],
                    {"curve": self},
                ]
            )
        else:  # if fitted Curve
            out.append(
                [
                    self.updateFitParam,
                    "Modify fit",
                    ["n", "J0"],
                    roundSignificant(self.attr("_popt"), 5),
                ]
            )
        # split according to temperatures
        out.append(
            [
                GraphJscVoc.CurvesJscVocSplitTemperature,
                "Separate according to T",
                ["T max fluctuations"],
                [3],
                {"curve": self},
            ]
        )
        # split according to illumination intensities (Voc vs T)
        Tlim = [0, 350]
        if graph is not None:
            for c in range(len(graph) - 1):
                if graph[c] == self:
                    if np.array_equiv(graph[c + 1].x(), graph[c].x()):
                        Tlim = [
                            0.99 * np.min(graph.curve(c + 1).y()),
                            1.01 * np.max(graph.curve(c + 1).y()),
                        ]
                        break
        Tlim = roundSignificantRange(Tlim, 3)
        out.append(
            [
                GraphJscVoc.split_illumination,
                "Separate Voc vs T",
                ["T max fluct.", "fit Voc(T)", "T limits", "extend to 0"],
                [3, True, Tlim, True],
                {"curve": self},
                [{}, {"field": "Checkbutton"}, {}, {"field": "Checkbutton"}],
            ]
        )
        out.append([self.print_help, "Help!", [], []])

        out += FuncListGUIHelper.graph_axislabels(self, **kwargs)
        return out

    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        out.append(["Log10 abs", ["", "abs"], "semilogy"])
        return out

    # Handling of cell area
    def getArea(self):
        """return area of the cell as stored in the Curve parameters"""
        area = self.attr("area [cm2]")
        if is_number(area):
            return area
        return 1

    def setArea(self, new):
        """correct the cell area, and scale the y (list of Jsc) accordingly"""
        old = self.getArea()
        self.setY(self.y() * old / new)
        self.update({"area [cm2]": new})
        return True

    def fit_nJ0(self, Voclim=None, Jsclim=None, data=None, T=None):
        """perform fitting, returns best fit parameters"""
        datax, datay = self.selectData(xlim=Voclim, ylim=Jsclim, data=data)
        if len(datax) < 2 or len(datay) < 2:
            return [np.nan, np.nan]
        if T is None:
            T = self.T()
        # actual fitting
        datay = np.log(datay)
        z = np.polyfit(datax, datay, 1, full=True)[0]
        n = CST.q / CST.kb / T / z[0]
        J0 = np.exp(z[1])
        return [n, J0]

    def func_nJ0(self, Voc, n, J0, T=None):
        """fit function"""
        if T is None:
            T = self.T()
        out = J0 * np.exp(CST.q * Voc / (n * CST.kb * T))
        return out

    def T(self, default=None, silent=False):
        """Returns the acquisition temperature, otherwise default."""
        test = self.attr("temperature", 0)
        if test != 0:
            return test
        if not silent:
            print(
                "Curve JscVoc cannot find keyword temperature.", self.get_attributes()
            )
        if default is None:
            return CurveJscVoc.Tdefault
        return default

    def splitIllumination(self, curve, threshold=3):
        """
        Splits the compiled data into different data (one for each intensity)
        curve: stores the Jsc-Voc pairs - will need to find the T
        threshold in °C
        ifFit: bool. if True, also fits the output Curves
        fitTlim: T range to for the fit Voc(T)
        """
        T = GraphJscVoc.findCurveWithX(self, curve)
        if T == False:
            print(
                "Error JscVoc: cannot find temperature Curve, must have",
                "same Voc list as Jsc-Voc Curve. Aborted.",
            )
            return [], []
        Voc = curve.x()
        Jsc = curve.y()
        Tem = T.y()
        datas = []
        temps = []
        j = 0
        data, temp = [], []
        for i in range(len(Voc)):
            # check if is new temperature
            if j != 0:
                if np.abs(Tem[i] - np.average(temp)) > threshold:
                    datas.append(data)
                    temps.append(np.average(temp))
                    data = []
                    temp = []
                    j = 0
            if j == 0:  # is new temperature
                data = [[Voc[i]], [Jsc[i]]]
                temp = [Tem[i]]
            else:  # is not new
                data[0].append(Voc[i])
                data[1].append(Jsc[i])
                temp.append(Tem[i])
            j += 1
        datas.append(data)
        temps.append(np.average(temp))
        return datas, temps

    def CurvesJscVocSplitIllumination(self, curve=None):
        """
        Splits the compiled data into different curves (one for each intensity)

        :param curve: stores the Jsc-Voc pairs. If None: error. Weird prototype design
               to allow calls from GUI
        """
        if curve is None:
            print(
                'Error CurvesJscVocSplitIllumination, you must provide argument key "curve" with a Jsc-Voc Curve.'
            )
        datas, temps = GraphJscVoc.split_temperatures(self, curve, threshold=3)
        attr = curve.get_attributes()
        out = []
        for i in range(len(datas)):
            out.append(CurveJscVoc(datas[i], attr))
            out[-1].update({"temperature": temps[i], "type": "", "cmap": ""})
            out[-1].update(
                {"label": out[-1].attr("label") + " " + "{:.0f}".format(temps[i]) + "K"}
            )
        return out

    def print_help(self):
        print("*** *** ***")
        print("CurveJV offer basic treatment of Jsc-Voc pairs of solar cells.")
        print("Based on Thomas Weiss script, and on the following references:")
        print("  Schock, Scheer, p111, footnote 20")
        print("  Hages et al., JAP 115, 234504 (2014)")
        print("Curve transforms:")
        print(
            "- Linear: standard is current density [mA cm-2] versus [V], at different light intensitites."
        )
        print(
            "- Log 10 abs: logarithm of J vs V. Same display as JV curve, to visualize the diode behavior."
        )
        print("Analysis functions:")
        print("- Area correction: scale Jsc data according to cell area.")
        print(
            "- Fit J0 & ideality: fit Jsc vs Voc data and extract J0 and ideality factor A. Parameters:"
        )
        print("  Voclim, Jsclim: fit limits for Voc and Jsc")
        print(
            "  T max fluct.: identify groups of temperatures to fit only relevant data together."
        )
        print(
            '      A new temperature is identified if a point deviates more than "value" from the average.'
        )
        print(
            "  compile: after fitting data, returns Curves with results: ideality factor versus T,"
        )
        print("     J0 versus T, and J0 vs A * T.")
        print(
            "- Separate according to T: split data in several Curves according to the temperature identified."
        )
        print(
            "  T max fluct.: identify groups of temperatures to fit only relevant data together."
        )
        print(
            "      A new Curve is created once a point deviates more than this value from the average."
        )
        print(
            "- Extract Voc vs T: split data in several Curves, grouping Voc vs T data acquired with same illumination intensity."
        )
        print(
            "  T max fluct.: identify groups of temperatures to group relevant intensities together."
        )
        return True
