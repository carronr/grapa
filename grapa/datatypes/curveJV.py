﻿# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:46:13 2016

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit
from copy import deepcopy
from re import findall as refindall, sub as resub
import os


from grapa.graph import Graph
from grapa.curve import Curve
from grapa.constants import CST
from grapa.mathModule import xAtValue, is_number, roundSignificant, derivative, smooth
from grapa.utils.funcgui import FuncListGUIHelper, FuncGUI


class CurveJV(Curve):
    """
    CurveJV offer basic treatment of J-V curves of solar cells.
    Input units are [V] and [mA] (or [mA cm-2]).
    """

    CURVE = "Curve JV"

    AXISLABELS_X = {
        "": ["Bias voltage", "V", "V"],
        "CurveJV.xinversejminusjsc": ["1/(J-Jsc) (Sites)", "", "cm$^2$ A$^{-1}$"],
    }
    AXISLABELS_Y = {
        "": ["Current density", "J", "mA cm$^{-2}$"],
        "CurveJV.yDifferentialR": ["Differential resistance dV/dJ", "", "Ohm cm$^2$"],
    }

    defaultIllumPower = 1000  # W/m2

    DARK = "dark"
    ILLUM = "illuminated"
    DARK_ILLUM = {True: DARK, False: ILLUM}
    DARK_ILLUM_VALUESLIST = list(DARK_ILLUM.values())

    # to retrieve info from filename: sample name, cell, measurement id
    # FINDALLSTR = "^(I-V_)*(.*)_([a-zA-Z]+[0-9]+)_([0-9]+)([_a-zA-Z0-9]*).txt"
    FILENAMEPARSE_EXPR = [
        "^I-V_(.*)_([a-zA-Z#]?[0-9]+)_(fwd|bwd)?[_]*([-.a-zA-Z0-9]*)_([0-9]+).txt",
        "^I-V_(.*)_([a-zA-Z]?[0-9]+)_([0-9]+)([_a-zA-Z0-9]*).txt",
    ]
    FILENAMEPARSE_KEYS = [
        ["sample", "cell", "fwd_bwd", "illumspectrum", "measId"],
        ["sample", "cell", "measId", "additional"],
    ]

    FORMAT_AUTOLABEL = [
        "${sample} ${cell}",
        "${sample}",
        "${sample} ${cell} ${temperature:.0f} K",
        "${temperature:.0f} K",
    ]

    def __init__(
        self,
        dataJV,
        attributes,
        units=["V", "mAcm-2"],
        illumPower=defaultIllumPower,
        ifCalc=True,
        silent=False,
    ):
        # delete area from attributes, to avoid normalization during
        # initialization

        # area
        tempArea = {}
        if "area" in attributes:
            tempArea = {"area": attributes["area"]}
            del attributes["area"]
        Curve.__init__(self, dataJV, attributes, silent=silent)
        if "area" in tempArea:  # by default: do not normalize data
            Curve.update(self, tempArea)
        if self.area() == 0:  # set 1 if no input
            Curve.update(self, {"area": 1})  # to allow later changes
            if not silent:
                print("Cell area was set to 1.")
        elif self.attr("area", None) is None:
            Curve.update(self, {"area": self.area()})  # to allow later changes

        # internally all units are treated as V and mA/cm2
        if units[0] == "mV":
            self.setX(self.x() / 1000)
            print("class CurveJV: V axis rescaling into V.")
        if units[1] == "Am-2":  # converts in mAcm-2
            self.setY(self.y() * 1000 / (100 * 100))
            print("class CurveJV: J axis rescaling into mA/cm2.")

        # illumPower default 1000 W/m2
        illumPower = illumPower if is_number(illumPower) else float(illumPower)
        self.update({"_units": units, "illumPower": illumPower})

        # temperature. self.T for convenience. See also update() override
        if is_number(self.attr("Temperature")):
            self.T = self.attr("Temperature")
        if self.attr("temperature") == "":
            self.update({"temperature": float(CST.STC_T)})

        if self.attr("_fitDiodeWeight") == "":
            self.update({"_fitDiodeWeight": 5})
        self.update({"Curve": self.CURVE})  # for saved files further re-reading
        # some warning for the user
        if np.max(self.x() > 10):
            print(
                "Warning CurveJV: max voltage > 10 detected. Voltage input",
                "expected in V, please correct if possible. Fitting is",
                "likely to fail on provided input.",
            )
        # sample, cell
        self.parse_filename(to_attr=True)
        if ifCalc:
            try:
                self.calcVocJscFFEffMPP()
                self.darkOrIllum(forceCalc=True)  # maybe updated as Jsc value available
            except Exception as e:
                print("WARNING CurveJV init: Exception", type(e))
                print(e)

    # RELATED TO GUI
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...] (, [default1, default2, ...]) ]
        # sample, cell
        keys = ["Sample", "Cell"]
        item = FuncGUI(self.updateValuesDictkeys, "Save", {"keys": keys})
        item.append("Sample", self.attr("sample"), options={"width": 20})
        item.append("Cell", self.attr("cell"), options={"width": 10})
        out.append(item)

        darkillum = str(self.darkOrIllum(ifText=True))
        opts_di = {"state": "readonly"}
        opts_fb = {"width": 5}
        keys = ["darkorillum", "illumspectrum", "fwd_bwd"]
        item = FuncGUI(self.updateValuesDictkeys, "Save", {"keys": keys})
        item.appendcbb(" ", darkillum, self.DARK_ILLUM_VALUESLIST, options=opts_di)
        item.append("spectrum", self.attr("illumspectrum"), options={"width": 10})
        item.appendcbb("sweep", self.attr("fwd_bwd"), ["fwd", "bwd"], options=opts_fb)
        out.append(item)

        # auto-label
        item = FuncGUI(self.label_auto, "Auto label")
        item.appendcbb("template", self.FORMAT_AUTOLABEL[0], self.FORMAT_AUTOLABEL)
        out.append(item)

        # keys = ["Sample", "Cell"]
        # out.append([self.updateFitParam, "Save", keys, [self.attr(k) for k in keys], {"keys": keys}])
        # area
        lbl = "Area [cm2] (current value " + "{:4.3f}".format(self.area()) + ")"
        out.append(
            [self.setArea, "Area correction", [lbl], ["{:6.5f}".format(self.area())]]
        )
        out.append([self.calcVocJscFFEffMPP_print, "Voc, Jsc, FF, Eff, MPP", [], []])
        # if data and not fitted Curve
        if self.attr("_fitFunc") == "" or self.attr("_popt", None) is None:
            out.append(
                [
                    self.CurveJVFromFit_print,
                    "JV fit",
                    [
                        "Fit range",
                        "Fit weight diode region\n1 neutral, 10 incr. weight",
                    ],
                    ["[-0.5, np.inf]", self.attr("_fitDiodeWeight")],
                ]
            )
            out.append([self.diodeFit_BestGuess, "JV curve initial guess", [], []])
        else:  # if fitted Curve
            out.append(
                [
                    self.updateFitParam,
                    "Modify fit",
                    ["n", "Jl-Jsc", "J0", "Rs", "Rp"],
                    roundSignificant(self.attr("_popt"), 5),
                ]
            )
            out.append(
                [
                    self.fit_resampleX,
                    "Resample V",
                    ["delta x, or [-0.3, 1, 0.01]"],
                    [roundSignificant((self.V(1) - self.V(0)) / 10, 5)],
                ]
            )
        if "graph" in kwargs:
            out.append(
                [
                    self.extractJscVoc,
                    "Extract Jsc-Voc values",
                    [],
                    [],
                    {"graph": kwargs["graph"]},
                ]
            )
        out.append([self.print_help, "Help!", [], []])
        out += FuncListGUIHelper.graph_axislabels(self, **kwargs)
        self._funclistgui_memorize(out)
        return out

    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        doc = "Standard is current density [mA cm-2] versus [V]"
        out.append(["Log10 abs", ["", "abs"], "semilogy", doc])
        doc = "log(J). Allows visualize Rp, Rs and diode."
        out.append(["Log10 abs (raw)", ["", "abs0"], "semilogy", doc])
        doc = "logarithm of (J-Jsc). Allows visualize Rp, Rs and diode."
        out.append(
            [
                "Differential R = dV/dJ [Ohm cm2]",
                ["", "CurveJV.yDifferentialR"],
                "semilogy",
                doc,
            ]
        )
        out.append(
            [
                "dV/dJ vs 1/(J-Jsc) - [Ohm cm2] vs [cm2/A] - Sites",
                ["CurveJV.xinversejminusjsc", "CurveJV.yDifferentialR"],
                "",
                "Sites' method.",
            ]
        )
        return out

    # overloaded Curve methods
    def update(self, attributes):
        """
        Override default update method, to handle the case where a new area
        or temperature would be set.
        """
        for key, value in attributes.items():
            # area: bypass default behavior
            if key.lower() == "area":
                self.setArea(value)
            # temperature: behavior in addition to default
            elif key.lower() == "temperature":
                if is_number(value):
                    self.T = value
                Curve.update(self, {key: value})
            # darkillum: force boolean, dict for human-friendly interaction
            elif key.lower() == "darkorillum" and value not in [""]:
                if value in self.DARK_ILLUM.values():
                    for k, val in self.DARK_ILLUM.items():
                        if value == val:
                            Curve.update(self, {key: k})
                elif isinstance(value, bool):
                    Curve.update(self, {key: value})
                else:
                    msg = "CurveJV update darkorillum, bad input ({}: {})."
                    print(msg.format(key, attributes[key]))
                    Curve.update(self, {key: value})
            else:
                Curve.update(self, {key: value})

    # OTHER
    def area(self):
        if self.attr("area") != "":
            return self.attr("area")
        if self.attr("Acquis soft Cell area") != "":
            return self.attr("Acquis soft Cell area")
        return 0

    def setArea(self, area, ifCalc=True):
        """Normalize the cell area, and modifies the data accordingly."""
        if not is_number(area):
            print("CurveJV setArea ERROR: parameter not numeric (value", area, ").")
            return
        if area == self.area():
            if not self.silent:
                print("CurveJV setArea: new area identical to previous value.")
            return
        if self.area() == "":  # for example when loading fileGeneric
            Curve.update(self, {"area": area})
            return
        if not self.silent:
            print("CurveJV setArea: new area (new", area, ", old", self.area(), ")")
        oldarea = self.area()
        self.setY(self.y() * oldarea / area)
        Curve.update(self, {"area": area})
        if not self.silent:
            print("Area set to", area, "(old value", oldarea, ")")
        # recalculate basic parameters
        if ifCalc:
            self.calcVocJscFFEffMPP()
            # erase information about fitting
            self.update({"diodefit": ""})
        return True

    def V(self, idx=np.nan):
        return self.x(idx)

    def J(self, idx=np.nan):
        return self.y(idx)

    def yDifferentialR(self, index=np.nan, xyValue=None):
        """Returns differential resistance of the J-V curve R = dV/dI"""
        if xyValue is not None:
            return np.array(xyValue)[1]
        V, J = self.x(), self.y() / 1000  # J in A/cm2
        val = derivative(J, V)  # R in Ohm cm2
        if np.isnan(index).any():
            return val[:]
        return val[index]

    def xinversejminusjsc(self, index=np.nan, xyValue=None):
        """
        To be used for Sites method, dV/dJ vs 1/(J-Jsc)
        Returns value in [A-1 cm2]
        """
        if xyValue is not None:
            return np.array(xyValue)[0]
        jsc = self.attr("jsc")
        if not is_number(jsc):
            jsc = 0
        j = self.y(index=index)
        # Jsc is positive value, but J(0V)<0
        # return value in [A-1 cm2]
        return 1 / (j + jsc) * 1000

    def interpJ(self, V):
        # returns values of a spline interpolation degree 3 at the V values
        f = interpolate.interp1d(self.V(), self.J())
        idx = (V > np.min(self.V())) * (V < np.max(self.V()))
        out = []
        for i in range(len(V)):
            out.append(f(V[i] if idx[i] else np.nan))
        return np.array(out)

    def simpleLabel(self, forceCalc=False):
        if self.attr("label") == "" or forceCalc:
            old = os.path.split(self.attr("label"))[1]
            old = old.split("\\")[-1] + " "
            old = resub(
                r"_[0-9][0-9](?P<next>[_ d$])", r"\g<1>", old
            )  # remove msmt number
            old = (
                old.replace("I-V_", "").replace("_", " ").replace("  ", " ").strip(" ")
            )
            self.update({"label": old})
        return self.attr("label")

    def parse_filename(self, to_attr=False):
        """Parse the filename to extract information contained in it.
        Parsing regexp: see CurveJV.FILENAMEPARSE_EXPR and CurveJV.FILENAMEPARSE_KEYS.

        :param to_attr: if True, update the Curve attributes: 'sample', 'cell',
               'direction', 'illumspectrum', 'measid', 'additional'.
        """
        out = {
            "sample": "",
            "cell": "",
            "direction": "",
            "illumspectrum": "",
            "measId": "",
            "additional": "",
        }
        filename = os.path.split(self.attr("filename"))[1]
        for i, expr in enumerate(self.FILENAMEPARSE_EXPR):
            keys = self.FILENAMEPARSE_KEYS[i]
            res = refindall(expr, filename)
            if len(res) < 1 or len(res[0]) != len(keys):
                continue
            for j, key in enumerate(keys):
                if key is not None:
                    out[key] = res[0][j]
            if to_attr:
                self.update(out)
            return out

        if to_attr:
            self.update(out)
        return out

    def sample(self, forceCalc=False):
        if self.attr("sample") == "" or forceCalc:
            data = self.parse_filename()
            self.update({"sample": data["sample"]})
        return self.attr("sample")

    def cell(self, forceCalc=False):
        if self.attr("cell") == "" or forceCalc:
            data = self.parse_filename()
            self.update({"cell": data["cell"]})
        return self.attr("cell").lower()

    def measId(self, forceCalc=False):
        if self.attr("measId") == "" or forceCalc:
            data = self.parse_filename()
            self.update({"measId": data["measId"]})
        return self.attr("measId")

    def fwd_bwd(self, forceCalc=False):
        if self.attr("fwd_bwd") == "" or forceCalc:
            data = self.parse_filename()
            self.update({"fwd_bwd": data["fwd_bwd"]})
        return self.attr("fwd_bwd")

    def illumspectrum(self, forceCalc=False):
        if self.attr("illumspectrum") == "" or forceCalc:
            data = self.parse_filename()
            self.update({"illumspectrum": data["illumspectrum"]})
        return self.attr("illumspectrum")

    def darkOrIllum(self, ifText=False, forceCalc=False, ifJscBelow=0.1):
        if self.attr("darkOrIllum", "") == "" or forceCalc:
            ifDark = False  # to be set later on
            sure = False
            # standard parsing of filename
            illumspectrum = self.parse_filename()["illumspectrum"]
            if illumspectrum != "":
                ifDark = True if illumspectrum.lower() == "dark" else False
                sure = True
            # alternative parsing of filename
            if not sure:
                name = self.attr("filename").split("/")[-1]
                split = refindall("([dD][aA][rR][kK])", name)
                if len(split) > 0:
                    ifDark = True
                    sure = True
            # alternative: based on Jsc
            if not sure:
                if is_number(ifJscBelow):
                    if self.attr("Jsc") != "":
                        if np.abs(self.attr("Jsc")) < ifJscBelow:
                            # 0.1 mA/cm2 is criterion for a dark JV curve
                            ifDark = True
                        else:
                            ifDark = False
                        sure = True
            # last resort
            if not sure:
                ifDark = False
            self.update({"darkOrIllum": ifDark})
        # read
        out = self.attr("darkOrIllum")
        if not ifText:
            return out
        if out in self.DARK_ILLUM.keys():
            return self.DARK_ILLUM[out]
        return out  # should not happen
        # else, out = self.DARK if self.attr('darkOrIllum') else self.ILLUM
        # return out

    def CurveJVFromFit(self, fitRange=None, diodeFitWeight=None, V=None, silent=False):
        """
        Returns a CurveJV object, a fit to the self object on the same Voltage points.

        :param fitRange: if None, auto-detect.
        :param diodeFitWeight: 0 for no change, otherwise divide the fit sigma in the
               diode behavior region. Values such as 10 significantly improve the
               fittin in this region.
        :param V: alternative V datapoints on which evaluate function after the fit.
               Default is full set of V datapoints.
        :param silent: if False, prints more information
        """
        if diodeFitWeight is not None:
            self.update({"_fitDiodeWeight": diodeFitWeight})
        p = self.diodeFit(ifPlot=False, silent=True, fitRange=fitRange)
        if p is False:
            print("Could not fit CurveJV.")
            return None
        Jsc = self.attr("Jsc")
        if V is None:
            V = self.V()
        J = self.func_diodeResistors_Jsc(V, p[0], p[1], p[2], p[3], p[4])
        out = CurveJV(
            [V, J],
            {
                "color": "k",
                "Jsc": Jsc,
                "diodefit": p,
                "Temperature": self.T,
                "area": self.area(),
                "_popt": p,
                "_fitfunc": "func_diodeResistors_Jsc",
                "_fitDiodeWeight": self.attr("_fitDiodeWeight"),
                "data filename": self.attr("filename"),
            },
            ifCalc=False,
        )
        if not silent:
            print(
                "CurveJV fit (n, Jl-Jsc, J0, Rs, Rp):",
                ", ".join([str(num) for num in roundSignificant(p, 5)]),
            )
            print("   units: n [ ], Jl-Jsc, J0 [mA cm-2], Rs, Rp [Ohm cm2].")
        return out

    def CurveJVFromFit_print(self, fitRange=None, fitDiodeWeight=None):
        """fits the J-V curve with the diode equation with 1 diode and 2 resistors model.
        The fit is performed using the log(J) values instead of J.
        The fit quality should be assessed visually on linear and logarithmic plots.

        - J0: good values are typically < 1e-4.

        - n: is ideality factor. Good values are < 1.7.

        - Jl: grapa implements the difference of the Jl to Jsc. Jl is not a fit
          parameter, but is adjusted so that J(V=0) = Jsc.

        :param fitRange: [V_min, V_max]
        :param fitDiodeWeight: increased fit weight for the datapoints in the diode
               region of the J-V curve. Can improve the accuracy of the fit near MPP.
        :return: a CurveJV fitted curve.
        """
        if fitDiodeWeight is not None:
            self.update({"_fitDiodeWeight": fitDiodeWeight})
        ans = self.CurveJVFromFit(fitRange=fitRange)
        out = self.printShort(header=True).replace("\n", "")
        try:
            out += self.printShort().replace
            print(out)
        except Exception:
            pass
        return ans

    # expected output file format
    # Cell	Voc_V	Jsc_mApcm2	FF_pc	Eff_pc	Area_cm2	Vmpp_V	Jmpp_mApcm2	Pmpp_mWpcm2	Rp_Ohmcm2	Rs_Ohmcm2	MeasTime	Temp_DegC	IllumCorr	CellCommt	GlobalCommt	FileName
    # d1	0.66	18.6	72.3	8.9	0.57	0.53	16.8	8.9	9710	1.1	11.07.2016 16:02:23	25	1			...\I-V_Oct1048ref_d1_01.txt
    def printShort(self, header=False):
        if header:
            out = "Cell\tVoc [V]\tJsc [mA/cm2]\tFF [%]\tEff. [%]\tArea [cm-2]\tVmpp [V]\tJmpp [mA/cm2]\tPmpp [mW/cm2]\tRp [Ohmcm2]\tRs [Ohmcm2]\tn\tJ0 [A/cm2]\tRsquare diode region\tTemperature [K]\tAcquis. T [°C]\tRp acquis. software [Ohmcm2]\tRs acquis. software [Ohmcm2]\tFilename\tRemarks\n"
            return out

        def st(st, precis):
            return str(np.round(st, precis + 1))
            # +1: hacked my own function to increase precision for every parameter

        attr = self.get_attributes()
        popt = self.attr("diodefit", [np.nan] * 5)
        rsquare_diode = self.attr("_fitdiode_rsquare_diode", np.nan)

        msg = "{}\t{}\t{}\t{}\t{}\t{}"
        out = msg.format(
            self.cell(),
            st(attr["voc"], 4),
            st(attr["jsc"], 2),
            st(attr["ff"] * 100, 2),
            st(attr["eff"] * 100, 2),
            st(self.area(), 4),
        )
        msg = "\t{}\t{}\t{}\t{}\t{}\t{}"
        out += msg.format(
            st(attr["mpp"][0], 4),
            st(-attr["mpp"][1], 2),
            st(attr["mpp"][2], 2),
            st(attr["rp"], 0),
            st(popt[3], 2),
            st(popt[0], 2),
        )
        msg = "\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t"
        out += msg.format(
            "{:.2e}".format(0.001 * popt[2]),
            st(rsquare_diode, 6),
            st(self.T, 1),
            st(
                attr["acquis soft temperature"]
                if "acquis soft temperature" in attr
                else np.nan,
                1,
            ),
            st(attr["acquis soft rp"] if "acquis soft rp" in attr else np.nan, 1),
            st(attr["acquis soft rs"] if "acquis soft rs" in attr else np.nan, 1),
            attr["filename"],
        )
        out += self.attr("_fitDiodeWarning")  # will be '' if no warning issued.
        out += "\n"
        return out

    def calcVocJscFFEffMPP_print(self):
        """Computes and print the values of basic PV parameters Voc, Jsc, FF, MPP"""
        tmp = self.silent
        self.silent = False
        self.calcVocJscFFEffMPP()
        self.silent = tmp
        return True

    def calcVocJscFFEffMPP(self):
        """Computes the values of basic PV parameters Voc, Jsc, FF, MPP"""
        # remove NaN values from data, as well as saturated datapoints
        V, J = self.cleanDataVJ()
        # calculate Jsc
        Jsc = self.calcJsc(V, J)
        self.update({"Jsc": Jsc})
        # compute Rp, same idea as Jsc (update: V in [-0.2 V, +0.1])
        Rp = self.calcRp(V, J)
        self.update({"Rp": Rp})
        # calculate Voc
        # idea: interpolate JV curve with 3-order polynom between 2 closest
        # datapoints, then look for V @ J=0
        Jabs = np.abs(J)
        if np.abs(np.sum(np.sign(J))) == len(J):
            # J does not cross 0
            Voc = np.nan
        else:
            try:
                i = list(Jabs).index(min(Jabs))
                idx = list(range(i - 1, i + 3) if J[i] < 0 else range(i - 2, i + 2))
                if idx[-1] >= len(V):
                    idx = [-4, -3, -2, -1]
                if idx[0] < 0:
                    idx = [0, 1, 2, 3]
                f = interpolate.interp1d(V[idx], J[idx], 3)  # spline interpolation
                dx = np.abs(V[idx[:-1]] - V[idx[1:]])
                dxmin = np.min(dx[dx > 0])
                maxV = np.max(V[idx])
                x = np.arange(
                    V[idx[0]], V[idx[-1]], dxmin / 100
                )  # tolerance is 1/100 of smallest dx
                x = np.array([min(v, maxV) for v in x])
                Voc = xAtValue(x, f(x), 0, silent=True)
            except ValueError:
                Voc = np.nan
        self.update({"Voc": Voc})
        # calculate Eff (method with interpolation)
        # idea: interpolate JV curve with 3-order polynom between 2 closest
        # datapoints, then look for max(J*V)
        powDens = -J * V
        try:
            i = list(powDens).index(max(powDens))
            i = max(1, min(len(powDens) - 2, i))
            idx = list(
                range(i - 1, i + 3)
                if powDens[i - 1] < powDens[i + 1]
                else range(i - 2, i + 2)
            )
            if idx[0] < 0:
                idx = [0, 1, 2, 3]
            if idx[-1] >= len(powDens):
                idx = [-4, -3, -2, -1]
            f = interpolate.interp1d(V[idx], powDens[list(idx)], kind=3)
            dx = np.abs(V[idx[:-1]] - V[idx[1:]])
            dxmin = np.min(dx[dx > 0])
            maxV = np.max(V[idx])
            x = np.arange(V[idx[0]], V[idx[-1]], dxmin / 100)
            x = np.array([min(v, maxV) for v in x])
            try:
                Vmpp = x[list(f(x)).index(max(f(x)))]
                Pmpp = f(Vmpp)
                Eff = f(Vmpp) / (self.attr("illumPower") / 10)
                # *10 to convert illumpower in mW/cm2
                self.update({"Eff": Eff, "MPP": [Vmpp, -f(Vmpp) / Vmpp, float(Pmpp)]})
            except ValueError as e:
                print("Error CurveJV MPP: cannot identify MPP. Exception:")
                print(e)
                Eff = 0
                self.update({"Eff": 0.0, "MPP": [np.nan, np.nan, np.nan]})
        except ValueError:
            Eff = np.nan
            self.update({"Eff": np.nan, "MPP": [np.nan] * 3})
        # calculate FF
        FF = np.abs(
            Eff * (self.attr("illumPower") / 10) / (Voc * Jsc)
        )  # /10 to convert illumpower in mW/cm2
        self.update({"FF": FF})
        # some printing of the results
        if not self.silent:
            print("Jsc", Jsc)
            print("Voc", Voc)
            print("FF ", FF)
            print("Eff", Eff)
            print("MPP", self.attr("MPP"))

    # Ideal diode equation
    # J (V) = JL - J0 * ( exp( q / (n k T) * V ) - 1 )

    # Equation with resistors
    # J (V) = JL - J0 * ( exp( q / (n k T) * (V + J Rs)) - 1 ) - ( V + J Rs) / Rp

    def func_diodeIdealAbsLog10(self, V, n, Jl, J0):
        return np.log10(np.abs(self.func_diodeIdeal(V, n, Jl, J0)))

    def func_diodeIdealAbsLog(self, V, n, Jl, J0):
        return np.log(np.abs(self.func_diodeIdeal(V, n, Jl, J0)))

    def func_diodeIdeal(self, V, n, Jl, J0):
        return -(Jl - J0 * (np.exp(CST.q / (n * CST.kb * self.T) * V) - 1))

    def fit_func_diodeResistors_AbsLog(
        self, V, logJ, n, Jl, J0, Rs, Rp, Vshift=0, sigma=[]
    ):
        if len(sigma) == 0:
            sigma = [1] * len(V)
        J0, Rs, Rp = np.abs(J0), np.abs(Rs), np.abs(Rp)
        self.fitFixPar = {"Vshift": Vshift, "Jl": Jl, "Rs": Rs, "Rp": Rp}
        # trick: define fit variable J0shift as J(Vshift) in order to help
        # decoupling the effect of n and J0 in the fit function
        # Vshift is where slope of log(V) is highest
        # Change also paramettrization of Rp to get values closer to 1
        J0 = J0 * (np.exp(CST.q * Vshift / (n * CST.kb * self.T)))
        Rp = np.log10(Rp)
        # Fit with n, J0, Rs, Rp
        p0 = np.array([n, J0, Rs, Rp])
        popt, pcov = curve_fit(
            self.func_diodeResistorsAbsLog10_modParam_red, V, logJ, p0=p0, sigma=sigma
        )
        [n, J0, Rs, Rp] = popt
        # Jl is never fitted as is computed as to ensure J(V=0) = Jsc
        # restore inital problem parametrization
        Rp = np.power(10, np.abs(Rp))
        Rs = np.abs(Rs)
        J0 = np.abs(J0) / (np.exp(CST.q * Vshift / (n * CST.kb * self.T)))
        # calculation of Jl
        Jsc = self.attr("Jsc") if self.attr("Jsc") != "" else 0
        Jl = self.JlSuchAsJ0VJsc(Jsc, n, J0, Rs, Rp)
        popt = [n, Jl, J0, Rs, Rp]
        return popt, pcov

    def func_diodeResistorsAbsLog10_modParam(self, V, n, Jl, J0, Rs, Rp):
        Vshift = self.fitFixPar["Vshift"]
        J0 = J0 / (np.exp(CST.q * Vshift / (n * CST.kb * self.T)))
        Rp = np.power(10, Rp)
        return np.log10(np.abs(self.func_diodeResistors(V, n, Jl, J0, Rs, Rp)))

    def func_diodeResistorsAbsLog10_modParam_red(self, V, n, J0, Rs, Rp):
        Vshift = self.fitFixPar["Vshift"]
        Jl = self.fitFixPar["Jl"]
        J0 = np.abs(J0) / (np.exp(CST.q * Vshift / (n * CST.kb * self.T)))
        Rp = np.power(10, Rp)
        return np.log10(np.abs(self.func_diodeResistors(V, n, Jl, J0, Rs, Rp)))

    def func_diodeResistorsAbsLog10(self, V, n, Jl, J0, Rs, Rp):
        return np.log10(np.abs(self.func_diodeResistors(V, n, Jl, J0, Rs, Rp)))

    def func_diodeResistorsAbsLog(self, V, n, Jl, J0, Rs, Rp):
        return np.log(np.abs(self.func_diodeResistors(V, n, Jl, J0, Rs, Rp)))

    def func_diodeResistors_Jsc(self, V, n, Jl, J0, Rs, Rp):
        if np.isnan(n):
            return [np.nan] * len(V)
        #        return self.func_diodeResistors (V, n, Jl, J0, Rs, Rp)
        Jsc = self.attr("Jsc") if self.attr("Jsc") != "" else 0
        return -Jsc + self.func_diodeResistors(V, n, Jl, J0, Rs, Rp)

    def JlSuchAsJ0VJsc(self, Jsc, n, J0, Rs, Rp):
        Jl = (
            J0 * (np.exp(CST.q / (n * CST.kb * self.T) * (-Jsc) * Rs) - 1)
            + ((-Jsc) * Rs) / Rp
        )
        return Jl

    def func_diodeResistors(self, V, n, Jl, J0, Rs, Rp):
        T = self.T
        Jsc = (
            self.attr("Jsc") if self.attr("Jsc") != "" else 0
        )  # but we need to correct for the J*Rs terms
        # compute diode equation using the bissection algorithm (slow but robust)
        Rs = (
            -Rs / 1000
        )  # want kOhm/cm2 as current is given in mA/cm2. Negative value relates to sign convention
        Rp = Rp / 1000  # want kOhm/cm2
        # Jsc was substracted from J, therefore Jl is almost 0
        Jl = self.JlSuchAsJ0VJsc(Jsc, n, J0, Rs, Rp)
        # Jl = J0 * (np.exp(self.q / (n * self.k * T) * (-Jsc) * Rs) - 1) + ((-Jsc) * Rs) / Rp

        def f(J, V, n, Jl, J0, Rs, Rp, Jsc):
            # probe different values of J, with corresponding/unique value of V
            out = -J - (
                Jl
                - J0 * (np.exp(CST.q / (n * CST.kb * T) * (V + (J - Jsc) * Rs)) - 1)
                - (V + (J - Jsc) * Rs) / Rp
            )
            if np.isinf(out).any():  # do not accept inf values
                for i in range(len(out)):
                    if np.isinf(out[i]):
                        out[i] = 1e308 * np.sign(out[i])
            return out

        # initial boundary conditions: need 1 J value with positive fuction value, and another with negative value
        precis = 1e-9  # absolute precision desired on J # for example 1e-7
        Jlow_ = V * 0 - 1e2
        Jhigh = V * 0 + 1e4
        niter = (
            1 + (np.log10(np.abs(max(Jhigh - Jlow_))) - np.log10(precis)) * 3.322
        )  # 15+7 orders of magnitude * 3.322, approx 3.322 iterations to gain x10 in precision
        flow_ = f(Jlow_, V, n, Jl, J0, Rs, Rp, Jsc)
        fhigh = f(Jhigh, V, n, Jl, J0, Rs, Rp, Jsc)
        while (
            not (np.sign(flow_) * np.sign(fhigh) < 0).all() and np.abs(Jlow_[0]) < 1e300
        ):
            Jlow_ = Jlow_ * 1e2
            Jhigh = Jhigh * 1e2
            niter = niter + 2 * 3.322
            flow_ = f(Jlow_, V, n, Jl, J0, Rs, Rp, Jsc)
            fhigh = f(Jhigh, V, n, Jl, J0, Rs, Rp, Jsc)

        # start iterating to find zero of the f function
        #        flow_ = f (Jlow_, V, n, Jl, J0, Rs, Rp)
        #        fhigh = f (Jhigh, V, n, Jl, J0, Rs, Rp)
        for j in range(int(niter)):  # esti stands for estimator
            Jesti = (Jlow_ + Jhigh) * 0.5  # bissection. Slow but most robust
            festi = f(Jesti, V, n, Jl, J0, Rs, Rp, Jsc)
            test = np.sign(festi) == np.sign(flow_)
            for i in range(len(Jesti)):
                if test[i]:
                    Jlow_[i] = Jesti[i]
                    flow_[i] = festi[i]
                else:
                    Jhigh[i] = Jesti[i]
        #                    fhigh[i] = festi[i] # never actually needed
        # return best result estimate
        return (Jlow_ + Jhigh) * 0.5

    def diodeFit_BestGuess(self, V=None, J=None, alsoVShift=True):
        """Returns the initial guess for fit of the J-V curve."""
        if V is None or J is None:
            V, J = self.cleanDataVJ()
            Jsc = self.attr("Jsc") if self.attr("Jsc") != "" else 0
            J += Jsc
        # compute best guesses
        Jl = 0  # illumination current is 0 because Jsc takes that into account
        # numerical derivative - take care of indexing! -  do not want last point
        dlogJdV = (
            (np.log10(np.abs(J[2:-1])) - np.log10(np.abs(J[0:-3])))
            / (V[2:-1] - V[0:-3])
            / 2
            * (10 / np.exp(1))
        )  # (10/np.exp(1)) to convert from log to log10
        idx_maxdlogJdV = len(dlogJdV) - 1
        # to improve
        maxdlogJdV = 0
        for i in np.arange(len(dlogJdV) - 1, 0, -1):
            if dlogJdV[i] > maxdlogJdV:
                idx_maxdlogJdV = i
                maxdlogJdV = dlogJdV[i]
            else:  # assumed we arrived in low current region -> out of interesting region
                # print ('end of search for max derivative', V[i+1], J[i+1], dlogJdV[i])
                break
            if V[i + 1] < 0.15:  # assume the diode behavior should be above 0.15 V
                break
        test = (dlogJdV > 0.9 * maxdlogJdV) * (V[1:-2] > 0.15)
        idx = [i for i in range(len(test)) if test[i]]
        # check is fitting is possible - if there were points with high derivative for V>0.15
        if len(idx) == 0:
            # cell shunted, etc
            print(
                "ERROR CurveJV diodeFit: unable to perform fitting (file",
                self.attr("filename"),
                ").",
            )
            return [np.nan] * (6 if alsoVShift else 5)
        # point with highest numerical derivative
        idx_maxdlogJdV = (
            int(np.median(idx)) + 1
        )  # indexing correspond to J, V curves#        idx_maxdlogJdV = list(dlogJdV).index(maxdlogJdV) + 1 # indexing correspond to J, V curves
        Vshift = V[idx_maxdlogJdV]
        # Rp: actually no calculation, the standard value is the most robust indicator
        Rp = self.attr("Rp")  # Ohm/cm2
        # calculation of n. Starting value between 1 and 2 (* maxV to fit modules)
        maxV = max(1, np.max(V))
        n = CST.q / (CST.kb * self.T * max(0.001, maxdlogJdV)) * 0.75
        n = min(2 * maxV, max(1, n))
        #        Rs = dlogJdV[-1]*0.05  # Ohm/cm2 # rough approximetion, only temporary for I0 estimation! -> not close enough to reality
        Rs = 0.5  # start with a low value
        # J0: assume Jsc=0, invert diode equation
        i = idx_maxdlogJdV
        J0_init = np.abs(Jl - J[i] - V[i] / (Rp / 1000)) / (
            np.exp(CST.q / (n * CST.kb * self.T) * (V[i] + J[i] * (-Rs / 1000))) - 1
        )
        if J0_init == 0:  # must prove a sensical starting value
            J0_init = 1e-5
        J0 = J0_init
        # update previous crude value of Rs, by calculation at last point
        Rs = (
            -1000
            * 1
            / J[-2]
            * (n * CST.kb * self.T / CST.q * np.log(1 + (J[-2] - Jl) / J0) - V[-2])
        )  # in front for sign convention+units : -1/1000
        # without Rp. normally good enough Rs_ = -1000 * 1/J[-2] * (n*self.k*self.T/self.q * np.log(1 + (J[-2]-Jl+V[-2]/Rp*1000)/J0) - V[-2]) # in front for sign convention+units : -1/1000
        if dlogJdV[-1] > dlogJdV[-2]:
            Rs = 0
        Rs = max(0, Rs)
        if alsoVShift:
            return [n, Jl, J0, Rs, Rp, Vshift]
        return [n, Jl, J0, Rs, Rp]

    def diodeFit(self, fitRange=None, ifPlot=False, silent=True):
        """V has to be should be sorted, increasing"""
        # select dataset: remove saturated dsatapoints, select according to fitRange
        diodeFitWeight = (
            self.attr("_fitDiodeWeight") if self.attr("_fitDiodeWeight") != "" else 0
        )
        V, J, Jsc_ = self.selectJVdataForFit(fitRange=fitRange)
        Jsc = self.attr("Jsc")
        J_ = deepcopy(J)
        J += Jsc
        JLogAbs = np.log10(np.abs(J))

        # catch best guess for parameters
        [n, Jl, J0, Rs, Rp, Vshift] = self.diodeFit_BestGuess(V=V, J=J, alsoVShift=True)
        if np.isnan(n):
            self.update({"diodeFit": [np.nan] * 5})
            return [np.nan] * 5

        # estimation of noise: minimal noise level, plus relative proportonal
        # to input signal; assuming 8e-3 fluctuations in signal (ie. lamp), and
        # base noise 1e-5*max
        noise = 1e-5 * max(np.abs(J_)) + 8e-3 * np.abs(J_)  # J-Jsc
        signalToNoise = smooth(np.abs(J_) + 1e-4 * max(np.abs(J)) / noise, 5)
        sigma = 1 / signalToNoise
        #        sigma = np.power(np.abs(np.log10(np.abs(J+sigma)) - np.log10(np.abs(J))/np.log10(np.abs(J))), 0.25)
        sigma = np.power(
            np.abs(
                np.log10(np.abs(J + sigma)) - np.log10(np.abs(J)) / np.log10(np.abs(J))
            ),
            0.35,
        )
        idx_DiodeRegion = self.idxFitHighSensitivityNJ0(
            V, derivative(V, np.log10(np.abs(J)))
        )
        if diodeFitWeight != 0:
            for i in idx_DiodeRegion:
                sigma[i] /= diodeFitWeight

        # perform actual fit
        try:
            popt, pcov = self.fit_func_diodeResistors_AbsLog(
                V, JLogAbs, n, Jl, J0, Rs, Rp, Vshift=Vshift, sigma=sigma
            )
        except RuntimeError as e:
            print("ERROR CurveJV diodeFit:", e)
            return False

        # assess fit quality in diode behavior region
        JfitLogAbs = np.log10(
            np.abs(
                self.func_diodeResistors(V, popt[0], popt[1], popt[2], popt[3], popt[4])
            )
        )
        sqResiduals = (JfitLogAbs - JLogAbs) ** 2
        self.diodeFitCheckQuality(sqResiduals[idx_DiodeRegion])

        ss_res = np.sum(sqResiduals[idx_DiodeRegion])
        JLogAbs_avg = np.average(JLogAbs[idx_DiodeRegion])
        ss_tot = np.sum((JLogAbs[idx_DiodeRegion] - JLogAbs_avg) ** 2)
        rsquare_diode = 1 - ss_res / ss_tot

        if not silent:
            print("fit diode + resistors")
            print("   param, inital, fit")
            print("   n ", n, popt[0], "")
            print("   Jl", Jl, popt[1], "mA/cm2")
            print("   J0", J0, popt[2], "mA/cm2")
            print("   J0", J0 / 1000, popt[2] / 1000, "A/cm2")
            print("   Rs", Rs, popt[3], "Ohmcm2")
            print("   Rp", Rp, popt[4], "Ohmcm2")
            print("   Rsquare diode region", rsquare_diode)

        if ifPlot:
            x = V
            guess = self.func_diodeResistors(V, n, Jl, J0, Rs, Rp)
            fit = self.func_diodeResistors(
                x, popt[0], popt[1], popt[2], popt[3], popt[4]
            )

            self.fitFixPar = {
                "Vshift": Vshift,
                "Jl": popt[1],
                "Rs": popt[3],
                "Rp": popt[4],
            }

            units = self.data_units()

            fig, ax = plt.subplots()
            ax.plot(V, np.log10(np.abs(J)), "bx")
            # ax.plot (V, self.func_diodeIdealAbsLog     (V, n, T, Jl, J0ideal), 'r')
            ax.plot(V, np.log10(np.abs(guess)), "r")
            ax.plot(x, np.log10(np.abs(fit)), "k")
            #  J0_mod  = popt[2] * (np.exp(self.q * Vshift / (popt[0] * self.k * self.T)))
            #  bestfi_=self.func_diodeResistorsAbsLog10_modParam_red   (x, popt[0]-1, J0_mod*0.6)
            #  ax.plot (x, bestfi_, 'c')
            ax.set_xlabel("Bias voltage [" + units[0] + "]")
            ax.set_ylabel("log(current density [" + units[1] + "])")

            fig, ax = plt.subplots()
            ax.plot(V, -Jsc + J, "bx")
            ax.plot(V, -Jsc + guess, "r")
            ax.plot(x, -Jsc + fit, "k")
            ax.set_xlabel("Bias voltage [" + units[0] + "]")
            ax.set_ylabel("Current density [" + units[1] + "]")
        #            ax.plot (V, sigma, 'c')

        self.update({"diodefit": popt, "_fitdiode_rsquare_diode": rsquare_diode})
        return popt

    def fit_resampleX(self, Vspacing):
        """Specifies new basis for datapoints for the x data, and recomputes the fitted
        values on the new basis of points"""
        if not self.has_attr("_popt"):
            msg = "Fit_resampleX: should only execute on fitted curve ('_popt' defined)"
            print(msg)
            return msg
        newV = ""
        if isinstance(Vspacing, float):
            Vspacing = np.abs(Vspacing)
            newV = np.arange(self.V(0), self.V(-1) + Vspacing, Vspacing)
        if isinstance(Vspacing, list):
            if len(Vspacing) == 3:
                newV = np.arange(Vspacing[0], Vspacing[1], Vspacing[2])
            else:
                newV = np.array(Vspacing)
        # if valid input, do something
        if len(newV) > 0 or newV != "":
            # recreate data container, with new x values both in x and y
            # positions. next step it to compute new y values
            self.data = np.array([newV, newV])
            self.updateFitParam(*self.attr("_popt"))
            return 1
        return "Invalid input."

    def calcJsc(self, V=None, J=None):
        # idea: fit 1-degree polynom over 7 points close to V=0
        if V is None:
            V = self.V()
        if J is None:
            J = self.J()
        # idea: fit 1-degree polynom over 7 points close to V=0
        Vabs = np.abs(V)
        try:
            i = list(Vabs).index(min(Vabs))
        except ValueError:  # if "fake" Curve with a only a single nan inside
            return np.nan
        idx = list(range(i - 2, i + 5) if V[i] < 0 else range(i - 3, i + 4))
        if idx[0] < 0:
            idx = [0, 1, 2, 3, 4, 5]
        idx = [i for i in idx if 0 <= i < len(V)]
        z = np.polyfit(V[idx], J[idx], 1, full=True)[0]
        return -z[1]

    def calcRp(self, V=None, J=None):
        if V is None:
            V = self.V()
        if J is None:
            J = self.J()
        test = (V > -0.2) * (V < +0.1)
        idx = [i for i in range(len(test)) if test[i]]
        if idx == []:
            Vabs = np.abs(V)
            try:
                i = list(Vabs).index(min(Vabs))
            except ValueError:  # fake JVCurve with only a NaN inside
                return np.nan
            idx = list(range(i - 2, i + 5) if V[i] < 0 else range(i - 3, i + 4))
        if idx[0] < 0:
            idx = [0, 1, 2, 3, 4, 5]
        idx = [i_ for i_ in idx if i_ < len(V)]
        z = np.polyfit(V[idx], J[idx], 1, full=True)[0]
        Rp = (
            1 / z[0] * 1000
        )  # *1000 to get in Ohm/cm2 (J is in mA/cm2, so slope fit slope is in kOhm/cm2)
        if Rp < 0:
            Rp = 1e5
            print("CurveJV calcRP: Caution! Rp was found < 0. Set to", Rp, ".")
        return Rp

    def idxRange(self, V, fitRange=None):
        if is_number(fitRange):
            fitRange = [fitRange, np.inf]
        if fitRange is None or not isinstance(fitRange, list):
            fitRange = [-0.5, np.inf]
        idx = []
        try:
            for i in range(len(V)):
                if np.min(fitRange) < V[i] < np.max(fitRange) and np.abs(V[i]) > 0.1:
                    idx.append(i)
        except Exception:
            fitRange = [-0.5, np.inf]
            print(
                "ERROR CurveJV fit: fit range not understandable. Value set to",
                fitRange,
                ".",
            )
            for i in range(len(V)):
                if np.min(fitRange) < V[i] < np.max(fitRange) and np.abs(V[i]) > 0.1:
                    idx.append(i)
        return idx

    def cleanDataVJ(self, V=None, J=None):
        """Return data stripped of saturated datapoints."""
        if V is None:
            V = deepcopy(self.V())
        if J is None:
            J = deepcopy(self.J())
        # remove NaN points
        isnan_ = np.isnan(V) + np.isnan(J)
        V = np.array([V[i] for i in range(len(V)) if not isnan_[i]])
        J = np.array([J[i] for i in range(len(J)) if not isnan_[i]])
        # identify and remove saturated data points
        nSatur = 0
        for i in np.arange(len(J) - 1, 0, -1):
            if (J[i] - J[i - 1]) / J[i] < 1e-4:
                nSatur = nSatur + 1
                J[i] = np.nan
            else:
                break
        if nSatur > 0:
            V = V[: -nSatur - 1]
            J = J[: -nSatur - 1]
        # check if duplicates on V
        if len(np.unique(V)) != len(V):
            print(
                "WARNING CurveJV: duplicate found in V data, unpredictable results!",
                self.attr("filename"),
            )
        # check if V is in ascending datapoint order
        is_sorted = all(a <= b for a, b in zip(V, V[1:]))
        if not is_sorted:
            print(
                "Curve JV: input data are not sorted. Presumably ok.",
                self.attr("filename"),
            )
            J = np.array([x for _, x in sorted(zip(V, J))])
            V = np.sort(V)
        return V, J

    def selectJVdataForFit(self, V=None, J=None, fitRange=None):
        """Returns dataseet suitable for fitting, taking into account a fit range given by user.
        Saturated datapoints are striped.
        Caution: the value of Jsc return in only computed on the fitRange.
        """
        if V is None:
            V = deepcopy(self.V())
        if J is None:
            J = deepcopy(self.J())
        # strip saturated datapoints
        V, J = self.cleanDataVJ(V, J)
        # fit range
        idx = self.idxRange(V, fitRange=fitRange)
        Jsc = self.calcJsc(V[idx], J[idx])
        return V[idx], J[idx], Jsc

    def idxFitHighSensitivityNJ0(self, V, dJLogAbsdV, threshold=0.70, minWidth=None):
        """
        threshold: criterion to select datapoint is that derivative is > threshold*max(deriv)
        also winimal voltage width for example [-0.05, 0.1]
        """
        if minWidth is None:
            minWidth = [-0.1, 0.1]

        # for simplicity rename variable
        dJdV = dJLogAbsdV

        # restrict find maximum to V>0.2, make it more robust
        dJdV02V = [dJdV[i] for i in range(len(dJdV)) if V[i] > 0.2]
        maxDerivIdx = np.where(dJdV == max(dJdV02V))[-1][-1]
        # print('CurveJV idxFitHighSensitivityNJ0', maxDerivIdx, type(maxDerivIdx))
        # print('   ',np.where(dJdV == max(dJdV02V)))
        regMaxDerivIdx = [maxDerivIdx]
        flag = True
        for i in range(maxDerivIdx + 1, len(V)):
            if dJdV[i] < threshold * dJdV[maxDerivIdx]:
                flag = False
            if flag or V[i] - V[maxDerivIdx] < minWidth[1]:
                regMaxDerivIdx.append(i)
            if not flag and V[i] - V[maxDerivIdx] > minWidth[1]:
                break
        flag = True
        for i in range(maxDerivIdx - 1, -1, -1):
            if dJdV[i] < threshold * dJdV[maxDerivIdx]:
                flag = False
            if flag or V[maxDerivIdx] - V[i] < np.abs(minWidth[0]):
                regMaxDerivIdx.append(i)
            if not flag and V[i] - V[maxDerivIdx] > minWidth[1]:
                break
        regMaxDerivIdx = sorted(regMaxDerivIdx)
        return regMaxDerivIdx

    def calcResiduals(self, diodeFit=None, ifPlot=False):
        V, J, Jsc = self.selectJVdataForFit(fitRange=[0.1, np.inf])
        # do not want to use the Jsc calculated with restrictions
        Jsc = self.attr("Jsc")
        J += Jsc
        JLogAbs = np.log10(np.abs(J))
        dJdV = derivative(V, JLogAbs)
        d2JdV2 = derivative(V, derivative(V, JLogAbs))
        p = diodeFit if diodeFit is not None else self.attr("diodeFit")
        if p == "":
            p = self.diodeFit()
        Jfit = self.func_diodeResistors(V, p[0], p[1], p[2], p[3], p[4])
        JfitLogAbs = np.log10(np.abs(Jfit))

        # define fitting regions
        # region 0 is dominated by Rp
        # region 1 is below onset of diode, maybe non ideal behavior
        # region 2 is diode beavior (starting close to inflexion point)
        # region 3 is dominated by Rs (starting at inflexion point in log plot)
        regionIdx = [[], [], [], []]
        # fill points above max derivative in log plot
        reg = 3
        for i in range(len(J) - 1, -1, -1):
            if reg == 3 and d2JdV2[i] > 0:
                reg = 2
                M = d2JdV2[i]
            if reg == 2 and JLogAbs[i] > M:
                M = d2JdV2[i]
            if reg == 2 and d2JdV2[i] < 0.1 * M and regionIdx[3][-1] > i + 4:
                reg = 1
                Vreg2 = V[i]
            if reg == 1 and V[i] < Vreg2 / 2:
                reg = 0
            regionIdx[reg].append(i)
        for i in range(len(regionIdx)):
            regionIdx[i] = sorted(regionIdx[i])
        sqResiduals = (JfitLogAbs - JLogAbs) ** 2
        regionSqResAvg = [
            np.average(sqResiduals[regionIdx[i]]) if len(regionIdx[i]) > 0 else np.nan
            for i in range(len(regionIdx))
        ]
        regionSqResMed = [
            np.median(sqResiduals[regionIdx[i]]) if len(regionIdx[i]) > 0 else np.nan
            for i in range(len(regionIdx))
        ]
        print(
            "   regions start ",
            *roundSignificant(
                [V[regIdx[0]] if len(regIdx) > 0 else np.nan for regIdx in regionIdx], 3
            )
        )
        #        print ('regionSqResAverage', roundSignificant(regionSqResAvg,3))
        print("   regionSqResMedian ", *roundSignificant(regionSqResMed, 3))

        # determine a region close to the maximum slope
        regMaxDerivIdx = self.idxFitHighSensitivityNJ0(V, dJdV)
        maxDerivSqResMed = (
            np.median(sqResiduals[regMaxDerivIdx])
            if len(regMaxDerivIdx) > 0
            else np.nan
        )
        print("   MaxDerivSqResMedian ", roundSignificant(maxDerivSqResMed, 3))
        alertThreshold = 2e-4
        if maxDerivSqResMed > 2e-4:
            print(
                "CurveJV fit quality check: n and I0 probably off! (sq. residuals",
                "{:1.4f}".format(maxDerivSqResMed),
                ", alert threshold",
                "{:1.4f}".format(alertThreshold),
                ")",
            )

        if ifPlot:
            fig = plt.figure()
            ax1 = plt.subplot(211)
            ax1.set_title(self.attr("filename"))
            plt.plot(self.V(), np.log10(np.abs(self.J() + Jsc)), "k", label="")
            plt.plot(V, JLogAbs, ".k", label="data")
            plt.plot(V, JfitLogAbs, "r", label="fit")
            for i in range(len(regionIdx)):
                if len(regionIdx[i]) > 0:
                    plt.axvline(V[regionIdx[i][0]], 0, 1, color="k")
            plt.axvline(V[regMaxDerivIdx[0]], 0, 1, color="r")
            plt.axvline(V[regMaxDerivIdx[-1]], 0, 1, color="r")
            ax0 = ax1.twinx()
            ax0.plot(V, sqResiduals, label="Squared residuals")
            ax0.set_ylim([0, 0.03])
            ax1.legend(loc="best", fancybox=True, framealpha=0.5)
            ax0.legend(loc="best", fancybox=True, framealpha=0.5)
            ax1.set_ylim(bottom=-3)

            ax3 = plt.subplot(212, sharex=ax1)
            plt.axhline(0, 0, 1, color="k")
            plt.plot(V, derivative(V, dJdV), label="2nd derivative")
            for i in range(len(regionIdx)):
                if len(regionIdx[i]) > 0:
                    plt.axvline(V[regionIdx[i][0]], 0, 1, color="k")
            ax3.legend(loc="best", fancybox=True, framealpha=0.5)
            ax2 = ax3.twinx()
            plt.plot(V, dJdV, "r")

        return regMaxDerivIdx

    def diodeFitCheckQuality(self, sqResiduals, threshold=2e-4):
        diodeFitWeight = (
            self.attr("_fitDiodeWeight") if self.attr("_fitDiodeWeight") != "" else 0
        )
        SqResMed = np.median(sqResiduals) if len(sqResiduals) > 0 else np.nan
        if SqResMed > threshold:
            print(
                "CurveJV diode fit check: probably bad fit of n and I0!\n",
                "   (sq. res.",
                "{:1.4f}".format(SqResMed),
                ", alert thres.",
                "{:1.4f}".format(threshold),
                ", fit diode weight",
                "{:1.1f}".format(diodeFitWeight),
                ")",
            )
            self.update(
                {
                    "_fitDiodeWarning": "WARNING: n and I0 probably off (sq. res. "
                    + "{:1.4f}".format(SqResMed)
                    + ", threshold "
                    + "{:1.4f}".format(threshold)
                    + ")"
                }
            )
        else:
            self.update({"_fitDiodeWarning": ""})

    def extractJscVoc(self, graph):
        """
        Extract Jsc-Voc: extracts Jsc, Voc and T datapoints of selected JV
        curve, and appends the results to new/existing Curves identified with
        keywords "_extractJscVoc_Jsc" and "_extractJscVoc_T".
        """
        identifierJ = "_extractJscVoc_Jsc"
        identifierT = "_extractJscVoc_T"
        if not isinstance(graph, Graph):
            print("CurveJV.extractJscVoc(): must provide a Graph as parameter")
            return False
        self.calcVocJscFFEffMPP()
        jsc = self.attr("jsc")
        voc = self.attr("voc")
        flag = False
        for c in range(len(graph)):  # store data in suitable curve
            if graph[c].attr(identifierJ, False):
                graph[c].appendPoints([voc], [jsc])
                if (
                    c + 1 < len(graph)
                    and graph[c + 1].attr(identifierT, False)
                    and len(graph[c].x()) == len(graph[c + 1].x()) + 1
                ):
                    graph[c + 1].appendPoints([voc], [self.T])
                else:
                    msg = "WARNING: CurveJV extractJscVoc: could not identify curve to store temperature, or data length mismatch"
                    print(msg)
                flag = True
                break
        if not flag:
            attr = {
                "label": "Voc-Jsc pair",
                "linespec": "o",
                "type": "scatter",
                identifierJ: True,
            }
            curve0 = Curve([[voc], [jsc]], attr)
            graph.append(curve0)
            curve1 = Curve([[voc], [self.T]], {identifierT: True, "type": "scatter_c"})
            graph.append(curve1)
            graph.castCurve("CurveJscVoc", -2, silentSuccess=True)
        return True
