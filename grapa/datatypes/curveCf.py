# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:37:38 2017

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np

from grapa.mathModule import is_number, derivative
from grapa.curve import Curve
from grapa.utils.funcgui import FuncListGUIHelper, FuncGUI, AlterListItem

STR_FURTHER_ANALYSIS = (
    "\nFurther analysis:\n"
    "- Measure and report the inflection points obtained at different\n"
    "  temperatures, then fit the corresponding activation energy.\n"
    "  Traps can follow `omega = 2 ksi T^2 exp(- E_omega / (kT)`, therefore\n"
    "  plot `ln(omega T^-2) = ln(2 ksi) - E_omega / (kT)`\n"
    "References:\n"
    "- Decock et al., J. Appl. Phys. 110, 063722 (2011); doi: 10.1063/1.3641987\n"
    "- Walter et al., J. Appl. Phys. 80, 4411 (1996); doi: 10.1063/1.363401\n"
)


class CurveCf(Curve):
    """
    CurveCf offer basic treatment of C-f curves of solar cells.
    Default data units are frequency [Hz], and [nF] (or [nF cm-2]).
    """

    CURVE = "Curve Cf"

    _label_f = ["Frequency", "f", "Hz"]
    _label_c = ["Capacitance", "C", "nF cm$^{-2}$"]
    AXISLABELS_X = {
        "": _label_f,
        "CurveCV.x_CVdepth_nm": ["Apparent depth", "", "nm"],
    }
    AXISLABELS_Y = {
        "": _label_c,
        "idle": _label_c,
        "CurveCf.y_mdCdlnf": ["Derivative -f dC/df", "", ""],
        "x": _label_f,
    }

    FORMAT_AUTOLABEL = [
        "${sample} ${cell} ${temperature [k]:.0f} K",
        "${temperature [k]:.0f} K",
        "${sample}",
    ]

    def __init__(self, data, attributes, silent=False):
        """
        Constructor with minimal structure: Curve.__init__, and set the
        'Curve' parameter.
        """
        Curve.__init__(self, data, attributes, silent=silent)
        self.update({"Curve": CurveCf.CURVE})

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # auto-label
        line = FuncGUI(self.label_auto, "Auto label")
        line.appendcbb("template", self.FORMAT_AUTOLABEL[0], self.FORMAT_AUTOLABEL)
        out.append(line)
        # set area
        out.append([self.setArea, "Set cell area", ["New area"], [self.getArea()]])
        # help
        out.append([self.print_help, "Help!", [], []])

        out += FuncListGUIHelper.graph_axislabels(self, **kwargs)

        self._funclistgui_memorize(out)
        return out

    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        # doc = "Linear: standard is Capacitance per area [nF cm-2] vs [Hz]."
        doc = "semilogx: horizontal axis is log(f) instead of f."
        out.append(AlterListItem("semilogx", ["", "idle"], "semilogx", doc))

        doc = "derivative of C with ln(f), to identify inflection points."
        out.append(
            AlterListItem("-dC / dln(f)", ["", "CurveCf.y_mdCdlnf"], "semilogx", doc)
        )

        lbl = "frequency [Hz] versus apparent depth [nm]"
        out.append(AlterListItem(lbl, ["CurveCV.x_CVdepth_nm", "x"], "semilogy", ""))
        return out

    def y_mdCdlnf(self, index=np.nan, xyValue=None):
        # if ylim is set, keep the ylim restriction and not do not ignore ylim
        if xyValue is not None:
            return xyValue[1]
        # do not want to use self.x(index) syntax as we need more points to
        # compute the local derivative. HEre we compute over all data, then
        # restrict to the desired datapoint
        val = -derivative(np.log(self.x(xyValue=xyValue)), self.y(xyValue=xyValue))
        if len(val) > 0:
            if np.isnan(index).any():
                return val[:]
            return val[index]
        return val

    # FUNCTIONS RELATED TO GUI (fits, etc.)
    def setArea(self, value):
        """To normalize input capacitance values. This module assumes data input in unit
        [nF cm-2].

        :param value: the new device area value, presumably in cm2."""
        oldArea = self.getArea()
        self.update({"cell area (cm2)": value})
        self.setY(self.y() / value * oldArea)
        return True

    def getArea(self):
        return self.attr("cell area (cm2)", 1)

    # Function related to data picker
    def getDataCustomPickerXY(self, idx, **kwargs):
        """
        Overrides Curve.getDataCustomPickerXY
        Returns Temperature, omega instead of f, C
        """
        if "strDescription" in kwargs and kwargs["strDescription"]:
            return "(Temperature, omega) instead of (f, C)"
        # actually fo things
        try:
            from grapa.datatypes.curveArrhenius import CurveArrheniusCfdefault

            attr = CurveArrheniusCfdefault.attr
        except Exception:
            attr = {
                "Curve": "Fit Arrhenius",
                "_Arrhenius_variant": "Cfdefault",
                "label": "omega vs Temperature",
            }
            print(
                "WARNING Exception during opening of CurveArrhenius module.",
                " Does not perturb saving of the data.",
            )
        attr.update({"sample name": self.attr("sample name")})
        T = np.nan
        for key in ["temperature [k]", "temperature"]:
            T = self.attr(key, np.nan)
            if not np.isnan(T):
                if not is_number(T):
                    T = float(T)
                break
        if not np.isnan(T):
            msg = "Data picker CurveCf T = {}, omega = {}."
            print(msg.format(T, self.x(idx)[0] * 2 * np.pi))
            return T, self.x(idx)[0] * 2 * np.pi, attr
        return Curve.getDataCustomPickerXY(self, idx, **kwargs)

    def print_help(self):
        super().print_help()
        print(STR_FURTHER_ANALYSIS)
        return True
