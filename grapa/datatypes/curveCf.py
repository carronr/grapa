# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:37:38 2017

@author: Romain Carron
Copyright (c) 2024, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np

from grapa.mathModule import is_number, derivative
from grapa.curve import Curve


class CurveCf(Curve):
    CURVE = "Curve Cf"

    FORMAT_AUTOLABEL = [
        "${sample} ${cell} ${temperature [k], :.0f} K",
        "${temperature [k], :.0f} K",
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
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
        # auto-label
        out.append(
            [
                self.autoLabel,
                "Auto label",
                ["template"],
                [self.FORMAT_AUTOLABEL[0]],
                {},
                [{"field": "Combobox", "values": self.FORMAT_AUTOLABEL}],
            ]
        )
        # set area
        out.append([self.setArea, "Set cell area", ["New area"], [self.getArea()]])
        # help
        out.append([self.printHelp, "Help!", [], []])
        return out

    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        # out.append(['nm <-> eV', ['nmeV', ''], ''])
        out.append(["semilogx", ["", "idle"], "semilogx"])
        out.append(["-dC / dln(f)", ["", "CurveCf.y_mdCdlnf"], "semilogx"])
        out.append(
            [
                "Frequency [Hz] vs Apparent depth [nm]",
                ["CurveCV.x_CVdepth_nm", "x"],
                "semilogy",
            ]
        )
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
        attr.update({"sample name": self.getAttribute("sample name")})
        T = np.nan
        for key in ["temperature [k]", "temperature"]:
            T = self.getAttribute(key, np.nan)
            if not np.isnan(T):
                if not is_number(T):
                    T = float(T)
                break
        if not np.isnan(T):
            msg = "Data picker CurveCf T = {}, omega = {}."
            print(
                msg.format(
                    T,
                    self.x(idx)[0] * 2 * np.pi,
                )
            )
            return T, self.x(idx)[0] * 2 * np.pi, attr
        return Curve.getDataCustomPickerXY(self, idx, **kwargs)

    def printHelp(self):
        print("*** *** ***")
        print("CurveCf offer basic treatment of C-f curves of solar cells.")
        print("Default units are frequency [Hz] and [nF] (or [nF cm-2]).")
        print("Curve transforms:")
        print(" - Linear: standard is Capacitance per area [nF cm-2] vs [Hz].")
        print(" - semilogx: horizontal axis is log(f) instead of f.")
        print(
            " - -dC / dln(f): derivative of C with ln(f), to identify",
            "inflection points.",
        )
        print("Analysis functions:")
        print(
            " - Set area: can normalize input data. For proper data",
            "analysis the units should be [nF cm-2].",
        )
        print("Further analysis:")
        print(
            " - Report inflection point for different T, then the fit",
            "activation energy.",
        )
        print("   Traps can follow omega = 2 ksi T^2 exp(- E_omega / (kT)")
        print("   plot ln(omega T^-2) = ln(2 ksi) - E_omega / (kT)")
        print("References:")
        print(
            "Decock et al., J. Appl. Phys. 110, 063722 (2011);",
            "doi: 10.1063/1.3641987",
        )
        print("Walter et al., J. Appl. Phys. 80, 4411 (1996);", "doi: 10.1063/1.363401")
        return True
