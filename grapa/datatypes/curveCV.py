# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2023, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

import numpy as np
import warnings

from grapa.curve import Curve
from grapa.mathModule import roundSignificant, derivative, is_number


class CurveCV(Curve):
    CURVE = "Curve CV"

    CST_q = 1.6021766208e-19  # C # electrical charge
    CST_eps0 = 8.85418782e-12  # m-3 kg-1 s4 A2 # vacuum permittivity
    CST_epsR = 10  # relative permittivity

    CST_MottSchottky_Vlim_def = [0, 0.4]
    CST_MottSchottky_Vlim_adaptative = [-0.5, np.inf]

    FORMAT_AUTOLABEL = [
        "${sample} ${cell} ${temperature [k], :.0f} K",
        "${temperature [k], :.0f} K",
        "${sample}",
    ]

    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        # for saved files further re-reading
        self.update({"Curve": CurveCV.CURVE})

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
        # fit Mott-Schottky
        if self.attr("_fitFunc", None) is None or self.attr("_popt", None) is None:
            out.append(
                [
                    self.CurveCV_fitVbiN,
                    "Fit Mott-Schotty",
                    ["ROI [V]"],
                    [CurveCV.CST_MottSchottky_Vlim_def],
                ]
            )
            out.append(
                [
                    self.CurveCV_fitVbiN_smart,
                    "Fit Mott-Schotty (around min N_CV)",
                    ["Within range [V]"],
                    [CurveCV.CST_MottSchottky_Vlim_adaptative],
                ]
            )
        else:
            param = roundSignificant(self.attr("_popt"), 5)
            out.append(
                [
                    self.updateFitParam,
                    "Update fit",
                    ["V_bi", "N_CV"],
                    [param[0], "{:1.4e}".format(param[1])],
                ]
            )
        # curve extraction for doping at 0 V
        if self.attr("_curvecv_0V", None) is None:
            out.append([self.CurveCV_0V, "Show doping at", ["V="], [0]])
        else:
            volt = self.x()[2]
            scrw = self.x_offsets(alter="CurveCV.x_CVdepth_nm")[2]
            doping = self.y_offsets(alter="CurveCV.y_CV_Napparent")[2]
            scrw, doping = roundSignificant([scrw, doping], 5)
            out.append(
                [
                    None,
                    "Doping at {} V".format(volt),
                    ["Depth [nm]", "Carrier density N_CV [cm-3]"],
                    [scrw, doping],
                    {},
                    [{"width": 9}, {"width": 11}],
                ]
            )
        # set epsilon
        out.append(
            [
                self.setEpsR,
                "Set epsilon r \u03F5\u1D63",
                ["default = 10"],
                [self.getEpsR()],
            ]
        )
        # help
        out.append([self.printHelp, "Help!", [], []])
        # 1/C2 = 2 / q / Nc / Ks / eps0 / A2 (Vbi-V)

        return out

    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        # out.append(['nm <-> eV', ['nmeV', ''], ''])
        out.append(["Mott-Schottky (1/C^2 vs V)", ["", "CurveCV.y_ym2"], ""])
        out.append(
            ["Carrier density N_CV [cm-3] vs V", ["", "CurveCV.y_CV_Napparent"], ""]
        )
        out.append(
            [
                "Carrier density N_CV [cm-3] vs depth [nm]",
                ["CurveCV.x_CVdepth_nm", "CurveCV.y_CV_Napparent"],
                "",
            ]
        )
        return out

    # FUNCTIONS used for curve transform (alter)
    def y_ym2(self, xyValue=None, **kwargs):
        """Mott-Schottky plot: 1 / C**2"""
        if xyValue is not None:
            return xyValue[1]
        return 1 / (self.y(**kwargs) ** 2)

    def x_CVdepth_nm(self, **kwargs):
        """apparent probing depth, assuming planar capacitor."""
        eps_r = CurveCV.getEpsR(self)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # y defined as nF/cm2, output depth in [nm]
            w = eps_r * CurveCV.CST_eps0 / (1e-5 * self.y(**kwargs)) * 1e9
        return w

    def y_CV_Napparent(self, xyValue=None, **kwargs):
        """apparent carrier density N_CV"""
        if xyValue is not None:
            return xyValue[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # C from [nF cm2] to [F m-2]
            dCm2dV = derivative(self.x(**kwargs), (1e-5 * self.y(**kwargs)) ** (-2))
        eps_r = CurveCV.getEpsR(self)
        # output unit in [cm-3]
        N = -2 / (CurveCV.CST_q * eps_r * CurveCV.CST_eps0 * (dCm2dV)) * 1e-6
        return N

    # FUNCTIONS RELATED TO GUI (fits, etc.)
    def setArea(self, value):
        oldArea = self.getArea()
        self.update({"cell area (cm2)": value})
        self.setY(self.y() / value * oldArea)
        return True

    def getArea(self):
        return self.attr("cell area (cm2)", 1)

    # handle custom-defined eps_r
    def getEpsR(self):
        if self.attr("epsR", 0) != 0:
            return self.attr("epsR")
        return CurveCV.CST_epsR

    def setEpsR(self, value):
        self.update({"epsR": value})

    # functions for fitting Mott-Schottky plot
    def CurveCV_fitVbiN(self, Vlim=None, silent=False):
        """
        Returns a Curve based on a fit on Mott-Schottky plot.
        Calls fit_MottSchottkyto get parameters, then call func_MottSchottky
        to generate data.
        """
        Vbi, N_CV = self.fit_MottSchottky(Vlim=Vlim)
        fname = self.attr("filename").split("/")[-1].split("\\")[-1]
        attr = {
            "color": "k",
            "area": self.getArea(),
            "_popt": [Vbi, N_CV],
            "_fitFunc": "func_MottSchottky",
            "_Vlim": Vlim,
            "filename": "fit to " + fname,
        }
        if not silent:
            print(
                "Fit Mott-Schottky plot: Vbi =",
                Vbi,
                "V, apparent doping",
                "N_CV =",
                "{:1.4e}".format(N_CV) + ".",
            )
        return CurveCV([self.x(), self.func_MottSchottky(self.x(), Vbi, N_CV)], attr)

    def CurveCV_fitVbiN_smart(self, Vrange=None, window=[-2, 2], silent=False):
        """
        Returns a Curve based on a fit on Mott-Schottky plot, after first
        guessing best possible range for fifting (where N_CV is lowest)
        """
        Vlim = self.smartVlim_MottSchottky(Vlim=Vrange, window=window)
        return self.CurveCV_fitVbiN(Vlim=Vlim, silent=False)

    def fit_MottSchottky(self, Vlim=None):
        """
        Fits the C-V data on Mott-Schottky plot, in ROI Vlim[0] to Vlim[0].
        Returns built-in voltage Vbi [V], apparent doping density N_CV [cm-3].
        """
        datax, datay = self.selectData(xlim=Vlim)
        if len(datax) == 0:
            return np.nan, np.nan
        datay = 1 / (datay * 1e-5) ** 2  # calculation in SI units [F m-2]
        z = np.polyfit(datax, datay, 1, full=True)[0]
        Vbi = -z[1] / z[0]
        N_CV = -2 / (CurveCV.CST_q * CurveCV.getEpsR(self) * CurveCV.CST_eps0) / z[0]
        return Vbi, N_CV * 1e-6  # N_CV in [cm-3]

    def func_MottSchottky(self, V, Vbi, N_CV):
        """
        Returns C(V) which will appear linear on a Mott-Schottky plot.
        V_bi built-in voltage [V]
        N_CV apparent doping density [cm-3]
        Output: C [nF cm-2]
        """
        out = V * np.nan
        if np.isnan(Vbi) or N_CV < 0:
            return out
        mask = V < Vbi
        Cm2 = (
            2
            / (CurveCV.CST_q * CurveCV.getEpsR(self) * CurveCV.CST_eps0 * (N_CV * 1e6))
            * (Vbi - V)
        )
        out[mask] = Cm2[mask] ** (-0.5)
        return out * 1e5  # output unit [nF is cm-2]

    def smartVlim_MottSchottky(self, Vlim=None, window=[-2, 2]):
        """
        Returns Vlim [Vmin, Vmax] offering a possible best range for Mott-
        Schottky fit.
        Assumes V in monotoneous increasing/decreasing.
        Window: how many points around best location are taken. Default [-2,2]
        """
        V = self.x()
        N = self.y(alter="CurveCV.y_CV_Napparent")
        # identify index within given V limits
        if Vlim is None:
            Vlim = CurveCV.CST_MottSchottky_Vlim_adaptative
        Vlim = [min(Vlim), max(Vlim)]
        ROI = [np.inf, -np.inf]
        for i in range(len(V)):
            if V[i] >= Vlim[0]:
                ROI[0] = min(i, ROI[0])
            if V[i] <= Vlim[1]:
                ROI[1] = max(i, ROI[1])
        # identify best: take few points around minimum of N
        N[N < 0] = np.inf
        from scipy.signal import medfilt

        N_ = medfilt(N, 3)  # thus we eliminate faulty points
        idx = np.argmin(N_[ROI[0] : ROI[1]])
        # print(self.attr('temperature [k]'), [V[ROI[0]+idx+window[0]], V[ROI[0]+idx+window[1]]])
        lim0 = max(ROI[0] + idx + window[0], 0)
        lim1 = min(ROI[0] + idx + window[1], len(V) - 1)
        return [V[lim0], V[lim1]]

    def CurveCV_0V(self, Vtarget=0):
        """
        Creates a curve with require data to compute doping at V=0
        Parameters:
            Vtarget: extract doping around other voltage
        """
        if not is_number(Vtarget):
            print("CurveCV.CurveCV_0V: please provide a number")
            return False
        i = np.argmin(np.abs(self.x() - Vtarget))
        if i > 0 and i < len(self.x()) - 1:
            x = np.concatenate(([0], self.x()[i - 1 : i + 2], [0]))
            y = np.concatenate(([np.nan], self.y()[i - 1 : i + 2], [np.nan]))
            curve = CurveCV([x, y], self.attributes)
            curve.update(
                {
                    "linespec": "s",
                    "markeredgewidth": 0,
                    "labelhide": 1,
                    "label": curve.attr("label") + " V=" + str(Vtarget),
                    "_curvecv_0V": True,
                }
            )

            if len(curve.x()) > 2:
                msg = "Apparent space charge region width [nm] {}, Apparent carrier density (doping) [cm-3] {}"
                print(
                    msg.format(
                        curve.x_offsets(alter="CurveCV.x_CVdepth_nm")[2],
                        curve.y_offsets(alter="CurveCV.y_CV_Napparent")[2],
                    )
                )

            return curve
        return False

    # print help
    def printHelp(self):
        print("*** *** ***")
        print("CurveCV offer basic treatment of C-V curves of solar cells.")
        print("Input units must be [V] and [nF] (or [nF cm-2]).")
        print("Curve transforms:")
        print(" - Linear: standard is Capacitance per area [nF cm-2] vs [V].")
        print(
            " - Mott-Schottky: [1/(C cm-2)^2] vs V. Allows extraction of",
            "built-in voltage V_bi (intercept with y=0) and carrier density",
            "(from slope).",
        )
        print(
            " - Carrier density N_CV [cm-3] vs V: the apparent carrier",
            "density is calculated from formula:",
        )
        print("   N_CV = -2 / (q eps0 epsr d/dV(C^-2)).")
        print(
            " - Carrier density N_CV [cm-3] vs depth [nm]: the apparent",
            "depth is calculated from parallel plate capacitor:",
        )
        print("   C = eps0 epsr / d.")
        print("Analysis functions:")
        print(
            " - Set area: can normalize input data. For proper data analysis",
            "the units should be [nF cm-2].",
        )
        print(
            " - Extract N_CV, Vbi: fits the linear segment on the",
            "Mott-Schottky plot.",
        )
        print("   Select the suitable ROI before fitting (enter min and max voltages).")
        return True
