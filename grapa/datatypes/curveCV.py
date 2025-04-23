# -*- coding: utf-8 -*-
"""CurveCV provides functionalities to process C-V capacitance versus voltage
measurements of solar cells.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import warnings
import numpy as np

from grapa.curve import Curve
from grapa.mathModule import roundSignificant, derivative, is_number
from grapa.constants import CST
from grapa.utils.funcgui import FuncListGUIHelper, FuncGUI, AlterListItem


class CurveCV(Curve):
    """CurveCV offer basic treatment of C-V capacitance versus voltage curves of
    solar cells. Input units must be [V] and [nF] (or [nF cm-2])."""

    CURVE = "Curve CV"

    AXISLABELS_X = {
        "": ["Voltage", "V", "V"],
        "CurveCV.x_CVdepth_nm": ["Apparent depth", "", "nm"],
    }
    AXISLABELS_Y = {
        "": ["Capacitance", "C", "nF"],  # unit updated upon use
        "CurveCV.y_ym2": ["Mott-Schottky 1/C$^2$", "", "nF$^{-2}$"],  # updated upon use
        "CurveCV.y_CV_Napparent": ["Apparent carrier density", "N_{CV}", "cm$^{-3}$"],
    }
    UNIT_LOOKUP_Y = {
        "nF": {"": "nF", "CurveCV.y_ym2": "nF$^{-2}$"},  # explicit better than implicit
        "nF cm-2": {"": "nF cm$^{-2}$", "CurveCV.y_ym2": "nF$^{-2}$ cm$^4$"},
    }

    CST_epsilonR = 10  # relative permittivity

    CST_MottSchottky_Vlim_def = [0, 0.4]
    CST_MottSchottky_Vlim_adaptative = [-0.5, np.inf]

    FORMAT_AUTOLABEL = [
        "${sample} ${cell} ${temperature [k]:.0f} K",
        "${temperature [k]:.0f} K",
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
        line = FuncGUI(self.label_auto, "Auto label")
        line.appendcbb("template", self.FORMAT_AUTOLABEL[0], self.FORMAT_AUTOLABEL)
        out.append(line)
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
        line = FuncGUI(self.setEpsR, "Set epsilon r \u03F5\u1D63")
        line.append("default = 10", self.getEpsR())
        out.append(line)
        # help
        out.append([self.print_help, "Help!", [], []])
        # 1/C2 = 2 / q / Nc / Ks / eps0 / A2 (Vbi-V)
        try:
            lookup_unity = self.UNIT_LOOKUP_Y[self.attr("_units")[1]]
        except (TypeError, KeyError, IndexError):
            lookup_unity = None
        out += FuncListGUIHelper.graph_axislabels(self, lookup_y=lookup_unity, **kwargs)
        # finally memorize
        self._funclistgui_memorize(out)
        return out

    def alterListGUI(self):
        out = super().alterListGUI()
        # Mott-Schottky
        doc = (
            "To extract built-in voltage V_bi (intercept to y=0) and carrier "
            "density (from slope)"
        )
        item = AlterListItem(
            "Mott-Schottky (1/C^2 vs V)", ["", "CurveCV.y_ym2"], "", doc
        )
        out.append(item)
        # doping vs V
        label = "Carrier density N_CV [cm-3] vs V"
        doc = (
            "the apparent carrier density is calculated from formula:\n"
            "    N_CV = -2 / (q eps0 epsr d/dV(C^-2))."
        )
        item = AlterListItem(label, ["", "CurveCV.y_CV_Napparent"], "semilogy", doc)
        out.append(item)
        # doping versus depth
        item = AlterListItem(
            "Carrier density N_CV [cm-3] vs depth [nm]",
            ["CurveCV.x_CVdepth_nm", "CurveCV.y_CV_Napparent"],
            "semilogy",
            "the apparent depth is calculated from parallel plate capacitor:\n"
            "    C = eps0 epsr / d",
        )
        out.append(item)
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
            w = eps_r * CST.epsilon0 / (1e-5 * self.y(**kwargs)) * 1e9
        return w

    def y_CV_Napparent(self, xyValue=None, **kwargs):
        """apparent carrier density N_CV"""
        if xyValue is not None:
            return xyValue[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # C from [nF cm2] to [F m-2]
            dcm2_dv = derivative(self.x(**kwargs), (1e-5 * self.y(**kwargs)) ** (-2))
        eps_r = CurveCV.getEpsR(self)  # must be called on a Curve not CurveCV
        # output unit in [cm-3]
        N = -2 / (CST.q * eps_r * CST.epsilon0 * dcm2_dv) * 1e-6
        return N

    # FUNCTIONS RELATED TO GUI (fits, etc.)
    def setArea(self, value):
        """Normalizes the device area. Modifies the y capacitance data. This module is
        designed for input units as [nF cm-2].

        :param value: new value for device area, presumably in cm-2.
        """
        oldArea = self.getArea()
        self.update({"cell area (cm2)": value})
        self.setY(self.y() / value * oldArea)
        return True

    def getArea(self):
        """Return cell area, using dedicated keyword"""
        return self.attr("cell area (cm2)", 1)

    # handle custom-defined eps_r
    def getEpsR(self):
        if self.attr("epsR", 0) != 0:
            return self.attr("epsR")
        return CurveCV.CST_epsilonR

    def setEpsR(self, value):
        """Set the epsilon_r (relative permittivity) value of the material of interest.
        The value affects the different data transformation (apparent doping, apparent
        depth etc.)
        For CIGS, suitable values may be 10 to 13.

        :param value: the new value for relative permittivity
        """
        self.update({"epsR": value})

    # functions for fitting Mott-Schottky plot
    def CurveCV_fitVbiN(self, Vlim=None, silent=False):
        """Fits the linear segment on the Mott-Schottky plot.

        :param Vlim: [V_min, V_max] range of interest for the fit.
        :param silent: if False, prints additional information.
        :return: a Curve based on a fit on Mott-Schottky plot.
        """
        # Calls fit_MottSchottky to get parameters, then call func_MottSchottky
        # to generate data.
        v_bi, n_cv = self.fit_MottSchottky(Vlim=Vlim)
        fname = self.attr("filename").split("/")[-1].split("\\")[-1]
        attr = {
            "color": "k",
            "area": self.getArea(),
            "_popt": [v_bi, n_cv],
            "_fitFunc": "func_MottSchottky",
            "_Vlim": Vlim,
            "filename": "fit to " + fname,
        }
        if not silent:
            msg = "Fit Mott-Schottky plot: Vbi ={} V, apparent doping N_CV = {:1.4e}."
            print(msg.format(v_bi, n_cv))
        return CurveCV([self.x(), self.func_MottSchottky(self.x(), v_bi, n_cv)], attr)

    def CurveCV_fitVbiN_smart(self, Vrange=None, window=[-2, 2], silent=False):
        """Returns a Curve based on a fit on Mott-Schottky plot, after first
        guessing the best possible range for fifting (where N_CV is lowest).

        :param Vrange: voltage limits [V_min, V_max]
        :param window:
        :param silent: if False, prints additional information
        :return: a Curve object containing the Mott-Schottky fit
        """
        Vlim = self.smartVlim_MottSchottky(Vlim=Vrange, window=window)
        return self.CurveCV_fitVbiN(Vlim=Vlim, silent=silent)

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
        v_bi = -z[1] / z[0]
        n_cv = -2 / (CST.q * CurveCV.getEpsR(self) * CST.epsilon0) / z[0]
        return v_bi, n_cv * 1e-6  # N_CV in [cm-3]

    def func_MottSchottky(self, volts, v_bi, n_cv):
        """
        Computes C(V) which will appear linear on a Mott-Schottky plot.
        - v_bi: built-in voltage [V]
        - n_CV: apparent doping density [cm-3]
        Returns: C [nF cm-2]
        """
        out = volts * np.nan
        if np.isnan(v_bi) or n_cv < 0:
            return out
        mask = volts < v_bi
        c_m2 = (
            2
            / (CST.q * CurveCV.getEpsR(self) * CST.epsilon0 * (n_cv * 1e6))
            * (v_bi - volts)
        )
        out[mask] = c_m2[mask] ** (-0.5)
        return out * 1e5  # output unit [nF is cm-2]

    def smartVlim_MottSchottky(self, Vlim=None, window=[-2, 2]):
        """
        Returns Vlim [Vmin, Vmax] offering a possible best range for Mott-
        Schottky fit.
        Assumes V in monotoneous increasing/decreasing.
        Window: how many points around best location are taken. Default [-2,2]
        """
        volts = self.x()
        dopings = self.y(alter="CurveCV.y_CV_Napparent")
        # identify index within given V limits
        if Vlim is None:
            Vlim = CurveCV.CST_MottSchottky_Vlim_adaptative
        Vlim = [min(Vlim), max(Vlim)]
        roi = [np.inf, -np.inf]
        for i in range(len(volts)):
            if volts[i] >= Vlim[0]:
                roi[0] = min(i, roi[0])
            if volts[i] <= Vlim[1]:
                roi[1] = max(i, roi[1])
        # identify best: take few points around minimum of N
        dopings[dopings < 0] = np.inf
        from scipy.signal import medfilt

        dopings_median = medfilt(dopings, 3)  # thus we eliminate faulty points
        idx = np.argmin(dopings_median[roi[0] : roi[1]])
        lim0 = max(roi[0] + idx + window[0], 0)
        lim1 = min(roi[0] + idx + window[1], len(volts) - 1)
        return [volts[lim0], volts[lim1]]

    def CurveCV_0V(self, volt=0):
        """Creates a curve with require data to compute doping at volt=0 V

        :param volt: a voltage value different than 0 V
        """
        if not is_number(volt):
            print("CurveCV.CurveCV_0V: please provide a number")
            return False
        i = np.argmin(np.abs(self.x() - volt))
        if 0 < i < len(self.x()) - 1:
            x = np.concatenate(([0], self.x()[i - 1 : i + 2], [0]))
            y = np.concatenate(([np.nan], self.y()[i - 1 : i + 2], [np.nan]))
            curve = CurveCV([x, y], self.get_attributes())
            curve.update(
                {
                    "linespec": "s",
                    "markeredgewidth": 0,
                    "labelhide": 1,
                    "label": curve.attr("label") + " V=" + str(volt),
                    "_curvecv_0V": True,
                }
            )

            if len(curve.x()) > 2:
                msg = (
                    "Apparent space charge region width [nm] {:.5f}, Apparent "
                    "carrier density (doping) [cm-3] {:.5f}"
                )
                print(
                    msg.format(
                        curve.x_offsets(alter="CurveCV.x_CVdepth_nm")[2],
                        curve.y_offsets(alter="CurveCV.y_CV_Napparent")[2],
                    )
                )

            return curve
        return False
