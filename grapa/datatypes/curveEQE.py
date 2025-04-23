# -*- coding: utf-8 -*-
"""Curve type to handle External quantum efficiency EQE data of solar cells

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import numpy as np
from scipy import interpolate
from scipy.signal import medfilt, savgol_filter
from scipy.optimize import curve_fit


from grapa.graph import Graph
from grapa.curve import Curve
from grapa.constants import CST
from grapa.mathModule import roundSignificant, roundSignificantRange, is_number, trapz

from grapa.utils.curve_subclasses_utils import FileLoaderOnDemand
from grapa.utils.funcgui import FuncListGUIHelper, FuncGUI, AlterListItem


class FileLoaderOnDemandAM0(FileLoaderOnDemand):
    """Modify FileLoaderOnDemand to provide proper units"""

    def _load_if_not_already(self):
        super()._load_if_not_already()
        # customization - make sure to instantiate with call_index=-1
        if self.content is not None:
            curve = self.content[0]
            energy = CST.h * CST.c / (1e-9 * curve.x())
            irrad = curve.y() / energy / 1000
            self.content.append(Curve([curve.x(), irrad], {}))


def bandgap_tauc(nm, eqe, ylim=None, xlim=None, mode="EQE"):
    """Performs fit of low-energy side of EQE and returns [bandgap, slope].
    Executed at initialization of CurveEQE.

    :param nm: x series
    :param eqe: y series
    :param ylim: by default, [0.25, 0.70]
    :param xlim: by default, [600, 1500]
    :param mode: 'EQE', or 'ln1-EQE'

    """
    if ylim is None:
        ylim = [0.25, 0.70]
    if xlim is None:
        xlim = [600, 1500]
    if max(eqe) > 10:
        print(
            "Function bandgap_tauc: max(EQE) > 1. Multiplied by 0.01 for "
            "datapoint selection."
        )
        eqe = eqe * 0.01
    # select suitable data range
    mask = np.ones(len(nm), dtype=bool)
    for i in reversed(range(len(mask))):
        if eqe[i] < ylim[0] or eqe[i] > ylim[1] or nm[i] < xlim[0] or nm[i] > xlim[1]:
            mask[i] = False
    nm = nm[mask]
    eqe = eqe[mask]
    # perform fit
    eV = CST.nm_eV / nm
    tauc = (eV * np.log(1 - eqe)) ** 2 if mode == "log1-EQE" else (eV * eqe) ** 2
    if len(tauc > 1):
        z = np.polyfit(eV, tauc, 1, full=True)[0]
        bandgap = -z[1] / z[0]  # p = np.poly1d(z)
        return [bandgap, z[0]]
    # print ('Function bandgapTauc: not enough suitable datapoints.')
    return [np.nan, np.nan]


class CurveEQE(Curve):
    """
    CurveEQE offer basic treatment of (external) quantum efficiency curves of
    solar cells.
    Input units should be in [nm] and values within [0-1]. If needed,
    CurveSpectrum can convert [eV] in [nm].
    """

    CURVE = "Curve EQE"

    AXISLABELS_X = {
        "": ["Wavelength", "\\lambda", "nm"],
        Curve.ALTER_NM_EV: ["Energy", "E", "eV"],
    }
    AXISLABELS_Y = {
        "": ["EQE", "", "%"],  # unit updated upon use
        Curve.ALTER_TAUC: ["Tauc", "", "(E * EQE)$^2$"],
        Curve.ALTER_TAUCLN1MINUSEQE: ["Tauc", "", "(E ln(1-EQE))$^2$"],
    }
    UNIT_LOOKUP_Y = {"": {"": ""}, "%": {"": "%"}}

    FORMAT_AUTOLABEL = ["${sample} ${cell}"]

    _PATHEQE = os.path.dirname(os.path.abspath(__file__))

    REF_AM15_FILE = "AM1-5_Ed2-2008.txt"
    REF_AM15 = FileLoaderOnDemand(os.path.join(_PATHEQE, REF_AM15_FILE), 1)
    REF_AM0_FILE = "AM0_2000_ASTM_E-490-00.txt"
    REF_AM0 = FileLoaderOnDemandAM0(os.path.join(_PATHEQE, REF_AM0_FILE), -1)

    # Note: Shockley-Queisser tabulated values below are for the case without reflector
    SHOCKLEYQUEISSER_FILE = "EQE_ShockleyQueisser.txt"
    SHOCKLEYQUEISSER = FileLoaderOnDemand(os.path.join(_PATHEQE, SHOCKLEYQUEISSER_FILE))

    REF_SPECTRA_FILE = "EQE_referenceSpectra.txt"
    REF_SPECTRA = FileLoaderOnDemand(os.path.join(_PATHEQE, REF_SPECTRA_FILE))

    # default settings for Savitzky-Golay filtering for bandgap by derivative method
    SG_WIDTH = 5
    SG_DEGREE = 2

    # Blackbody spectrum for Shockley-Queisser calculations: default geometrical factor
    BB_FG = 2

    def __init__(self, data, attributes, silent=False):
        # modify default label
        if "label" in attributes:
            if not isinstance(attributes["label"], str):
                attributes["label"] = str(attributes["label"])
            if "$" not in attributes["label"]:
                attributes["label"] = attributes["label"].replace("_", " ")
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        # detect possible empty lines at beginning (some versions of EQE files
        # have a different number of header lines)
        flag = True if len(data.shape) > 1 else False  # check if data loaded
        try:
            while flag:
                if data[0, 0] == 0:
                    data = data[:, 1:]
                else:
                    flag = False
        except IndexError:
            pass
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        # bandgap calculation
        popt = self.attr("_popt")
        if isinstance(popt, str) and popt == "":
            bandgap = [np.nan, np.nan]
            try:
                bandgap = bandgap_tauc(self.x(), self.y())
            except Exception:
                if not self.silent:
                    msg = "ERROR readDataFromFileEQE during Eg calculation {}, {}."
                    print(msg.format(self.attr("filename"), self.data))
            # also useful to register that software was *not* able to compute bandgap
            self.update({"bandgaptauc": bandgap[0], "bandgaptauc_slope": bandgap[1]})
        # for saved files further re-reading
        self.update({"Curve": CurveEQE.CURVE})

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
        # auto-label
        out += _funcListGUI_autolabel(self)
        # preparation
        muloffset = self.attr("muloffset", default=1)
        if isinstance(muloffset, list):
            muloffset = muloffset[1]
        try:
            muloffset = float(muloffset)
        except TypeError:
            muloffset = 1

        fitfunc = str(self.attr("_fitfunc"))
        popt = self.attr("_popt")
        popt_is_default = self.is_attr_value_default(popt)
        # Eg from derivative method: peak max, PV, sigma best guess
        if not popt_is_default and fitfunc == "func_gaussian_a_ev":
            out += _funcListGUI_updatefit(self)
            out += _funcListGUI_fit_resamplex(self)
        elif popt_is_default:
            out += _funcListGUI_derivativeanalysis(self)

        # tauc (E*EQE)**2
        if not popt_is_default and fitfunc == "func_bandgapTaucCurve":
            out.append(
                [
                    self.updateFitParam,
                    "Update fit",
                    ["Eg", "slope"],
                    roundSignificant(self.attr("_popt"), 5),
                ]
            )
        elif popt_is_default:
            ylim = [0.25 * muloffset, 0.70 * muloffset]
            out.append(
                [
                    self.CurveEQE_bandgapTauc,
                    "Bandgap fit Tauc (eV*EQE)^2",
                    ["Range EQE"],
                    [ylim],
                    {"silent": False},
                ]
            )

        # tauc (E*ln(1-EQE))**2
        if not popt_is_default and fitfunc == "func_bandgapTaucCurveLog":
            out.append(
                [
                    self.updateFitParam,
                    "Update fit",
                    ["Eg", "slope"],
                    roundSignificant(self.attr("_popt"), 5),
                ]
            )
        elif popt_is_default:
            muloffset_suf = " [%]" if muloffset == 100 else ""
            out.append(
                [
                    self.CurveEQE_bandgapLog,
                    "Bandgap fit Tauc (eV*ln(1-EQE))^2",
                    ["Range EQE values" + muloffset_suf],
                    ["[" + str(0.30 * muloffset) + ", " + str(0.88 * muloffset) + "]"],
                    {"silent": False},
                ]
            )

        # EQE current
        if popt_is_default:
            out += _funclistgui_currentcalc(self)

        # Voc short-circuit loss term (due to Jsc smaller than Jscsq)
        label0 = "\u0394" + "Voc^sc (Jsc < Jsc_SQ). Bandgap eV"
        li = FuncGUI(self.vocloss_shortcircuit, "Calculate", hiddenvars={})
        li.append(label0, "Auto", options={"width": 5})
        li.append("Jsc mA/cm2", "Auto", options={"width": 5})
        li.append("T K", "STC", options={"width": 5})
        out.append(li)

        # low energy exponential tail (proxy to Urbach despite at too high energy)
        if popt_is_default:
            out += _funclistgui_expdecay(self)

        # ERE external radiative efficiency
        if popt_is_default:
            Emin, unit = self._ERE_emin_auto(self.x(), self.y())
            fgvals = ["2 - w/o reflector", "1 - w/ reflector"]
            li = FuncGUI(self.ERE_GUI, "ERE & Voc loss analysis")
            li.append("cell Voc (V)", "0.700", options={"width": 6})
            li.append("T", CST.STC_T, options={"width": 7})
            li.append("cut data", Emin, options={"width": 7})
            li.appendcbb("", unit, ["nm", "eV"], options={"width": 4})
            li.append_newline()
            li.appendcbb("          fg", self.BB_FG, fgvals, options={"width": 13})
            li.append("bandgap (eV)", "auto", options={"width": 7})
            out.append(li)

        # Absorption edge (CdS, etc.)
        if popt_is_default:
            li = FuncGUI(self.CurveEQE_absorptionEdge, "Quick & dirty CdS estimate")
            li.append("thickness [nm]", 50, options={"width": 6})
            li.append("I0", 1.0, options={"width": 7})
            li.append("material", "CdS", options={"width": 7})
            out.append(li)
        if not popt_is_default and fitfunc == "func_absorptionedge":
            li = FuncGUI(self.updateFitParam, "Update fit")
            li.append("thickness [nm]", popt[0])
            li.append("I0", popt[1])
            li.append("material", popt[2])
            out.append(li)

        # reference spectra - 20.4%, 20.8%, etc.
        lbls = self.REF_SPECTRA.labels()
        li = FuncGUI(CurveEQE.REF_SPECTRA.get, "Add")
        li.set_funcdocstring_alt("Return: chosen reference EQE spectrum.")
        li.appendcbb("reference spectrum:", lbls[0], lbls, options={"width": 25})
        out.append(li)

        # Help button!
        out.append([self.print_help, "Help!", [], []])
        # graph label
        out += _funclistgui_graph_axislabels(self, **kwargs)

        self._funclistgui_memorize(out)
        return out

    def alterListGUI(self):
        out = super().alterListGUI()
        # nm <-> eV
        doc = "switches the horizontal axis from nm to eV."
        item = AlterListItem("nm <-> eV", [self.ALTER_NM_EV, ""], "", doc)
        out.append(item)
        # Tauc
        doc = "Displays (eV*EQE)^2 vs eV. The Tauc fit shows as a straight line."
        item = AlterListItem("Tauc plot", [self.ALTER_NM_EV, self.ALTER_TAUC], "", doc)
        out.append(item)
        # only if tauc-ln
        fitfunc = str(self.attr("_fitfunc"))
        if self.has_attr("_popt") and fitfunc == "func_bandgapTaucCurveLog":
            item = AlterListItem(
                "Tauc plot (E ln(1-EQE)^2)",
                [self.ALTER_NM_EV, self.ALTER_TAUCLN1MINUSEQE],
                "",
                "only appears if a Curve was fitted with the corresponding formula.",
            )
            out.append(item)
        return out

    def updateFitParamFormatPopt(self, f, param):
        """override, for func_absorptionedge"""
        if f == "func_absorptionedge":
            return list(param)  # not np.array: last parameter can be str
        return Curve.updateFitParamFormatPopt(self, f, param)

    def updateFitParam(self, *args, **kwargs):
        """Override. Additional behavior when changing gaussian fit to derivative"""
        out = super().updateFitParam(*args, **kwargs)
        if self.attr("_fitfunc") == "func_gaussian_a_ev":
            self.print_popt_func_gaussian_a_ev(self.attr("_popt"), "Bandgap")
        return out

    # UTILITY FUNCTIONS
    @classmethod
    def reference_curve(cls, file: str):
        """
        Returns a reference spectrum (spectral photon irradiance) based on the provided
        file name.

        :param file: (str) The name of the file containing the reference spectrum data.
               The file should be located in the same directory as the script.
        :returns: A Curve object containing the reference spectrum data, or
                  None if file not found or an error occurs while loading the file
        """
        if file.endswith(".txt"):
            file = file[:-4]
        if file == cls.REF_AM15_FILE[:-4]:
            return cls.REF_AM15()
        if file == cls.REF_AM0_FILE[:-4]:
            return cls.REF_AM0()
        # maybe a custom input ?
        try:
            return FileLoaderOnDemand(os.path.join(CurveEQE._PATHEQE, file))[0]
        except IndexError:
            pass
        print("ERROR CurveEQE._curve_reference: cannot find reference file", file)
        return None

    @staticmethod
    def _selectdata_peak_threshold(xdata, ydata, threshold_rel=0.5):
        """
        Returns a reduced data range comprising down to half max value.
        Pads the array with interpolated values
        """

        def cut(x, y, thres):
            # cuts data after reaching threshold
            maxi = np.argmax(y)
            threshold = y[maxi] * thres
            for i in range(maxi + 1, len(y)):
                if y[i] < threshold:
                    dx, dy = x[i] - x[i - 1], y[i] - y[i - 1]
                    frac = (threshold - y[i - 1]) / dy
                    x_, y_ = None, None
                    if frac > 1e-10:  # rather not add in round value.
                        y_ = y[i - 1] + frac * dy
                        x_ = x[i - 1] + frac * dx
                    for _ in range(i, len(y)):
                        x.pop()
                        y.pop()
                    if x_ is not None:
                        x.append(x_)
                        y.append(y_)
                    return

        xs, ys = list(xdata), list(ydata)  # work on a copy
        cut(xs, ys, threshold_rel)
        xs, ys = xs[::-1], ys[::-1]
        cut(xs, ys, threshold_rel)
        xs, ys = xs[::-1], ys[::-1]
        return np.array(xs), np.array(ys)

    # FIT FUNCTIONS
    @staticmethod
    def func_bandgapTaucCurve(nm, *bandgap):
        z = [bandgap[1], -bandgap[0] * bandgap[1]]
        p = np.poly1d(z)
        fit = p(CST.nm_eV / nm)
        for i in range(len(fit)):
            if fit[i] >= 0:
                fit[i] = np.sqrt(fit[i]) / (CST.nm_eV / nm[i])
            else:
                fit[i] = np.nan
        return fit

    @staticmethod
    def func_bandgapTaucCurveLog(nm, *bandgap):
        z = [bandgap[1], -bandgap[0] * bandgap[1]]
        p = np.poly1d(z)
        fit = p(CST.nm_eV / nm)
        for i in range(len(fit)):
            if fit[i] >= 0:
                fit[i] = 1 - np.exp(-np.sqrt(fit[i]) / (CST.nm_eV / nm[i]))
            else:
                fit[i] = np.nan
        return fit

    @staticmethod
    def func_gaussian_a(x, a, x0, sigma, *_args):
        """A simple gaussian formula to fit the derivative peak
        The amplitude parameter is peak value i.e. not the function integral
        _args will be ignored. Because GUI may send more than required parameters
        """
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    @staticmethod
    def func_gaussian_a_ev(xnm, a, x0ev, sigmaev, *_args):
        """A simple gaussian formula to fit the derivative peak.
        Amplitude parameter is peak value i.e. not the function integral
        args: will be ignored. Because GUI may send more than required parameters.
        """
        xev = CST.nm_eV / xnm  # x in eV, not in nm
        return a * np.exp(-((xev - x0ev) ** 2) / (2 * sigmaev**2))

    def func_absorptionedge(self, nm, thickness, intensity0, material="CdS"):
        try:  # check parameters are numeric
            thickness, intensity0 = float(thickness), float(intensity0)
        except ValueError:
            msg = (
                "CurveEQE.func_absorptionedge: Did you really enter thickness and "
                "I0 as float?"
            )
            print(msg)
            return False
        k = self._materialrefractivek(material)
        if k is None:
            return False
        alpha = 4 * np.pi * k.y() / (k.x() * 1e-7)  # in cm-1
        transmitted = intensity0 * np.exp(-alpha * thickness * 1e-7)
        f = interpolate.interp1d(k.x(), transmitted, kind="linear", bounds_error=False)
        return f(nm)

    def fit_resamplex(self, deltax):
        new = ""
        if isinstance(deltax, float):
            deltax = np.abs(deltax)
            new = np.arange(self.x(0), self.x(-1) + deltax, deltax)
        if isinstance(deltax, list):
            if len(deltax) == 3:
                new = np.arange(deltax[0], deltax[1], deltax[2])
            else:
                new = np.array(deltax)
        # if valid input, do something
        if len(new) > 0 or new != "":
            # recreate data container, with new x values both in x and y
            # positions. next step it to compute new y values
            self.data = np.array([new, new])
            self.updateFitParam(*self.attr("_popt"))
            return 1
        return "Invalid input."

    # FUNCTIONS RELATED TO FIT
    # Bandgap Tauc
    def CurveEQE_bandgapTauc(self, ylim=None, silent=True):
        """Performs fit of the low-energy side of EQE with Tauc formula (linear relation
         (eV * EQE)**2 vs eV)

        :param ylim: Relative fractions of EQE values to consider for the fit.
               By default, [0.25, 0.70].
        :param silent: if False, prints additional information.
        :return: a Curve with fit parameters and fitted datapoints
        """
        if ylim is None:
            ylim = [0.25, 0.70]
        try:
            ylim = np.array(ylim) / self.attr("muloffset", default=1)
            bandgap = bandgap_tauc(self.x(), self.y(), ylim=ylim)
        except Exception:
            bandgap = None
        x = self.x()
        if len(x) < 2:
            print("CurveEQE_bandgap: Cannot calculate bandgap with less than 2 points")
            return False
        nm = np.arange(min(x), max(x), (max(x) - min(x)) / 1000)
        fit = self.func_bandgapTaucCurve(nm, *bandgap)
        attr = {
            "_popt": bandgap,
            "_fitFunc": "func_bandgapTaucCurve",
            "color": "k",
            "data filename": self.attr("filename"),
            "muloffset": self.attr("muloffset", 1),
        }
        out = CurveEQE([nm, fit], attr)
        self.update({"bandgaptauc": bandgap[0]})
        if not silent:
            msg = "Bandgap (Tauc (E*EQE)^2): {} eV"
            print(msg.format(roundSignificant(out.attr("_popt")[0], 4)))
        return out

    # Bandgap Log
    def CurveEQE_bandgapLog(self, ylim=None, silent=True):
        """Performs fit of the low-energy side of EQE with modified Tauc formula
        (linear relation in (eV * ln(1 - EQE))^2 vs eV.
        In principle more exact, however highly sensitive to reflective and collection
        losses, therefore less robust than (eV * EQE)^2.

        :param ylim: Relative fractions of EQE values to consider for the fit. By
               default, [0.25, 0.70].
        :param silent: if False, prints additional information.
        :return: a Curve with fit parameters and fitted datapoints
        """
        if ylim is None:
            ylim = [0.25, 0.70]
        try:
            ylim = np.array(ylim) / self.attr("muloffset", default=1)
            bandgap = bandgap_tauc(self.x(), self.y(), ylim=ylim, mode="log1-EQE")
        except Exception:
            bandgap = None
        x = self.x()
        nm = np.arange(min(x), max(x), (max(x) - min(x)) / 1000)
        fit = self.func_bandgapTaucCurveLog(nm, *bandgap)
        attr = {
            "_popt": bandgap,
            "_fitFunc": "func_bandgapTaucCurveLog",
            "color": "k",
            "data filename": self.attr("filename"),
            "muloffset": self.attr("muloffset", default=1),
        }
        out = CurveEQE([nm, fit], attr)
        self.update({"bandgaplog": bandgap[0]})
        if not silent:
            msg = "Bandgap (Tauc (E*ln(1-EQE))^2): {} eV"
            print(msg.format(roundSignificant(out.attr("_popt")[0], 4)))
        return out

    # bandgap derivative max, PV ans best guess sigma
    def derivative_savgol(self, sgwidth=None, sgdegree=None, roi=None):
        """
        Returns a reduced x,y range and the Savitzky–Golay derivative
        sgwidth, sgdegree: width and degree of Savitzky–Golay
        roi: range of interest [nmmin, nmmax]
        """
        if sgwidth is None:
            sgwidth = self.SG_WIDTH
        if sgdegree is None:
            sgdegree = self.SG_DEGREE
        sgdegree = int(sgdegree)
        sgwidth = int(sgwidth)
        if sgdegree >= sgwidth:
            msg = (
                "Curve EQE bandgap derivative: Savitzky–Golay Degree must be lower "
                "than width"
            )
            print(msg)
            return None, None, None
        if sgwidth % 2 == 0:
            msg = "Curve EQE bandgap derivative: Savitzky–Golay Width must be odd."
            print(msg)
            return None, None, None
        # compute derivative
        x, y = self.x(), self.y()
        if len(x) < 5:
            print("Not enough datapoints, cannot continue with processing.")
            return None, None, None
        # determines if sorted asc. or desc. Do NOT check if not monotonous
        order = True if x[1] > x[0] else False
        # data selection - up to EQE maximum
        imax = np.argmax(y)
        if roi is None or isinstance(roi, str):
            if order:
                roi_ = range(max(0, imax - 2), len(y))
            else:
                roi_ = range(0, min(imax + 2, len(y)))
        else:
            roi_ = (x >= min(roi)) * (x <= max(roi))
        x_, y_, eV = x[roi_], y[roi_], CST.nm_eV / x[roi_]
        # locate peak using Savitzky–Golay filtering; then correct for possible
        # uneven data point spacing
        savgol = savgol_filter(y_, sgwidth, sgdegree, deriv=1)
        savgol /= np.append(
            np.append(eV[1] - eV[0], (eV[2:] - eV[:-2]) / 2), eV[-1] - eV[-2]
        )
        return x_, y_, savgol

    def derivative_bandgapmax(
        self, sgwidth=None, sgdegree=None, x_y_savgol=None, roi=None, silent=True
    ):
        """
        Computes the bandgap based on the Savitzky–Golay derivative peak of derivative.
        The peak is located based on a gaussian fit to few points near peak values.

        :param sgwidth: 5. Parameters of Savitzky–Golay filtering
        :param sgdegree: 3
        :param x_y_savgol: (optional) to avoid calculating twice if already computed
               somewhere else
        :param roi: [nmmin, nmmax]
        :param silent: if False, prints additional information
        :return: fit results popt, x in fit range
        """
        # need dummy x argument to use the updateFitParam method
        if x_y_savgol is None:
            x_, _, savgol = self.derivative_savgol(sgwidth, sgdegree, roi=roi)
        else:
            x_, _, savgol = x_y_savgol

        # Choose ROI for derivative fitting
        order = True if x_[1] > x_[0] else False
        # locate max, fit gaussian around location of max
        maxi = np.argmax(savgol)
        maxv = savgol[maxi]
        roifit = [maxi - 1, maxi + 2]
        testv = [maxv * 0.75, maxv * 0.5]
        if not order:
            testv = testv[::-1]
        while True:
            if roifit[0] <= 0:
                break
            i = roifit[0] - 1
            if savgol[i] < testv[0]:
                break
            roifit[0] -= 1
        while True:
            if roifit[1] >= len(savgol) - 1:
                break
            i = roifit[1] + 1
            if savgol[i] < testv[1]:
                break
            roifit[1] += 1
        roifitr = range(*roifit)
        datax, datay = x_[roifitr], savgol[roifitr]
        p0 = [
            savgol[maxi],
            CST.nm_eV / x_[maxi],
            np.abs(2 * (CST.nm_eV / datax[1] - CST.nm_eV / datax[0])),
        ]

        try:
            popt, _pcov = curve_fit(self.func_gaussian_a_ev, datax, datay, p0=p0)
        except Exception as e:
            msg = (
                "Exception CurveEQE.bandgapDeriv. Provided to curve_fit length x {"
                "}, length y {}, p0 {}."
            )
            print(msg.format(len(datax), len(datay), p0))
            print("datax", datax)
            print(type(e), e)
            popt = [np.nan, np.nan, np.nan]
        # store results in main curve
        att = {
            "bandgapDerivativemax": popt[1],
            # "bandgapDerivativemax_savgol": [sgwidth, sgdegree],
        }
        self.update(att)
        if not silent:
            self.print_popt_func_gaussian_a_ev(popt, "Bandgap derivative max")
        return popt, [x_[r] for r in roifit]

    def derivative_bandgappvrau(
        self, sgwidth=None, sgdegree=None, x_y_savgol=None, roi=None, silent=True
    ):
        """
        PV bandgap based on Rau et al. PRA 2017 (weighted energy over peak FWHM)
        https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.7.044016
        Assumes input energy data actually given in nm
        Returns: bandgap value (float), x, y (base for calculation)
        """
        # need dummy x argument to use the updateFitParam method
        if x_y_savgol is None:
            x_, _, savgol = self.derivative_savgol(sgwidth, sgdegree, roi=roi)
        else:
            x_, _, savgol = x_y_savgol
        # calculation
        xegpv, yegpv = self._selectdata_peak_threshold(x_, savgol, 0.5)
        xegpv_ev = CST.nm_eV / xegpv
        top = trapz(xegpv_ev * yegpv, xegpv_ev)
        bottom = trapz(yegpv, xegpv_ev)
        bandgappv = top / bottom
        self.update({"bandgappv": bandgappv})
        if not silent:
            print("Bandgap PV (Rau): {} eV".format(roundSignificant(bandgappv, 4)))
        return bandgappv, xegpv, yegpv

    def derivative_sigmabestguess(
        self, sgwidth=None, sgdegree=None, x_y_savgol=None, roi=None, silent=True
    ):
        if x_y_savgol is None:
            x_, _, savgol = self.derivative_savgol(sgwidth, sgdegree, roi=roi)
        else:
            x_, _, savgol = x_y_savgol
        # calculation
        xegpv, yegpv = self._selectdata_peak_threshold(x_, savgol, 0.333)
        xegpv_ev = CST.nm_eV / xegpv
        spanx = np.max(xegpv_ev) - np.min(xegpv_ev)
        p0 = [np.max(yegpv), np.average(xegpv_ev), 0.1 * spanx]  # maybe: bandgappv
        # bounds = ((-np.inf, bandgappv, -np.inf), (np.inf, bandgappv + 1e-10 * bandgappv, np.inf))
        popt, _pcov = curve_fit(self.func_gaussian_a_ev, xegpv, yegpv, p0=p0)
        sigma = popt[2]
        deltavocrad = self.deltavocrad_sigma(popt[2])
        # storing in self: not needed, there are analysis results compiled
        # self.update({"derivative best guess sigma (eV)": sigma})
        # self.update({"derivative best guess deltavocrad (V)": deltavocrad})
        if not silent:
            msg = "Sigma best guess: {} eV (DeltaVoc_rad {} V)"
            print(msg.format(*roundSignificant([sigma, deltavocrad], 4)))
        return popt, xegpv, yegpv, deltavocrad

    def CurveEQE_derivativeanalysis(
        self, sgwidth: int = None, sgdegree: int = None, roi=None, silent: bool = True
    ) -> list:
        """
        Bandgap derivative: Computes the bandgap and other derived data based on the
        Savitzky–Golay derivative dEQE/dE. Are implemented:

        - Peak of derivative, based on a gaussian fit to few points around the peak,

        - PV bandgap based on Rau et al. PRA 2017 (weighted energy over peak FWHM)
          https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.7.044016

        - Best guess for sigma value, based on gaussian fit with wider fit range

        :param sgwidth: width of Savitzky–Golay filter. Default 5.
        :param sgdegree: degree of Savitzky–Golay filter. Default 3.
               Parameter values (w,d) = (3,1) fall back onto the symmetrical discrete
               difference.
        :param roi: range of interest. [nm_min, nm_max]
        :param silent: if False, prints additional information.
        :returns: a list of Curves: the derivative, as well as auxiliary Curves showing
                  the data on which the bandgap is computed. The computed bandgap values
                  can be retrieved in the last Curve attributes.
        """
        # create Savitzky–Golay derivative
        x_, y_, savgol = self.derivative_savgol(
            sgwidth=sgwidth, sgdegree=sgdegree, roi=roi
        )
        out = []
        label = self.attr("label")
        attr = {
            "color": "k",
            "data filename": self.attr("filename"),
            "muloffset": self.attr("muloffset", 1) * 0.1,
            "labelhide": 1,
        }
        kwsg = {
            "sgwidth": sgwidth,
            "sgdegree": sgdegree,
            "x_y_savgol": [x_, y_, savgol],
            "silent": silent,
        }

        # bandgap derivative max
        poptm, xr = self.derivative_bandgapmax(**kwsg)
        bandgapmax = poptm[1]  # CST.nm_eV /
        # curve derivative savgol
        out.append(Curve([x_, savgol], attr))
        att = {
            "_savgol": [sgwidth, sgdegree],
            "label": label + " derivative",
        }
        out[-1].update(att)
        # bandgap max fit peak curve
        fitx = np.linspace(xr[0], xr[-1], 50)
        fity = self.func_gaussian_a_ev(fitx, *poptm)
        out.append(CurveEQE([fitx, fity], attr))
        att = {
            "linespec": "--",
            "color": [0.5, 0.5, 0.5],
            "_popt": list(poptm),
            "_fitFunc": "func_gaussian_a_ev",
            "label": label + " bandgap derivative max",
        }
        out[-1].update(att)

        # Bandgap PV (Rau)
        bandgappv, xegpv, yegpv = self.derivative_bandgappvrau(**kwsg)
        # curve integration domain
        out.append(Curve([xegpv, yegpv], attr))
        att = {
            "color": [0, 0, 0, 0.05],
            "bandgappv_rau": bandgappv,
            "type": "fill",
            "fill_padto0": 1,
            "label": label + " bandgap PV (Rau) {:.4f} eV".format(bandgappv),
        }
        out[-1].update(att)

        # best guess sigma
        popts, xegpvs, _yegpvs, deltavocrad = self.derivative_sigmabestguess(**kwsg)
        # curve fit sigma best guess
        sigmabest = popts[2]
        fitx = np.linspace(np.min(xegpvs), np.max(xegpvs), 50)
        fity = self.func_gaussian_a_ev(fitx, *popts)
        out.append(CurveEQE([fitx, fity], attr))
        att = {
            "linewidth": 0.4,
            "color": [0.5, 0.5, 0.5],
            "_popt": list(popts),
            "_fitFunc": "func_gaussian_a_ev",
            "label": label + " sigma best guess",
        }
        out[-1].update(att)

        # Curve to easily import-export important parameters
        if roi is None or isinstance(roi, str):
            roi = [np.nan, np.nan]
        quantities = [
            [bandgapmax, "Bandgap derivative max / eV"],
            [bandgappv, "Bandgap PV [Rau] / eV"],
            [sigmabest, "Sigma best guess / eV"],
            [deltavocrad, "DeltaVoc^rad from sigma best guess / V"],
            [sgwidth, "Savitzki-Golay width"],
            [sgdegree, "Savitzki-Golay degree"],
            [roi[0], "ROI range of interest min / nm"],
            [roi[1], "ROI range of interest max / nm"],
        ]
        curve = self._analysis_output_as_curve(quantities, label, "Analysis derivative")
        out.append(curve)
        return out

    def _analysis_output_as_curve(self, quantities, label, propertytitle):
        # beware: modifies self
        qvalues = [float(item[0]) for item in quantities]
        qlabels = [item[1] for item in quantities]
        attr = {
            "quantities": qlabels,
            "label": label + " " + propertytitle,
        }
        for item in quantities:
            element = {"{}: {}".format(propertytitle, item[1]): item[0]}
            attr.update(element)
            self.update(element)
        keys = ["cell", "filename", "label_initial", "lockinfrequency"]
        keys += ["lockinreserve", "lockinsensitivity", "lockinslope", "lockinsync"]
        keys += ["lockintimeconstant", "referencefile", "sample"]
        for key in keys:
            attr.update({key: self.attr(key)})
        curve = Curve([range(len(qvalues)), qvalues], attr)
        curve.visible(False)
        return curve

    @classmethod
    def print_popt_func_gaussian_a_ev(cls, popt, text="Bandgap"):
        """formatted printout of bandgap, sigma, Vocrad loss based on _popt"""
        bandgapderiv = popt[1]
        sigma = popt[2]
        deltavocrad = cls.deltavocrad_sigma(popt[2])
        msg = "{}: {} eV (sigma {} eV, DeltaVoc_rad {} eV at STC temperature)"
        values = [roundSignificant(q, 4) for q in [bandgapderiv, sigma, deltavocrad]]
        print(msg.format(text, *values))

    @staticmethod
    def deltavocrad_sigma(sigma, T=CST.STC_T):
        # sigma in eV, T in Kelvin
        deltavocrad = (sigma * CST.q) ** 2 / (2 * CST.q * CST.kb * T)
        return deltavocrad

    # Current calc
    def currentCalc(
        self,
        roi=None,
        interpolatekind="linear",
        spectralPhotonIrrad=None,
        showintegratedcurrent=False,
        silent=False,
        if_gui=False,
    ):
        """
        Computes the current EQE current using AM1.5 spectrum (or another if selected).
        Assumes the EQE values are in range [0,1] and NOT [0,100].

        :param roi: [nm_min, nm_max]
        :param interpolatekind: order for interpolation of EQE data. default 'linear'.
        :param spectralPhotonIrrad: filename in folder datatypes. If None, uses AM1.5G.
        :param showintegratedcurrent: returns the integrated current versus wavelength.
        :param silent: If False, prints additional information.
        :param if_gui: True will print the value, False will return the value.
        """
        if spectralPhotonIrrad is None:
            spectralPhotonIrrad = CurveEQE.REF_AM15_FILE
        ref_irrad = self.reference_curve(spectralPhotonIrrad)
        if ref_irrad is None:
            msg = "Error CurveEQE currentCalc, cannot find reference spectrum {}."
            print(msg.format(spectralPhotonIrrad))
            return None
        ROIdef = [min(self.x()), max(self.x())]
        if roi is None:
            roi = ROIdef
        else:
            roi = [max(ROIdef[0], roi[0]), min(ROIdef[1], roi[1])]
        # localize range of interest
        ref_roi = (ref_irrad.x() >= roi[0]) * (ref_irrad.x() <= roi[1])
        ref_data_x = ref_irrad.x()[ref_roi]
        ref_data_y = ref_irrad.y()[ref_roi]
        # interpolate data on ref x sampling
        # -> implicitly assuming sampling with more datapoints in ref than in data
        f = interpolate.interp1d(self.x(), self.y(), kind=interpolatekind)
        interpData = f(ref_data_x)
        # compute final spectrum
        finalSpectrum = ref_data_y * interpData
        # integrate QE*spektrum (on chosen range)
        Jo = trapz(finalSpectrum, ref_data_x)
        eqe_current = Jo * CST.q / 10
        if not silent or if_gui:
            print("Curve", self.attr("label"), "EQE current:", eqe_current, "mA/cm2")
        # return curve with integrated current
        if showintegratedcurrent:
            cumsum = [0]
            for i in range(1, len(finalSpectrum)):
                cumsum.append(
                    cumsum[-1]
                    + (finalSpectrum[i] + finalSpectrum[i - 1])
                    / 2
                    * (ref_data_x[i] - ref_data_x[i - 1])
                )
            cumsum = np.array(cumsum) * CST.q / 10
            color = self.attr("color", "")
            if color == "":
                color = "k"
            curvecumsum = Curve([ref_data_x, cumsum], {"color": "k"})
            curvecumsum.update(
                {
                    "label": self.attr("label") + " cumulative EQE current",
                    "ax_twinx": True,
                    "color": color,
                }
            )
            return curvecumsum
        if if_gui:
            return True
        return eqe_current

    # ERE and Voc analysis
    def ERE(
        self,
        voc: float,
        T: float,
        Emin="auto",
        EminUnit="nm",
        fg=BB_FG,
        bandgap=None,
        silent=True,
    ):
        """
        ERE: Computes the External Radiative Efficiency from the EQE curve and the
        cell Voc. Ref: Green M. A., Prog. Photovolt: Res. Appl. 2012; 20:472–476
        The more complete calculation is also provided from Rau et al., Phys. Rev.
        Applied 7, 044016 (2017)
        https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.7.044016
        Returns ERE, and a Curve showing the integrant.

        :param voc: cell Voc voltage in [V]
        :param T: temperature in [K]
        :param Emin: min energy on which the integral is computed. Unit: nm or eV.
        :param EminUnit: 'eV' if Emin is given in eV, 'nm' otherwise. Default 'nm'
        :param fg: Ruhle's geometrical factor. 2 for initial Shockley-Queisser
               derivation, 1 for hemispherical emission (i.e. back reflector)
        :param bandgap: in eV. if not provided, determines bandgap by derivative method
        :param silent: if False, prints additional information
        :return: ERE, qe_led, and a list of Curves - the last one contains all values
        """
        # does not matter if EQE is [0,1] or [0,100], this is corrected by the
        # calculation of Jsc (beware the day Jsc is patched!)

        out = []
        label = str(self.attr("label"))
        nm, EQE = self.x(), self.y()
        # variables check
        if Emin == "auto":
            Emin, EminUnit = self._ERE_emin_auto(nm, EQE)
        if EminUnit != "eV":
            EminUnit = "nm"
        if Emin != "auto" and EminUnit == "nm":
            Emin = CST.nm_eV / Emin  # convert nm into eV
        if not is_number(fg):
            try:
                if isinstance(fg, str) and len(fg) > 0 and fg[0] in ["1", "2"]:
                    fg = float(fg.split(" ")[0])
            except ValueError:
                pass
            if not is_number(fg):
                fg = 2
        fg = max(1, min(fg, 2))
        # retrieve data
        E = CST.nm_eV / nm * CST.q  # photon energy [J]
        jsc = self.currentCalc(silent=True) * 10  # [A m-2] instead of [mAcm-2]
        # mask for integration
        mask = (E >= Emin * CST.q) if Emin != "auto" else (E > 0)
        # compute the expression

        # # According to formalism of Green PIP 2012
        integrand = EQE * E**2 / (np.exp(E / (CST.kb * T)) - 1)  # [J2]
        integral = np.abs(trapz(integrand[mask], E[mask]))  # [J3]
        ERE = 2 * np.pi * CST.q / CST.h**3 / CST.c**2 / jsc  # [J-3]
        ERE *= np.exp(CST.q * voc / CST.kb / T)  # new term unit-less
        ERE *= integral  # output [unit-less]
        # lbl = ["ERE integrand to " + self.attr("label"), "", "eV$^2$"]
        # out = Curve(
        #     [nm[mask], integrand[mask] / (CST.q**2)],
        #     {"label": Graph().formatAxisLabel(lbl)},
        # )
        if not silent:
            msg = (
                "External radiative efficiency [Green]: {:.3E} (input Voc: {}, "
                "hemispherical emission (back reflector) ie. fg=1)"
            )
            print(msg.format(ERE, voc))

        # Do same thing and more, according to formalism of Rau et al. PRA 2017
        def func_phi_bb(energies, fg):
            # unit: J-1 m-2 s-1. Expression from Rau PRB 2007
            # 10.1103/PhysRevB.76.085303 CAUTION: added factor np.pi as compared to
            # Rau PRB 2007. This matches Green's as well as Ruhle calculations.
            return (
                fg
                * 2
                * np.pi
                * energies**2
                / (CST.h**3 * CST.c**2)
                / (np.exp(energies / (CST.kb * T)) - 1)
            )

        # Possibly: increase resolution for more exact integration results
        if E[1] < E[0]:
            E, EQE, mask = E[::-1], EQE[::-1], mask[::-1]  # increasing order for interp
        Etmp = np.arange(np.min(E), np.max(E), 0.001 * CST.q)
        EQE = np.exp(np.interp(Etmp, E, np.log(EQE)))
        mask = np.array(np.interp(Etmp, E, mask), dtype=bool)
        E = Etmp
        # Compute according to Rau et al. PRA 2017
        phi_bb = func_phi_bb(E, fg)
        integrand = EQE * phi_bb  # unit J-1 m-2 s-1
        j0rad = CST.q * np.abs(trapz(integrand[mask], E[mask]))
        # unit C s-1 m-2 = A m-2
        vocrad = CST.kb * T / CST.q * np.log(jsc / j0rad)
        deltavocnonrad = voc - vocrad
        qe_led = np.exp(deltavocnonrad / (CST.kb * T / CST.q))
        # turn around equation: deltavocnonrad = CST.kb * T / CST.q * np.ln(j0 / j0rad)
        j0 = j0rad * np.exp(-deltavocnonrad / (CST.kb * T / CST.q))
        lbl = [label + " q * EQE * $\\phi_{bb}$", "", "A m$^{-2}$ J$^{-1}$"]
        curve = Curve(
            [CST.q * CST.nm_eV / E[mask], CST.q * integrand[mask]],
            {"label": Graph().formatAxisLabel(lbl), "color": "k", "ax_twinx": 1},
        )
        out.append(curve)
        if not silent:
            msg = (
                "QE_LED [Rau]: {:.3E} (Vocrad {:.4f} eV, \u0394Vocnonrad {:.4f} eV, "
                "J0 {:.3E} A/m2. Input Voc {} V, Ruhle's fg={})"
            )
            print(msg.format(qe_led, vocrad, deltavocnonrad, j0, voc, fg))
        # out.append(Curve([CST.q * CST.nm_eV/E[mask], EQE[mask]], {"label":"EQE2"}))

        # estimate PL peak energy
        plx, ply = self._selectdata_peak_threshold(
            E[mask] / CST.q, CST.q * integrand[mask], 0.5
        )
        energyplpeak = trapz(plx * ply, plx) / trapz(ply, plx)

        # Follow-up: loss analysis
        if not is_number(bandgap):
            # popt, _ = self.derivative_bandgapmax()
            # bandgap = popt[1]  # in eV
            bandgap, _x, _y = self.derivative_bandgappvrau()  # in eV
        jscsqtable = CurveEQE.SHOCKLEYQUEISSER.get("Jsc [mA/cm2]")
        jscsq = np.interp(bandgap, jscsqtable.x(), jscsqtable.y()) * 10  # unit A/m2
        vocsqtable = CurveEQE.SHOCKLEYQUEISSER.get("Voc [V]")
        vocsqt = np.interp(bandgap, vocsqtable.x(), vocsqtable.y())
        E2 = np.arange(np.round(bandgap, 3), 5, 0.001) * CST.q  # 5 eV as ~infinity
        phi_bb_stepfunc = func_phi_bb(E2, fg)  # make sure energy range fully covered
        # stepfunc: not needed, by construction =1 over E2 range (and 0 outside)
        j0sq = CST.q * np.abs(trapz(phi_bb_stepfunc, E2))
        vocsq = CST.kb * T / CST.q * np.log(jscsq / j0sq)
        deltavocsc = CST.kb * T / CST.q * np.log(jsc / jscsq)  # jsc in A/m2
        deltavocrad = CST.kb * T / CST.q * np.log(j0sq / j0rad)
        voc_sc = vocsq + deltavocsc
        voc_rad = voc_sc + deltavocrad
        voc_nonrad = voc_rad + deltavocnonrad

        if not silent:
            warn = ""
            if np.abs(vocsqt - vocsq) > 0.005:
                warn = " Values differ, maybe fg != 2 ?"
            msg = (
                "   Using Eg deriv            {:.4f} eV\n"
                "   VocSQ calculated:         {:.4f} V (tabulated {:.4f} V){}\n"
                "   \u0394Voc_SC:     {:.4f} V -> {:.4f} V (as Jsc < JscSQ)\n"
                "   \u0394Voc_rad:    {:.4f} V -> {:.4f} V (accomodates Eg variations)."
                " For value from sigma, run 'Bandgap derivative & PV'\n"
                "   \u0394Voc_nonrad: {:.4f} V -> {:.4f} V (from EQE * blackbody "
                "integration) -> QE_LED {:.3E} (PLQY)"
            )
            values = [bandgap]
            values += [vocsq, vocsqt, warn]
            values += [deltavocsc, voc_sc]
            values += [deltavocrad, voc_rad]
            values += [deltavocnonrad, voc_nonrad, qe_led]
            print(msg.format(*values))

        # output of results as Curve, for easier reuse
        quantities = [
            [ERE, "ERE (hemispheric) / -"],
            [qe_led, "QE_LED (fg={}) [Rau] / -".format(fg)],
            [bandgap, "Bandgap / eV"],
            [vocsq, "Voc^SQ / eV"],
            [voc_sc, "Voc^sc / eV"],
            [voc_rad, "Voc^rad / eV"],
            [voc_nonrad, "Voc^nonrad / eV"],
            [deltavocsc, "DeltaVoc^sc / eV"],
            [deltavocrad, "DeltaVoc^rad / eV"],
            [deltavocnonrad, "DeltaVoc^nonrad / eV"],
            [j0sq, "J0^SQ / A m-2"],
            [j0rad, "J0^rad / A m-2"],
            [j0, "J0 / A m-2"],
            [energyplpeak, "PL peak energy / eV"],
            [energyplpeak - bandgap, "PL peak shift to bandgap / eV"],
            [vocsqt, "Voc^SQ tabulated (fg=2)"],
            [voc, "Voc / V"],
            [T, "Temperature / K"],
            [Emin, "Data cut energy or wavelength / eV or nm"],
            [fg, "fg Ruhle geometrical factor / -"],
        ]
        curve = self._analysis_output_as_curve(quantities, label, "Analysis Voc")
        out.append(curve)

        # possible warning to user if data quality concern
        intergmask = integrand[mask]
        if intergmask[0] > intergmask[1] or intergmask[0] > 0.75 * np.max(intergmask):
            print(
                "WARNING Curve_EQE ERE: make sure to cut data before reaching the "
                "noise level!. NOTE: one must clearly distinguish a peak in the "
                "EQE*phi_bb curve, below the bandgap energy."
            )
        return ERE, qe_led, out

    def ERE_GUI(self, voc, T, Emin="auto", EminUnit="nm", fg=BB_FG, bandgap=None):
        """Computes the External Radiative Efficiency from the EQE curve and the cell
        Voc. Ref: Green M. A., Prog. Photovolt: Res. Appl. 2012; 20:472–476
        The more complete calculation is also provided from Rau et al., Phys. Rev.
        Applied 7, 044016 (2017)
        https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.7.044016

        :param voc: cell Voc voltage in [V]
        :param T: temperature in [K]
        :param Emin: min energy on which the integral is computed. Unit: nm or eV.
        :param EminUnit: 'eV' if Emin is given in eV, 'nm' otherwise. Default 'nm'
        :param fg: Ruhle's geometrical factor. 2 for initial Shockley-Queisser
               derivation, 1 for hemispherical emission (i.e. back reflector)
        :param bandgap: in eV. if not provided, determines PV bandgap by derivative.
        :return: a list of Curves - the last one contains all values
        """
        _ERE, _qe_led, curves = self.ERE(
            voc, T, Emin=Emin, EminUnit=EminUnit, fg=fg, bandgap=bandgap, silent=False
        )
        return curves

    @staticmethod
    def _ERE_emin_auto(nm, eqe):
        # smart Emin autodetect, to which value to consider eqe for * blackbody?
        try:
            # identify nm where EQE = 0.5
            nmmax = np.max(nm[(eqe > 0.5 * np.max(eqe))])
        except Exception:  # no suitable point
            return [0, "nm"]
        mask = (nm > nmmax) * (eqe > 0)
        nm_, eqelog = nm[mask], np.log10(eqe[mask])
        E = CST.nm_eV / nm_
        # sort nm & EQE in ascending nm order
        eqelog = eqelog[nm_.argsort()]
        nm_.sort()
        diff = (eqelog[1:] - eqelog[:-1]) / (E[1:] - E[:-1])  # d/dE log(EQE)
        nfault = 0
        Emin = np.max(nm) + 1  # by default, just below lowest point
        for i in range(1, len(diff)):
            if diff[i] < np.min(diff[:i]) * 0.5:  # 0.5 can be adjusted
                nfault += 1  # or another algorithm implemented
            if nfault > 1 or diff[i] < 0:
                Emin = roundSignificantRange([np.mean(nm_[i : i + 2]), nm_[i]], 2)[0]
        return [Emin, "nm"]

    # Short-circuit Voc loss
    def vocloss_shortcircuit(self, bandgap=None, jsc=None, T=CST.STC_T, silent=False):
        """
        Calculation of DeltaV_{OC}^{JSc}, short-circuit loss term to Voc, as Jsc < JscSQ
        According to equation inline, slightly after Eq 11 in paper Rau PRA 2017
        https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.7.044016
        - bandgap: unit eV. If not provided, computed from derivative method.
        - jsc: unit mA/cm2. If not provided, computed with default parameters.
        - T: unit K. if not provided, STC value (298.15 K)
        """
        if not is_number(T):
            T = CST.STC_T
        if not is_number(bandgap):
            popt, _ = self.derivative_bandgapmax()
            bandgap = popt[1]
        if not is_number(jsc):
            jsc = self.currentCalc(if_gui=False, silent=True)
        jscsqtable = CurveEQE.SHOCKLEYQUEISSER.get("Jsc [mA/cm2]")
        jscsq = np.interp(bandgap, jscsqtable.x(), jscsqtable.y())
        vocsqtable = CurveEQE.SHOCKLEYQUEISSER.get("Voc [V]")
        vocsq = np.interp(bandgap, vocsqtable.x(), vocsqtable.y())
        deltavocsc = CST.kb * T / CST.q * np.log(jscsq / jsc)  # unit of eV
        if not silent:
            msg = (
                "DeltaVoc_SC: {:.3f} eV (bandgap {:.3f} eV, Voc_SQ {:.3f} V, Jsc_SQ {"
                ":.2f} mA/cm2, Jsc {:.2f} mA/cm2)"
            )
            print(msg.format(deltavocsc, bandgap, vocsq, jscsq, jsc))
        return deltavocsc

    # Urbach (or better, exponential decay) energy
    def CurveUrbachEnergy(self, roi_ev: list):
        """
        Exponential decay: to obtain a measure of the sub-bandgap disorder tails.
        The values are typically fitted at too high energies to be the Urbach tail as
        can be characterized from high-sensitivity PL spectroscopy. Instead, one obtains
        slightly higher values, with a contribution from bandgap distributions.

        :param roi_ev: limits to the fitted data, in eV. [ev_min, eV_max]
        :return: the fit as a Curve, giving exponential decay (proxy to Urbach energy).
        """

        from grapa.datatypes.curveArrhenius import (
            CurveArrhenius,
            CurveArrheniusExpDecay,
        )

        roi_ev.sort()
        curve = CurveArrhenius(self.data, CurveArrheniusExpDecay.attr)
        out = curve.CurveArrhenius_fit(roi_ev)
        out.update(self.get_attributes(["offset", "muloffset"]))
        return out

    # Quick & dirty calculation Layer thickness estimate (e.g. CdS)
    @staticmethod
    def _materialrefractivek(material):
        """
        Returns imaginary part of refractive index of CdS, k
        Energy to be given in nm
        """
        path = os.path.dirname(os.path.abspath(__file__))
        matclean = "EQE_absorption_{}.txt".format(
            material.replace("/", "").replace("\\", "")
        )  # "EQE_absorption_" + material.replace("/", "").replace("\\", "") + ".txt"
        pathtest = os.path.join(path, matclean)
        if os.path.exists(pathtest):
            return Graph(pathtest)[0]
        try:
            # try open the file, and return first Curve assumed to be (nm, k)
            return Graph(material)[0]
        except Exception as e:
            msg = (
                "CurveEQE._materialrefractivek: please choose a material, or a file "
                "with (nm,k) data as 2-column. Input: {}."
            )
            print(msg.format(material))
            print("Exception", type(e), e)
        return None

    def CurveEQE_absorptionEdge(
        self, thickness: float, intensity0: float, material: str = "CdS"
    ):
        r"""CdS estimate: returns a Curve for quick-and-dirty estimate layer thickness,
        e.g. CdS. The Curve represents the light transmitted through the layer and is
        computed from Beer-Lambert law as intensity0 \* exp(-alpha \* thickness),
        with alpha computed from k imaginary refractive index of material.

        :param thickness: layer thickness, in nm
        :param intensity0: amount of light entering the layer. Default: 1
        :param material: data is provided for 'CdS'. A file path might be provided
               instead, with a file containing k data in a 2-column file (nm, k)
        :return: a Curve for crude estimation of the absorption in an optical layer.
        """

        values = self.func_absorptionedge(self.x(), thickness, intensity0, material)
        attr = {
            "_fitfunc": "func_absorptionedge",
            "_popt": [thickness, intensity0, material],
            "muloffset": self.attr("muloffset"),
        }
        return CurveEQE([self.x(), values], attr)


def _funclistgui_expdecay(curve):
    limits = [0.1, 8e-4]  # to get the fit range
    out = []
    try:  # Exponential (Urbach) fitting
        from grapa.datatypes.curveArrhenius import CurveArrheniusExpDecay
    except ImportError as e:
        msg = "WARNING CurveEQE: do not find", "grapa.datatypes.curveArrhenius."
        print(msg, type(e), e)
        return out

    x0 = 1
    try:
        x0 = 1 + np.where(list(reversed(curve.y() > limits[0])))[0][0]
    except IndexError:
        pass
    x1 = len(curve.x())
    try:
        x1 = 1 + np.where(list(reversed(curve.y() > limits[1])))[0][0]
    except IndexError:
        pass
    x0, x1 = curve.shape()[1] - x0, curve.shape()[1] - x1
    nm_ev = curve.ALTER_NM_EV
    roi = [0, 0]
    if len(curve.x()) > 0:
        roi = [curve.x(index=x0, alter=nm_ev), curve.x(index=x1, alter=nm_ev)]
    roi.sort()
    roi = roundSignificantRange(roi, 3)
    # muloffset = self.attr("muloffset", 1)
    line = FuncGUI(curve.CurveUrbachEnergy, CurveArrheniusExpDecay.BUTTONFitLabel())
    line.append(CurveArrheniusExpDecay.BUTTONFitROI(), roi)
    out.append(line)
    return out


def _funcListGUI_autolabel(curve):
    line = FuncGUI(curve.label_auto, "Auto label")
    line.appendcbb("template", curve.FORMAT_AUTOLABEL[0], curve.FORMAT_AUTOLABEL)
    return [line]


def _funcListGUI_updatefit(curve):
    values = roundSignificant(list(curve.attr("_popt")[:3]), 5)
    line = FuncGUI(curve.updateFitParam, "Update fit")
    line.append("A", values[0])
    line.append("x0 (eV)", values[1])
    line.append("sigma (eV)", values[2])
    return [line]


def _funcListGUI_fit_resamplex(curve):
    nm = curve.x()
    lst = roundSignificant([nm[0], nm[-1], (nm[1] - nm[0])], 5)
    line = FuncGUI(curve.fit_resamplex, "Resample nm")
    line.append("delta nm, or [900, 1100, 10]", lst)
    return [line]


def _funcListGUI_derivativeanalysis(curve):
    xdata = curve.x()
    try:
        tmpmed = medfilt(curve.y(), 5)
        ROI = [xdata[np.argmax(tmpmed)], np.max(xdata)]
    except Exception:
        if len(xdata) > 0:
            ROI = [np.min(xdata), np.max(xdata)]
        else:
            ROI = [0, 0]
    line = FuncGUI(curve.CurveEQE_derivativeanalysis, "Bandgap derivative & PV")
    line.append("Savitzky–Golay width", 5)
    line.append("degree", 2)
    line.append("nm range", roundSignificantRange(ROI, 3))
    line.set_hiddenvars({"silent": False})
    return [line]


def _funclistgui_graph_axislabels(curve, **kwargs):
    out = []
    try:
        unity = curve.attr("_units")[1]
        if unity == "" and curve.attr("muloffset", 1) == 100:
            unity = "%"
        lookup_unity = curve.UNIT_LOOKUP_Y[unity]
    except (TypeError, KeyError, IndexError):
        lookup_unity = None
    out += FuncListGUIHelper.graph_axislabels(curve, lookup_y=lookup_unity, **kwargs)
    return out


def _funclistgui_currentcalc(curve):
    fnames = [
        curve.REF_AM15_FILE.replace(".txt", ""),
        curve.REF_AM0_FILE.replace(".txt", ""),
    ]
    mM = [0, 0]
    if len(curve.x()) > 0:
        mM = [min(curve.x()), max(curve.x())]
    valsi = ["linear", "quadratic", "cubic"]
    optsic = {"field": "Checkbutton"}
    li = FuncGUI(curve.currentCalc, "EQE current")
    li.set_hiddenvars({"if_gui": True})
    li.append("ROI [nm]", mM, options={"width": 15})
    li.appendcbb("interpolation", "linear", valsi, options={"width": 8})
    li.append_newline()
    li.appendcbb("          spectrum", fnames[0], fnames, options={"width": 15})
    li.append("integrated current as curve", False, options=optsic)
    return [li]
