# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

# NOTE 1: tried to reduce class complexity, outsourcing the complexity in functions
# NOTE 2: progressively adopting python naming convention... poor idea to mix I guess

from copy import deepcopy
import warnings
import tkinter as tk
from scipy.interpolate import splrep, splev
import numpy as np


from grapa.curve import Curve
from grapa.mathModule import (
    is_number,
    roundSignificant,
    roundSignificantRange,
    derivative,
    trapz,
    smooth,
    SMOOTH_WINDOW,
)
from grapa.utils.funcgui import FuncListGUIHelper, FuncGUI, AlterListItem


def bin_data(x, y, binning=4):
    """binning: how many points are merged."""
    if not is_number(binning) or binning < 1:
        msg = (
            "Warning CurveTRPL smoothBin: cannot interpret binning value (got {"
            "}, request int larger than 0.). Set 1."
        )
        print(msg.format(binning))
        binning = 1
    binning = int(binning)

    le = len(x)
    x_ = np.zeros(int(np.ceil(len(x) / binning)))
    y_ = np.zeros(int(np.ceil(len(x) / binning)))
    for i in range(len(x_)):
        x_[i] = np.average(x[i * binning : min(le, (i + 1) * binning)])
        y_[i] = np.average(y[i * binning : min(le, (i + 1) * binning)])
    return x_, y_


def differential_lifetime(time, signal):
    """returns the differential lifetime of a signal -d(ln(y))/dt"""
    lny = np.log(signal)
    tau = -derivative(lny, time)
    return tau


def find_onset(x, y):
    """define onset as average between 25% and 95% percentile, returns x value"""
    if len(y) == 0:
        return 0
    v25 = np.percentile(y, 25)
    v95 = np.percentile(y, 95)
    target = (v25 + v95) / 2
    iMax = np.argmax(y)
    for i in range(iMax, -1, -1):
        if y[i] < target:
            return x[i]
    return x[0]


def fit_exponentials(x, y, roi=None, fixed=None):
    # KNOWN BUG: fixed parameters do not work properly
    # fixed: has the desired length, to be done by calling function
    # fixed: e.g. [0, '', 10]
    # check for ROI
    mask, datax = _roi_to_mask(x, roi)
    datay = y[mask]
    # check for fixed params, construct p0
    dataxmax = np.max(datax)
    dataxmin = np.min(datax)
    datayargmax = np.argmax(datay)
    dataymax = datay[datayargmax]
    # try to find reasonable guess fo tau
    tau0max = dataxmax * 0.05  # max 5% of time span
    tau0min = np.abs((dataxmax - dataxmin) / len(datax)) * 10  # min 10 datapoints
    probeindex = min(len(datay) - 1, int(np.floor(len(datay) / 4)))
    proberatio = datay[probeindex] / dataymax
    probetau = (datax[datayargmax] - datax[probeindex]) / np.log(proberatio)
    if np.isnan(probetau):
        probetau = tau0max
    probetau /= 4  # likely curved -> shorter
    tau0 = max(tau0min, min(tau0max, probetau))
    p0default = [0, dataymax * 0.5, tau0]  # np.abs(dataxmax)) * 0.02]
    while len(p0default) < len(fixed):  # additional: tau*2, A/2
        p0default += [p0default[-2] / 4, p0default[-1] * 4]
    # fixed parameters: input value in initial guess, before recondition
    isFixed = []  # list of booleans
    for i in range(len(fixed)):
        isFixed.append(is_number(fixed[i]))
        if isFixed[i]:
            p0default[i] = fixed[i]
    # print('p0default', p0default)

    # p0default = [(p if p % 2 or p == 0 else np.log(p)) for p in p0default]

    def recondition(params):
        # all fitfunc parameters, including fixed variables
        p_ = params[0:2]
        # invert tau: +0.001 to -0.001 is dramatically different, +1e10 -1e10 is not
        p_.append(1 / params[2])
        for i in range(3, len(params), 2):
            p_.append(params[i])  # A value
            # tau value: ratios for numerical stability
            p_.append(1 / (params[i - 1] / params[i + 1]))
        return p_

    def decondition(params, ifFixed, fixed):
        # all fitfunc parameters, including fixed variables
        p_ = params[0:2]
        p_.append(1 / params[2])
        for i in range(3, len(params), 2):
            p_.append(params[i])  # A value
            p_.append(1 / (params[i - 1] / params[i + 1]))  # tau value
            if isFixed[i + 1]:
                p_[i + 1] = fixed[i + 1]
                # actual fixed value needed to decondition next variable values
        return p_

    def mergedvarfixed(params, isFixed, complementary):
        p = []
        j, k = 0, 0
        for i in range(len(isFixed)):
            if isFixed[i]:
                p.append(complementary[k])
                k += 1
            else:
                p.append(params[j])
                j += 1
        return p

    # recondition problem: tau are written differently
    p0default = recondition(p0default)
    # handle fixed and actual variables
    p0 = []
    isFixed = []  # list of booleans
    complementary = []
    for i in range(len(fixed)):
        isFixed.append(is_number(fixed[i]))
        if isFixed[i]:
            # fixed[i] may not == p0default[i] due to recondition
            complementary.append(fixed[i])
        else:
            p0.append(p0default[i])

    # custom fit function handling fixed and free fit parameters
    def func(datax, *p0):
        params = mergedvarfixed(p0, isFixed, complementary)
        # print('func a', params)
        # during decondition, need to fix again fixed values, e.g. tau1 changed by
        # fit algorithm but tau2 was fixed by used
        params = decondition(params, isFixed, fixed)
        # print('     b', params)
        return CurveTRPL.func_fitExp(datax, *params)

    # actual fitting
    # print('TRPL p0', p0)
    from scipy.optimize import curve_fit

    popt, _pcov = curve_fit(func, datax, datay, p0=p0)

    # construct output parameters including fixed ones
    params = mergedvarfixed(popt, isFixed, complementary)
    # back to standard parametrization
    params = decondition(params, isFixed, fixed)

    # sort output by ascending tau values
    if np.sum(isFixed[1:]) == 0:  # only if no fixed parameter
        pairs = [[params[i], params[i + 1]] for i in range(1, len(params), 2)]
        taus = [pair[1] for pair in pairs]
        pairs = [x for _, x in sorted(zip(taus, pairs))]
        p_ = [params[0]]
        for pair in pairs:
            p_ += pair
        params = p_

    return params


def spline(
    x, y, transform="semilogx", numknots=30, minspacingnpts=4, maxspacingrange=0.1
):
    """Fits the data with a spline and returns x_spline, y_spline.
    The generation of the knots can be adjusted with the optional parameters.

    :param x:
    :param y:
    :param transform: (str, optional): Transformation to apply to the data. Options
           are "semilogx", "semilogy", "loglog", "linlin", or "". Defaults to "semilogx"
    :param numknots: (int, optional): Target number of knots for spline
           interpolation. Defaults to 30.
    :param minspacingnpts: (int, optional): Minimum spacing between knots, in number
           of data points. Defaults to 4.
    :param maxspacingrange: (float, optional): Maximum spacing between knots, as a
           fraction of the x range. Defaults to 0.1.
    :return: x_new, y_new
    """
    if transform not in ["semilogx", "semilogy", "loglog", "linlin", ""]:
        transform = "semilogx"
    numknots = int(numknots)
    minspacingnpts = int(minspacingnpts)
    maxspacingrange = max(0, min(1, float(maxspacingrange)))
    # preparation to find suitable knots
    maxx = np.max(x)
    spaceknotsmin = (maxx - np.min(x)) / (len(x) - 1) * minspacingnpts
    spaceknotsmax = (maxx - np.min(x)) / min(1 / maxspacingrange, numknots)
    argmax = np.argmax(y)
    trial0 = list(np.arange(x[argmax], maxx, spaceknotsmin))[1:]  # lin spacing
    # transform data, if desired
    if transform in ["semilogx", "loglog"]:
        x = np.log(x)
        trial0 = list(np.log(trial0))
    if transform in ["semilogy", "loglog"]:
        y = np.log(y)
    # construct list of knots. try to make uniform on log space, but want both
    # min time delta (multiple of datapoint spacing) and
    # max time delta (not too long in lin scale)
    knots = [x[argmax]]
    trial1 = list(np.linspace(x[argmax], np.max(x), numknots))[1:]
    while True:
        if trial0[0] > trial1[0]:
            knots.append(trial0[0])
            trial0.pop(0)
            while len(trial1) > 0 and trial1[0] < knots[-1]:
                trial1.pop(0)
        else:
            knots += trial1
            break
    # cleaning up knots with excessive spacing in-between in lin scale
    for i in range(1, len(knots)):
        span = knots[i] - knots[i - 1]
        if transform in ["semilogx", "loglog"]:
            span = np.exp(knots[i]) - np.exp(knots[i - 1])
        if span > spaceknotsmax:
            knots = knots[:i]
            if transform in ["semilogx", "loglog"]:
                end = np.arange(np.exp(knots[-1]) + spaceknotsmax, maxx, spaceknotsmax)
                end = np.log(end)
            else:
                end = np.arange(knots[-1] + spaceknotsmax, maxx, spaceknotsmax)
            knots += list(end)
            break
    x_new = [x[0]] + knots + [x[-1]]
    knots = knots[:-1]  # remove last value, otherwise may sometimes fail
    # print("knots len", len(knots), knots)

    # Create a B-spline representation of the data
    try:
        tck = splrep(x, y, t=knots)
    except Exception as e:  # could be ValueError, or dfitpack.error
        print("CurveTRPL_spline: splrep failed, will try again with knots[1:-1].")
        # print("knot", knots)
        try:
            tck = splrep(x, y, t=knots[1:-1])
        except Exception as exc:
            print("Failed again ({}). knots: {}.".format(e, knots))
            raise e from exc
    # generate new y values
    y_new = splev(x_new, tck)

    # back to initial data transform
    if transform in ["semilogx", "loglog"]:
        x_new = np.exp(x_new)
    if transform in ["semilogy", "loglog"]:
        y_new = np.exp(y_new)
    return x_new, y_new


def integrate(curve, roi=None, alter=None):
    """Integrate: returns the integral of the curve, within ROI.

    :param curve: a Curve instance
    :param roi: example [xmin, xmax]
    :param alter: 'raw', or any Graph 'alter' value including (mul-)offsets.
    :param curve: if curve is None, act on self. Construct to apply code with
           Curves of a different type
    """
    # curve and not self: tweak to be able to integrate a Curve not CurveTRPL
    # (e.g. from GUI, with another Curve type such as CurveSpectrum)
    mask, _ = _roi_to_mask(curve.x(), roi)
    if alter is not None and alter not in ["raw"]:
        if isinstance(alter, str):
            alter = ["", alter]
        datax = curve.x_offsets(alter=alter[0])[mask]
        datay = curve.y_offsets(alter=alter[1])[mask]
    else:
        datax = curve.x()[mask]
        datay = curve.y()[mask]
    # actual calculation
    integral = trapz(datay, datax)
    return integral


class CurveTRPL(Curve):
    """CurveTRPL offers support to process and fit time-resolved photoluminence
    (TRPL) data."""

    CURVE = "Curve TRPL"

    AXISLABELS_X = {"": ["Time", "t", "ns"]}
    _YLABEL = ["Intensity", "", "counts"]
    AXISLABELS_Y = {"": _YLABEL, "idle": _YLABEL, "idle2": _YLABEL}

    AXISLABEL_DIFFLIFETIME = ["Differential lifetime", "\\tau", "ns"]

    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        self.update({"Curve": self.CURVE})

        # retrieve parameters required for normalization
        if self.attr("_repetfreq_Hz", None) is None:
            try:
                rep = float(self.attr("sync_frequency", None).split(" ")[0])
            except (ValueError, AttributeError):
                rep = 1
            self.update({"_repetfreq_Hz": rep})
        if self.attr("_acquistime_s", None) is None:
            time = 1
            try:
                time0 = float(self.attr("meas_time", "").split(" ")[0])
                time1 = 0
                try:
                    time1 = float(self.attr("meas_stop", "").split(" ")[0])
                except (ValueError, AttributeError):
                    pass
                time = np.abs(time1 - time0)
            except ValueError:
                pass
            if time == 0 or np.isnan(time):
                time = 1
            self.update({"_acquistime_s": time})
        if self.attr("_binwidth_s", None) is None:
            try:
                binw = float(self.attr("meas_binwidth", None).split(" ")[0]) * 1e-12
            except (ValueError, AttributeError):
                binw = 1
            self.update({"_binwidth_s": binw})

        # legacy attr "_unit"
        unit = str(self.attr("_unit"))
        if len(unit) > 0:
            self.update({"_unit": ""})
            self.data_units(unit_y=unit)

        # backward compatibility
        if (
            self.attr("_spectrumOffset", None) is not None
            and self.attr("_TRPLOffset", None) is None
        ):
            self.update({"_TRPLOffset": self.attr("_spectrumOffset")})

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = super().funcListGUI(**kwargs)
        out += _gui_funclist_trpl_xoffset(self)
        out += _gui_funclist_trpl_yoffset(self)
        out += _gui_funclist_trpl_normalize(self)
        out += _gui_funclist_trpl_fit(self)
        out += _gui_funclist_trpl_difflifetime(self)
        out += _gui_funclist_trpl_difflifetimevssignal(self)
        out += _gui_funclist_trpl_smoothbin(self)
        out += _gui_funclist_trpl_integration(self, **kwargs)
        out += _gui_funclist_trpl_printhelp(self)

        out += _funclistgui_graph_auto_axislabels(self, **kwargs)

        self._funclistgui_memorize(out)
        return out

    def alterListGUI(self):
        out = super().alterListGUI()
        doc = "semilogy, to visualize exponential decays"
        out.append(AlterListItem("semilogy", ["", "idle"], "semilogy", doc))
        doc = "log-log, to visualize power laws"
        out.append(AlterListItem("loglog", ["", "idle2"], "loglog", doc))
        return out

    # handling of offsets - same as Curve Spectrum, with same keyword
    def getOffset(self):
        return self.attr("_TRPLOffset", 0)

    def getFactor(self):
        return self.attr("_TRPLFactor", 1)

    def setIntensity(self, offsetnew=None, factornew=None):
        """Data stored as (raw + offset) * factor"""
        factorold = self.getFactor()
        cts = (self.y() - self.getOffset()) / factorold
        if is_number(offsetnew):
            self.update({"_TRPLOffset": offsetnew})
        if is_number(factornew):
            self.update({"_TRPLFactor": factornew})
            self.update({"_TRPLOffset": self.getOffset() * factornew / factorold})
        self.setY((cts * self.getFactor()) + self.getOffset())
        # backward compatibility
        if self.attr("_spectrumOffset", None) is not None:
            self.update({"_spectrumOffset": self.attr("_TRPLOffset")})
        return True

    def addOffset(self, value):
        """to adjust the background level. Data are modified. The previous adjustment is
        shown. The original data can be re-computed by setting the offset back to 0"""
        if is_number(value):
            self.setIntensity(offsetnew=value)
            return True
        return False

    def normalize(self, repetfreq_Hz, duration_s, binwidth_ps):
        """
        Normalizes intensity of TRPL to account for repetition rate,
        acquisition duration and binwidth. Assumes data is in unit 'counts'.
        Data: (raw+offset) * (1 / (syncfreq * (measstop-meastime) * binwidth))
        """
        try:
            factor = float(1 / (repetfreq_Hz * duration_s * (1e-12 * binwidth_ps)))
        except ValueError:
            return False
        if factor == 0 or np.isinf(factor):
            print("CurveTRPL.normlize: non-sensical normalization factor (0 or inf).")
            return False
        # should not happen if using only the GUI
        if self.data_units()[1] != "":
            msg = (
                "CurveTRPL.normalize: data may have been already normalized (Curve "
                "labelled as '{}', '{}')."
            )
            print(msg.format(self.data_units(), self.attr("_unitfactor")))
        self.setIntensity(factornew=factor)
        self.data_units(unit_y="cts/Hz/s/s")
        return True

    def normalizerevert(self, *_args):
        self.setIntensity(factornew=1)
        self.data_units(unit_y="")
        return True

    # temporal offset
    def getXOffset(self):
        return self.attr("_TRPLxOffset", 0)

    def addXOffset(self, value: float):
        """Adds a temporal offset, to place the peak onset at t=0. The curve data are
        modified. Adding a time offset is relevant as interpretation of fit intensity A
        values assumes the decay starts at t=0.

        :param value: (float). If value is not a number, the software tries to
               autodetect the leading edge, as the last point below a threshold defined
               as the average of the 25% and 95% percentiles.
        """
        if is_number(value):
            self.setX(self.x() + value - self.getXOffset())
            self.update({"_TRPLxOffset": value})
            return True
        return self.addXOffset(-self.findOnset())

    def findOnset(self):
        return find_onset(self.x(), self.y())

    def CurveTRPL_fitExp(
        self,
        nbExp=2,
        ROI=None,
        fixed=None,
        showResiduals=False,
        showfitonroi=False,
        silent=False,
    ):
        """
        Fts the data as a constant plus a sum of exponentials.
        Returns a Curve as the best fit to the TRPL decay.
        Formula: y(t) = BG + A1 exp(-t/tau1) + A2 exp(-t/tau2) + ...
        thus y(0) = BG + A1 + A2 + ...

        :param nbExp: The number of exponentials in the fit
        :param ROI: minimum and maximum on horizontal axis, e.g. [min, max]
        :param fixed: an array of values. When a number is set, the corresponding
               fit parameter is fixed. The order of fit parametersis as follows:
               BG, A1, tau1, A2, tau2, .... e.g. [0,'','']
        :param showResiduals: when 1, also returns a Curve as the fit residuals.
        :param showfitonroi: restrict the xrange of the fitted curve to the ROI
        :param silent: if False, prints additional information
        :return: a CurveTRPL instance
        """
        # set nb of exponentials, adjust variable "fixed" accordingly
        if fixed is None:
            fixed = [0, "", ""]
        while len(fixed) > 1 + 2 * nbExp:
            del fixed[-1]
        missing = int(1 + 2 * nbExp - len(fixed))
        if missing > 0:
            fixed += [""] * missing
        # clean input shworesiduals
        if isinstance(showResiduals, str):
            if showResiduals in ["True", "1"]:
                showResiduals = True
            else:
                showResiduals = False
        # fit
        popt = list(fit_exponentials(self.x(), self.y(), roi=ROI, fixed=fixed))
        # output
        attr = {
            "color": "k",
            "_ROI": ROI,
            "_popt": popt,
            "_fitFunc": "func_fitExp",
            "filename": "fit to "
            + self.attr("filename").split("/")[-1].split("\\")[-1],
            "label": "fit to " + self.attr("label"),
        }
        attr.update(self.get_attributes(["offset", "muloffset", "_repetfreq_Hz"]))
        attr.update(
            self.get_attributes(
                ["_acquistime_s", "_binwidth_s", "_units", "_trplfactor"]
            )
        )
        x = self.x()
        y = self.y()
        if not showfitonroi:
            mask, xmask = _roi_to_mask(x, [0, max(self.x())])
        else:
            mask, xmask = _roi_to_mask(x, ROI)
        fitted = CurveTRPL([xmask, self.func_fitExp(xmask, *popt)], attr)
        if showResiduals:
            mask, xmask = _roi_to_mask(x, ROI)
            attrresid = {"color": [0.5, 0.5, 0.5], "alpha": 0.5, "label": "Residuals"}
            attrresid.update(self.get_attributes(["offset", "muloffset"]))
            resid = CurveTRPL(
                [xmask, self.func_fitExp(xmask, *popt) - y[mask]], attrresid
            )
            fitted = [fitted, resid]
        if not silent:
            taus = ", ".join([str(popt[i]) for i in range(2, len(popt), 2)])
            print("Fitted with", int(nbExp), "exponentials: tau", taus, ".")
            print("Params\t", "\t".join([str(p) for p in popt]))
        return fitted

    @staticmethod
    def func_fitExp(t, bg, a1, tau1, *args):
        """
        computes the sum of a cst plus an arbitrary number of exponentials
        To keep within class, will be called as getattr(CurveTRPL, functionname)()
        """
        out = bg + a1 * np.exp(-t / tau1)
        i = 0
        while len(args) > i + 1:
            out += args[i] * np.exp(-t / args[i + 1])
            i += 2
        return out

    def CurveTRPL_sequence_fitexp(
        self, nb_exp=3, roi=None, roi0=None, multiplier=5, maxpoints=None
    ):
        """Fits the data with a sum of exponentials, in a piece-wise manner.

        :param nb_exp: (int, optional) Number of exponentials to fit. Defaults to 3.
        :param roi: (list, optional) Overall data range of interest, specified as
               [xmin, xmax]. Defaults to None.
        :param roi0: (list, optional) Initial data range of interest for the first fit,
               specified as [xmin, xmax]. Defaults to None.
        :param multiplier: (int, optional) Factor to multiply the initial roi0 to get
               the next one. Defaults to 5.
        :param maxpoints: (int, optional) Maximum number of points in the fit, to limit
               data size. Defaults to None.
        :return: (list) A list of CurveTRPL objects, each representing a piece-wise fit.
        """
        # sanitize input
        if maxpoints is not None:
            try:
                maxpoints = int(maxpoints)
            except ValueError:
                maxpoints = None
        if isinstance(roi, list):
            roi = list(roi)
        else:
            roi = [np.min(self.x()), np.max(self.x())]
        roiact = [1, 25]
        if isinstance(roi0, list):
            roiact = list(roi0)
        roiact[0] = max(roiact[0], roi[0])
        # run sequence
        flagend = False
        out = []
        nfitmax = 50
        while len(out) < nfitmax and not flagend:
            if roiact[1] >= roi[1]:
                flagend = True
            roiact[1] = min(roiact[1], roi[1])
            try:
                res = self.CurveTRPL_fitExp(
                    nbExp=nb_exp, ROI=roiact, showfitonroi=True, silent=True
                )
                if maxpoints is not None:
                    x = res.x()
                    if len(x) > maxpoints:
                        # reduce number of datapoints for fits
                        # otherwise may be absurdly high
                        xmin, xmax = min(x[0], x[-1]), max(x[0], x[-1])
                        step = np.abs((x[-1] - x[0]) / (maxpoints - 1))
                        res.fit_resampleX([xmin, xmax + step / 2, step])
                lbl = res.attr("label")
                if lbl.startswith("fit"):
                    lbl = "fit #{} {}".format(len(out), lbl[4:])
                    res.update({"label": lbl})
                out.append(res)
            except RuntimeError as e:
                msg = "Exception during fitting ROI {}, ignore and proceed."
                print(msg.format(roiact), type(e), e)
            roiact = [roiact[0] * multiplier, roiact[1] * multiplier]
        if len(out) == nfitmax:
            msg = (
                "CurveTRPL_sequence_fitexp: max number of fits reached ({}), stop here."
            )
            print(msg.format(nfitmax))
        return out

    def CurveTRPL_spline(
        self,
        roi=None,
        transform="semilogx",
        numknots=30,
        minspacingnpts=4,
        maxspacingrange=0.1,
    ):
        """Fits the data with a spline and returns a CurveTRPL object.

        :param roi: (list, optional): Region of interest for the fit, specified as
               [xmin, xmax]. Defaults to None.
        :param transform: (str, optional): Transformation to apply to the data. Options:
               "semilogx", "semilogy", "loglog", "linlin", or "". Defaults "semilogx".
        :param numknots: (int, optional): Target number of knots for spline
               interpolation. Defaults to 30.
        :param minspacingnpts: (int, optional): Minimum spacing between knots, in number
               of data points. Defaults to 4.
        :param maxspacingrange: (float, optional): Maximum spacing between knots, as a
               fraction of the x range. Defaults to 0.1.
        :return: A CurveTRPL object with the fitted spline data.
        """
        # selection of datapoints
        mask, x = _roi_to_mask(self.x(), roi)
        y = self.y()[mask]
        # actual heavy lifting
        x_new, y_new = spline(
            x, y, transform, numknots, minspacingnpts, maxspacingrange
        )
        # output
        lbl = "fit spline to {}".format(self.attr("label"))
        attr = {
            "color": "k",
            "label": lbl,
            "_fitfunc": "none",  # to prevent possibility to fit the output
            "_popt": [],  # to prevent possibility to fit the output
            "_splineroi": roi,
            "_splinetransform": transform,
            "_splinenumknots": numknots,
            "_splineminspacingnpts": minspacingnpts,
            "_splinemaxspacingrange": maxspacingrange,
        }
        attr_self = [
            "filename",
            "sample",
            "_acquistime_s",
            "_binwidth_s",
            "_repetfreq_Hz",
            "_trploffset",
            "_trplxoffset",
        ]
        for at in attr_self:
            attr.update({at: self.attr(at)})
        curve = CurveTRPL([x_new, y_new], attr)
        return curve

    def fit_resampleX(self, spacing):
        """
        This method modifies self.
        :param spacing: e.g. 3, or [xmin, xmax, xstep]
        """
        try:
            newx = _fit_resample_x_newx(self.x(), spacing)
        except RuntimeError as e:
            print("Error fit_resampleX: invalid input ({}): {}.".format(spacing, e))
            return False
        # if valid input, modfy self
        # recreate data container, with new x values both in x and y positions.
        # next step it to compute new y values
        self.data = np.array([newx, newx])
        self.updateFitParam(*self.attr("_popt"))
        return True

    def CurveTRPL_smoothBin(self, window_len=9, window="hanning", binning=4):
        """
        Returns a copy of the Curve after smoothening and data binning.

        :param window_len: (width) number of points in the smooth window
        :param window: (convolution widow) the type of window. Possible values:
               'hanning', 'hamming', 'bartlett', 'blackman', or 'flat' (moving average).
        :param binning: how many points are merged.
        """
        # smooth
        smt = smooth(self.y(), window_len, window)
        # bin
        x_, y_ = bin_data(self.x(), smt, binning=binning)
        # format results
        attr = deepcopy(self.get_attributes())
        comment_ = "Smoothed curve ({}) smt {} {} bin {}"
        comment = comment_.format(self.attr("label"), window_len, window, binning)
        attr.update({"comment": comment})
        attr.update({"label": str(self.attr("label")) + " " + "smooth"})
        return CurveTRPL([x_, y_], attr)

    def integrate(self, ROI=None, alter=None):
        """
        Returns the integral of the curve, within ROI.

        :param ROI: example [xmin, xmax]
        :param alter: 'raw', or any Graph 'alter' value including (mul-)offsets.
        """
        return integrate(self, roi=ROI, alter=alter)

    def Curve_differential_lifetime(self):
        """Returns a CurveTRPL object with differential (instantaneous) lifetime.
        Assuming y = A exp(- t / tau), formula is: tau(t) = - dt / d(ln(y))
        Make sure to remove background before calculating differential lifetime.
        """
        time = self.x()
        tau = differential_lifetime(time, self.y())
        # prepare output
        attr = deepcopy(self.get_attributes())
        comment = "Differential lifetime ({})".format(self.attr("label"))
        attr.update({"comment": comment})
        attr.update({"label": "{} Differential lifetime".format(self.attr("label"))})
        curve = Curve([time, tau], attr)
        curve.update(
            {
                "offset": "",
                "muloffset": "",
                Curve.KEY_AXISLABEL_X: list(CurveTRPL.AXISLABELS_X[""]),
                Curve.KEY_AXISLABEL_Y: list(CurveTRPL.AXISLABEL_DIFFLIFETIME),
            }
        )
        return curve

    def Curve_differential_lifetime_vs_signal(self):
        """
        Returns differential lifetime versus signal (log(signal) proportional to QFLS)
        Differential lifetime: see Curve_differential_lifetime
        Make sure to remove background before calculating differential lifetime.
        """
        curve = self.Curve_differential_lifetime()
        curve.setX(self.y())
        msg = "{} Differential lifetime vs signal"
        curve.update(
            {
                "label": msg.format(self.attr("label")),
                Curve.KEY_AXISLABEL_X: list(CurveTRPL.AXISLABELS_Y[""]),
                Curve.KEY_AXISLABEL_Y: list(CurveTRPL.AXISLABEL_DIFFLIFETIME),
            }
        )
        return curve

    def fitparams_to_clipboard(self):
        popt = self.attr("_popt")
        values = "\t".join([str(v) for v in popt])
        time = self.x()
        if np.max(np.abs(popt[2::2])) > (np.max(time) - np.min(time)):
            msg = "long tau, average not calculated"
            avg = [msg, msg]
        else:
            avg = [str(val) for val in self.fitparams_weightedaverage()]
        text = "{}\t{}\t\t\t\t{}".format(self.attr("label"), values, "\t".join(avg))

        a = tk.Tk()
        a.clipboard_clear()
        a.clipboard_append(text)
        a.destroy()

    def fitparams_weightedaverage(self, silent=False):
        """
        Weighted averages of decays: returns tau_effective with tau_i weighted by A_i,
        and by A_i * tau_i (i.e. integral of exponential). Formulas:
        (sum A_i * tau_i) / (sum A_i), and
        (sum A_i * tau_i^2) / (sum A_i * tau_i)
        Do not use if any tau values > time range of interest. The fit is weakly
        constrained, and the resulting average values may be artifact.
        Warning is raised if any tau value is larger than the time span.

        :param silent: prints suff if False
        :return: Sum (A tau) / Sum (A), Sum (A tau**2) / Sum (A tau)
        """
        with warnings.catch_warnings(record=True) as w:
            avgAtauA, avgAtau2Atau = _sumexp_weightedaverage(
                self.attr("_popt"), self.x()
            )
            # may issue runtime warning if e.g. unsuitable tau values
            if len(w) > 0:
                for w_ in w:
                    print(w_.message)
        if not silent:
            print("Sum (A tau) / Sum (A): {}".format(avgAtauA))
            print("Sum (A tau**2) / Sum (A tau): {}".format(avgAtau2Atau))
        return avgAtauA, avgAtau2Atau


# Private functions
def _roi_to_mask(x, roi):
    """roi: [xmin, xmax]"""
    if roi is None:
        roi = [min(x), max(x)]
    mask = np.ones(len(x), dtype=bool)
    for i in range(len(mask)):
        if x[i] < roi[0] or x[i] > roi[1]:
            mask[i] = False
    return mask, x[mask]


def _sumexp_weightedaverage(popt, time):
    """
    Weighted averages of decays: returns tau_effective with tau_i weighted by A_i,
    and by A_i * tau_i (i.e. integral of exponential). Formulas:
    (sum A_i * tau_i) / (sum A_i), and
    (sum A_i * tau_i^2) / (sum A_i * tau_i)
    Do not use if any tau values > time range of interest. The fit is weakly
    constrained, and the resulting average values may be artifact.
    Warning is raised if any tau value is larger than the time span.
    """
    atau = popt[1:]
    pow2 = np.sum([atau[i] * atau[i + 1] ** 2 for i in range(0, len(atau), 2)])
    pow1 = np.sum([atau[i] * atau[i + 1] for i in range(0, len(atau), 2)])
    pow0 = np.sum([atau[i] for i in range(0, len(atau), 2)])
    if np.max(np.abs(atau[1::2])) > (np.max(time) - np.min(time)):
        msg = "WARNING CurveTRPL, long tau value, weighted average may be artifact"
        warnings.warn(msg, RuntimeWarning)
    return pow1 / pow0, pow2 / pow1


def _fit_resample_x_newx(x, spacing):
    if isinstance(spacing, np.ndarray):
        spacing = list(spacing)
    if isinstance(spacing, (float, int)):
        spacing = np.abs(float(spacing))
        return np.arange(x[0], x[-1] + spacing, spacing)
    if isinstance(spacing, list):
        if len(spacing) == 3:
            return np.arange(spacing[0], spacing[1], spacing[2])
        return np.array(spacing, dtype=float)
    raise RuntimeError("fit_resampleX_newx is rather crude, sorry for that")


def _gui_funclist_trpl_printhelp(curve):
    item = [curve.print_help, "Help!", [], []]
    return [item]


def _gui_funclist_trpl_integration(curve, **kwargs):
    # integration
    alter = str(kwargs["graph"].attr("alter")) if "graph" in kwargs else "['', '']"
    ROI = roundSignificantRange([min(curve.x()), max(curve.x())], 2)
    item = FuncGUI(curve.integrate, "Integrate")
    item.append("ROI", ROI)
    item.appendcbb("data transform", alter, ["raw", alter])
    return [item]


def _gui_funclist_trpl_smoothbin(curve):
    # smooth & bin
    item = [
        curve.CurveTRPL_smoothBin,
        "Smooth & bin",
        ["width", "convolution", "binning"],
        ["9", "hanning", "1"],
        {},
        [
            {},
            {"field": "Combobox", "values": SMOOTH_WINDOW},
            {"field": "Combobox", "values": ["1", "2", "4", "8", "16"]},
        ],
    ]
    return [item]


def _gui_funclist_trpl_difflifetime(curve):
    # differential lifetime
    msg = "First remove background. Tip: apply on fit, or smooth before."
    item = FuncGUI(curve.Curve_differential_lifetime, "Differential lifetime")
    item.append(msg, None, widgetclass=None)
    return [item]


def _gui_funclist_trpl_difflifetimevssignal(curve):
    # differential lifetime vs signal
    msg = "See above."
    item = FuncGUI(
        curve.Curve_differential_lifetime_vs_signal, "Differential lifetime vs signal"
    )
    item.append(msg, None, widgetclass=None)
    msg = (
        "Differential lifetime vs signal to be visualised in log-log plot.\n"
        "Consider first reducing the number of datapoints on the fit curve."
    )
    item.set_tooltiptext(msg)
    return [item]


def _gui_funclist_trpl_fit(curve):
    # fit
    out = []
    if curve.attr("_fitFunc", None) is None or curve.attr("_popt", None) is None:
        ROI = roundSignificantRange([10, max(curve.x())], 2)
        line = FuncGUI(curve.CurveTRPL_fitExp, "Fit exp")
        line.appendcbb("# exp", 2, ["1", "2", "3", "4"], width=3)
        line.append("ROI", ROI, width=12)
        line.append("Fixed values", [0, "", ""])
        line.append("", "", "Frame")
        line.append("      show residuals", False, options={"field": "Checkbutton"})
        line.append(
            "fit curve restricted to ROI", True, options={"field": "Checkbutton"}
        )
        out.append(line)
        # fit as a series of fits
        ROI = roundSignificantRange([1, np.max(curve.x())], 2)
        ROI0 = roundSignificantRange([1, min(50, np.max(curve.x()))], 2)
        line = FuncGUI(curve.CurveTRPL_sequence_fitexp, "Fit piece-wise")
        line.appendcbb("# exp", 3, ["1", "2", "3", "4"], width=2)
        line.append("ROI", ROI, width=9)
        line.append("first", ROI0, width=6)
        line.append("mult", 4)
        line.append("fit max #pts", 1000, width=4)
        msg = (
            "Use-case: fit piece-wise, then\ncompute Differential lifetime vs "
            "signal on the fits, and\ndisplay in loglog tau vs signal."
        )
        line.set_tooltiptext(msg)
        out.append(line)
        # fit using splines
        roi = roundSignificantRange([0.5, np.max(curve.x())], 2)
        line = FuncGUI(curve.CurveTRPL_spline, "Fit spline")
        line.append("ROI", roi, width=12)
        line.appendcbb("data", "semilogx", ["", "semilogx", "semilogy", "loglog"])
        line.append("target # knots", 30)
        line.append("", "", "Frame")
        line.append("      knots spacing min # datapts", 4)
        line.append("spacing max fraction range", 0.1)
        msg = (
            "Fit data with a spline. Knots are chosen to be uniform in log "
            "scale, with minimal spacing every # datapoints, and max spacing "
            "as a fraction of the full range.\nDon't forget to first put peak "
            "at t=0 mand remove background!"
        )
        line.set_tooltiptext(msg)
        out.append(line)
    # if is fit
    else:
        if curve.attr("_fitFunc") in ["func_fitExp"]:
            values = roundSignificant(curve.attr("_popt"), 5)
            params = ["BG"]
            fieldred = {}  # {'width': 6} if len(values)-1 > 4 else {}
            fields = [fieldred]
            while len(values) > len(params) + 1:
                n = "{:1.0f}".format((len(params) + 1) / 2)
                params += ["A" + n, "\u03C4" + n]
                fields += [fieldred, fieldred]
            item = FuncGUI(curve.updateFitParam, "Update fit")
            for i in range(len(params)):
                item.append(params[i], values[i], options=fields[i])
                if i > 2 and (i + 1) % 4 == 1:
                    item.append("", "", widgetclass="Frame")
            out.append(item)
            # fits params to clipboard
            item = FuncGUI(curve.fitparams_to_clipboard, "Send to clipboard")
            item.append("fit parameters and averages", None, widgetclass=None)
            out.append(item)
            # fit weighted averages
            item = FuncGUI(curve.fitparams_weightedaverage, "Calculate tau_eff")
            item.append("tau weighted averages", None, widgetclass=None)
            out.append(item)
            # resample X
            x = curve.x()
            xmin, xmax = min(x[0], x[-1]), max(x[0], x[-1])
            step = np.abs((x[-1] - x[0]) / (len(x) - 1))
            roi = [xmin, xmax, step]
            msg = "delta t, or [0, 1000, 0.5] (now: {} points)"
            out.append(
                [
                    curve.fit_resampleX,
                    "Resample time",
                    [msg.format(len(curve.x()))],
                    [roundSignificant(roi, 5)],
                ]
            )
    return out


def _gui_funclist_trpl_normalize(curve):
    # normalization
    out = []
    unit = curve.data_units()[1]
    revert = False if unit == "" else True
    if not revert:
        out.append(
            [
                curve.normalize,
                "Normalize intensity",
                ["pulse freq Hz", "acquis time s", "bin ps"],
                [
                    curve.attr("_repetfreq_Hz", 1),
                    curve.attr("_acquistime_s", 1),
                    curve.attr("_binwidth_s", 1) * 1e12,
                ],
                {},
                [{"width": 10}, {"width": 7}, {"width": 6}],
            ]
        )
    else:  # get back cts data
        out.append(
            [
                curve.normalizerevert,
                "Restore intensity cts",
                ["Current intensity unit: " + str(unit) + ". factor"],
                [curve.getFactor()],
                {},
                [{"field": "Label"}],
            ]
        )
    return out


def _gui_funclist_trpl_yoffset(curve):
    # y offset (background)
    unit = curve.data_units()[1]
    if unit == "":
        unit = "cts"
    line = FuncGUI(curve.addOffset, "Add offset")
    line.append("new vertical offset (" + unit + ")", curve.getOffset())
    msg = (
        "Remove background.\nTip: Smooth data and display data in lin scale to "
        "be more precise!"
    )
    line.set_tooltiptext(msg)
    return [line]


def _gui_funclist_trpl_xoffset(curve):
    # x offset
    xOffset = curve.getXOffset()
    xOffsetval = xOffset if xOffset != 0 else ""
    line = FuncGUI(curve.addXOffset, "Time offset")
    line.append("new horizontal offset (leave empty for autodetect)", xOffsetval)
    line.set_tooltiptext("The laser pulse should be at t=0")
    return [line]


def _funclistgui_graph_auto_axislabels(curve, **kwargs):
    unitx, unity = curve.data_units()
    lookup_x = {} if unitx == "" else {"": unitx}
    if unity == "":
        unity = "cts"
    lookup_y = {}
    for key in curve.AXISLABELS_Y:
        lookup_y[key] = unity
    return FuncListGUIHelper.graph_axislabels(
        curve, lookup_y=lookup_y, lookup_x=lookup_x, **kwargs
    )
