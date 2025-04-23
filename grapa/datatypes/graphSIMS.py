# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import numpy as np
import copy
from scipy.optimize import fsolve


from grapa.graph import Graph
from grapa.utils.graphIO import GraphIO
from grapa.datatypes.curveSIMS import CurveSIMS

from grapa.mathModule import roundSignificant, is_number


class GraphSIMS(Graph):
    """
    msmtid is an id based on the initial datafile name, in order to be able to
    process simultaneously several SIMS measurements
    class must be written like instance of self is actually a Graph object,
    not a GraphSIMS. Cannot call a method of GraphSIMS as method of self.
    """

    SHORTCUTS = [
        {"aliases": ["GGI", "ggi"], "short": [["^71Ga+"], ["^71Ga+", "^113In+"]]},
        {"aliases": ["CGI", "cgi"], "short": [["Cu+"], ["^71Ga+", "^113In+"]]},
        {"aliases": ["GI", "gi"], "short": [["^71Ga+", "^113In+"], []]},
        {"aliases": ["AAC", "aac"], "short": [["^109Ag+"], ["^109Ag+", "Cu+"]]},
        {
            "aliases": ["I/III", "i/iii"],
            "short": [["^109Ag+", "Cu+"], ["^71Ga+", "^113In+"]],
        },
        {"aliases": ["CuSn", "cusn"], "short": [["Cu+"], ["Sn+"]]},
        {"aliases": ["CuZn", "cuzn"], "short": [["Cu+"], ["Zn+"]]},
        {"aliases": ["ZnSn", "znsn"], "short": [["Zn+"], ["Sn+"]]},
        {"aliases": ["CuZnSn", "cuznsn"], "short": [["Cu+"], ["Zn+", "Sn+"]]},
        {"aliases": ["CGT", "cgt"], "short": [["Cu+"], ["^70Ge+", "Sn+"]]},
        {"aliases": ["GGS", "ggt"], "short": [["^72Ge+"], ["^72Ge+", "Sn+"]]},
    ]

    TOHIDE = [
        "F+",
        "Mg+",
        "Al+",
        "^41K+",
        "Fe+",
        "^65Cu+",
        "^66Zn+",
        "Ga+",
        "Se+",
        "^94Mo+",
        "In+",
        "^118Sn+",
        "^119Sn+",
        "^110Cd+",
        "^Cd112+",
        "^113Cd+",
        "^92Mo+",
        "^96Mo+",
        "Cs+",
    ]

    YIELDS = {
        "^71Ga+": 1,
        "^113In+": 5.5,
        "Cu+": 300,
        "^70Ge+": 4500,
        "Sn+": 480,
        "Zn+": 6000,
    }

    THRESHOLD_SATURATION = 89000

    FILEIO_GRAPHTYPE = "SIMS data"

    AXISLABELS = [["Sputter time", "t", "s"], ["Intensity", "", "counts"]]

    @classmethod
    def isFileReadable(
        cls, _filename, fileext, line1="", line2="", line3="", **_kwargs
    ):
        if fileext == ".txt" and (
            line1.startswith("\ttotal\t") or line3.startswith("Sputter Time (s)")
        ):
            return True
        line1 = line1.strip(" #")
        line2 = line2.strip(" #")
        line3 = line3.strip(" #")
        if line2.startswith("\ttotal\t") and line3.startswith("\tN/A\t"):
            return True
        return False

    def readDataFromFile(self, attributes, **_kwargs):
        # open using fileGeneric
        GraphIO.readDataFromFileGeneric(self, attributes)
        tot, nan = 0, 0
        for curve in self:
            tot += len(curve.y())
            nan += np.sum(np.isnan(curve.y()))
        if len(self) == 0 or nan > tot / 10:
            GraphIO.readDataFromFileGeneric(
                self, attributes, ifReplaceCommaByPoint=True, lstrip=" #"
            )
        # print(self[1].get_attributes())
        # print(self[2].get_attributes())
        self.update({"typeplot": "semilogy"})
        ylabel = self[-1].attr("sputter time (s)")
        if ylabel == "":
            ylabel = GraphSIMS.AXISLABELS[1]
        if self[-1].attr("sputter time (s)") != "":
            self.update({"xlabel": self.formatAxisLabel(GraphSIMS.AXISLABELS[0])})
            self.update({"ylabel": self.formatAxisLabel(ylabel)})
        # set correct labels
        sample = self[-1].attr("sample")
        if sample == "":
            sample = os.path.basename(self[-1].attr("filename")).split(".")[:-1]
            sample = " ".join(sample).replace("_", " ")
        for c in range(len(self)):
            self[c].update({"label": self[c].attr("total"), "sample": sample})
            self.castCurve("CurveSIMS", c, silentSuccess=True)
            self[c].label_auto("${_simselement}")  # BEWARE
        # prints default keywords
        msg = "SIMS available ratio keywords: {}."
        print(msg.format(", ".join(key["aliases"][0] for key in GraphSIMS.SHORTCUTS)))
        # set default SIMS relative yields
        msmtid = self[-1].attr("_SIMSmsmt")
        GraphSIMS.setYieldCoefs(self, msmtid, ifAuto=True)
        # add temporary Ga+In curve
        ok = GraphSIMS.appendReplaceCurveRatio(self, msmtid, "GI", "Ga+In")
        if ok:
            # find edges of Ga+In curve
            c = GraphSIMS.getCurve(self, msmtid, "Ga+In", silent=True)
            ROI = (
                c.findLayerBorders(returnIdx=False)
                if c is not None
                else self[-1].x(index=[0, self[-1].shape(1)])
            )
            GraphSIMS.setLayerBoundaries(self, msmtid, ROI)
        # by default, hide some curves - add others, for kestertes?
        hidden = []
        toHide = GraphSIMS.TOHIDE
        for h in toHide:
            c = GraphSIMS.getCurve(self, msmtid, h, silent=True)
            if c is not None:
                c.update({"linestyle": "none"})
                hidden.append(h)
        if len(hidden) > 0:
            msg = "SIMS curves hidden automatically: {}."
            print(msg.format(", ".join(hidden)))

    def getLayerBoundaries(self, msmtid):
        for c in range(len(self)):
            if self[c].attr("_SIMSmsmt") == msmtid:
                return self[c].attr("_SIMSLayerBoundaries")
        msg = (
            "ERROR graphSIMS getLayerBoundaries cannot find curve with desired "
            "msmtid ({})"
        )
        print(msg.format(msmtid))
        return False

    def setLayerBoundaries(self, msmtid, ROI):
        for c in range(len(self)):
            if self[c].attr("_SIMSmsmt") == msmtid:
                self[c].update({"_SIMSLayerBoundaries": [min(ROI), max(ROI)]})
        self.update({"axvline": [min(ROI), max(ROI)]})
        return GraphSIMS.getLayerBoundaries(self, msmtid)

    def setLayerBoundariesDepthParameters(self, msmtid, ROI, depth):
        ROI = GraphSIMS.setLayerBoundaries(self, msmtid, ROI)
        offset, mult = None, None
        for c in range(len(self)):
            if self[c].attr("_SIMSmsmt") == msmtid:
                if offset is None:
                    ROI = np.argmin(np.abs(self[c].x() - ROI[0])), np.argmin(
                        np.abs(self[c].x() - ROI[1])
                    )
                    x = self[c].x(index=ROI)
                    offset = -x[0]
                    mult = depth / (x[1] - x[0])
                self[c].setDepthParameters(offset, mult)
        if offset is None:
            msg = (
                "ERROR GraphSIMS setLayerBoundariesDepthParameters cannot find "
                "curve with desired msmtid ({})"
            )
            print(msg.format(msmtid))
        return True

    def setLayerBoundariesDepthParametersGUI(self, depth, ROI, msmtid=None):
        """Calibrate depth by linear scaling of input x axis, given two positions and
        a corresponding distance.

        :param depth: known depth difference of ROI
        :param ROI: range of interest, in units of input x data. [xmin, xmax].
        :param msmtid: identifier for the Curves to consider.
        """
        return GraphSIMS.setLayerBoundariesDepthParameters(self, msmtid, ROI, depth)

    def getCurve(self, msmtid, element, silent=False, returnIdx=False):
        for c in range(len(self)):
            if (
                self[c].attr("_SIMSelement") == element
                and self[c].attr("_SIMSmsmt") == msmtid
            ):
                if returnIdx:
                    return c
                return self[c]
        if not silent:
            msg = (
                "ERROR getCurve: cannot find curve associated with element {} and "
                "id {}."
            )
            print(msg.format(element, msmtid))
            return False
        return None

    # function setting custom or defaults SIMS yields
    def setYieldCoefs(self, msmtid, elementYield=None, ifAuto=False):
        if elementYield is None:
            elementYield = {}
        if ifAuto:
            default = copy.deepcopy(GraphSIMS.YIELDS)
            default.update(elementYield)
            elementYield = default
        for key in elementYield:
            c = GraphSIMS.getCurve(self, msmtid, key, silent=True)
            if c is not None:
                c.update({"_SIMSYieldCoef": elementYield[key]})
                print("Graph SIMS setYields {}, {}".format(key, elementYield[key]))

    def appendReplaceCurveRatio(
        self, msmtid, ratioCurves, curveId, attributes={}, savgol=None
    ):
        """
        savgol: 2-element list width and degree for Savistky-Golay smoothening
        """
        if savgol is not None:  # savistky golay smoothening of the Curves
            from scipy.signal import savgol_filter

            savgol = list(savgol)
            if len(savgol) < 2 or not is_number(savgol[0]) or not is_number(savgol[1]):
                savgol = None
                msg = (
                    "Warning GraphSIMS.appendReplaceCurveRatio: savgol must be a "
                    "2-elements list of ints for savistky-golay width and degree. "
                    "Received {}."
                )
                print(msg.format(savgol))
            if savgol is not None:  # tests on savlues
                savgol[0] = max(1, int((savgol[0] - 1) / 2) * 2 + 1, 1)
                # must be an odd number
                savgol[1] = min(max(int(savgol[1]), 1), savgol[0] - 1)
            if savgol == [1, 1]:
                savgol = None
            if savgol is not None:
                msg = (
                    "GraphSIMS: smooth curve data using Savistky-Golay window {} "
                    "degree {}."
                )
                print(msg.format(savgol[0], savgol[1]))

        ratio_curves = GraphSIMS.getRatioProcessRatiocurves(self, ratioCurves)
        ret = True
        ratio = None
        x = None
        attrs = {
            "filename": "",
            "_SIMSlayerBoundaries": "",
            "_simsdepth_offset": "",
            "_simsdepth_mult": "",
            "sample": "",
        }
        yieldcoefs = []
        for i in range(len(ratio_curves)):
            yieldcoefs.append([])
            for j in range(len(ratio_curves[i])):
                curve = GraphSIMS.getCurve(
                    self, msmtid, ratio_curves[i][j], silent=True
                )
                if curve is None:
                    msg = "WARNING SIMS appendReplaceCurveRatio Curve not found: {}."
                    print(msg.format(ratio_curves[i][j]))
                    ret = False
                    return False
                GraphSIMS.check_saturation(curve)
                if ratio is None:
                    tmp = copy.deepcopy(curve.y())
                    ratio = np.array([tmp * 0.0, tmp * 0.0])
                yieldcoef = curve.attr("_SIMSYieldCoef", default=1)
                yieldcoefs[-1].append(yieldcoef)
                tmp = curve.y() * yieldcoef
                if savgol is not None:
                    tmp = savgol_filter(tmp, *savgol, deriv=0)
                try:
                    ratio[i, :] += tmp
                except ValueError as e:
                    msg = (
                        "ERROR GraphSIMS appendReplaceCurveRatio: curve not same size."
                    )
                    print(msg)
                    msg = "{} shape curve.y {} into array {}."
                    print(
                        msg.format(
                            ratio_curves[i][j], curve.y().shape, ratio[i, :].shape
                        )
                    )
                    print(e)
                    print(curve.y())
                if x is None:
                    x = curve.x()
                for key in attrs:
                    if attrs[key] == "":
                        attrs[key] = curve.attr(key)
        for i in range(len(ratio_curves)):
            if len(ratio_curves[i]) == 0:
                ratio[i, :] = 1
        msg = "Create/replace ratio curves {}, using coefficients: {}."
        print(msg.format(ratioCurves, str(yieldcoefs)))
        attrs.update({"label": curveId})
        attrs.update(attributes)
        new = CurveSIMS([x, ratio[0] / ratio[1]], attributes=attrs)
        c = GraphSIMS.getCurve(self, msmtid, curveId, silent=True, returnIdx=True)
        if c is None:
            self.append(new)
        else:
            self.curve_replace(new, c)
        return ret

    def appendReplaceCurveRatioGUI(self, ratioCurves, curveId, msmtid=None):
        return GraphSIMS.appendReplaceCurveRatio(self, msmtid, ratioCurves, curveId)

    def appendReplaceCurveRatioGUISmt(
        self, ratioCurves, curveId, SGw, SGd, msmtid=None
    ):
        """Creates a Curve computed as a ratio of Curves.

        :param ratioCurves: which Curve ratio is to be computed
        :param curveId: give it a beautiful label (e.g. "GGI")
        :param SGw: width of a Savitsky-Golay smooth to apply to the elemental traces
               prior to combination, using window and degree. No effect if w 1 d 1.
               Possibly suitable smoothing w 21 d 3.
        :param SGd: degree of the Savitsky-Golay smoothing.
        :param msmtid: identifier for the Curves to consider.
        """
        return GraphSIMS.appendReplaceCurveRatio(
            self, msmtid, ratioCurves, curveId, savgol=[SGw, SGd]
        )

    def getRatioProcessRatiocurves(self, ratioCurves):
        # handle keywords for ratiocurves
        if isinstance(ratioCurves, str):
            flag = False
            for short in GraphSIMS.SHORTCUTS:
                if ratioCurves in short["aliases"]:
                    msg = "Replaced keyword {} with ratio {}"
                    print(msg.format(ratioCurves, str(short["short"])))
                    ratioCurves = short["short"]
                    flag = True
                    break
            if not flag:
                msg = (
                    "WARNING: SIMS setYieldsFromExternalData unknown key ({}). Used "
                    "{} instead."
                )
                print(msg.format(ratioCurves, GraphSIMS.SHORTCUTS[0]["aliases"][0]))
                ratioCurves = GraphSIMS.SHORTCUTS[0]["short"]
        return ratioCurves

    def getRatio(
        self, msmtid, ROI, ratioCurves, ifReturnAvg=False, ifReturnCoefs=False
    ):
        # output: [ratioOfAveragesOverROI, AverageOverROIofLocalRatio]
        ratio_curves = GraphSIMS.getRatioProcessRatiocurves(self, ratioCurves)
        avgs_of_ratios = np.array([ROI * 0.0, ROI * 0.0])
        ratios_of_avgs = np.array([0.0, 0.0])
        # lenmax = np.max([len(r) for r in ratio_curves])
        out_avg = []
        out_coefs = []
        for i in range(len(ratio_curves)):
            out_avg.append([0] * len(ratio_curves[i]))
            out_coefs.append([1] * len(ratio_curves[i]))
            for j in range(len(ratio_curves[i])):
                key = ratio_curves[i][j]
                curve = GraphSIMS.getCurve(self, msmtid, key, silent=True)
                if curve is not None:
                    GraphSIMS.check_saturation(curve)
                    out_coefs[i][j] = curve.attr("_SIMSYieldCoef", default=1)
                    avgs_of_ratios[i, :] += curve.y(index=ROI) * out_coefs[i][j]
                    out_avg[i][j] = np.average((curve.y(index=ROI))) * out_coefs[i][j]
                    ratios_of_avgs[i] += out_avg[i][j]
                else:
                    msg = "ERROR: SIMS getRatio cannot find curve {}."
                    print(msg.format(key))
            if len(ratio_curves[i]) == 0:
                avgs_of_ratios[i, :] = ROI * 0.0 + 1
                ratios_of_avgs[i] = 1
        ratio_of_avg = ratios_of_avgs[0] / ratios_of_avgs[1]
        mask = avgs_of_ratios[1] != 0
        avg_of_ratio = np.average(avgs_of_ratios[0][mask] / avgs_of_ratios[1][mask])

        if ifReturnAvg:
            out = [ratio_of_avg, avg_of_ratio, out_avg]
        else:
            out = [ratio_of_avg, avg_of_ratio]
        if ifReturnCoefs:
            out = out + [out_coefs]
        return out

    def targetRatioSetYield(
        self, msmtid, ROI, ratioCurves, element, target, silent=False
    ):
        ratioInit = ratioCurves
        ratioCurves = GraphSIMS.getRatioProcessRatiocurves(self, ratioCurves)
        ratios = GraphSIMS.getRatio(self, msmtid, ROI, ratioCurves, ifReturnAvg=True)
        tunable = [[False] * len(r) for r in ratioCurves]
        for i in range(len(ratioCurves)):
            for j in range(len(ratioCurves[i])):
                if ratioCurves[i][j] == element:
                    tunable[i][j] = True

        def func(coef, avgs, tunable, target):
            # print("func avgs", avgs, "tunable", tunable, "target", target)
            # print("   coef", coef, type(coef))
            if isinstance(coef, np.ndarray):
                coef = coef[0]
            out0 = np.sum(
                [
                    avgs[0][j] * coef if tunable[0][j] else avgs[0][j]
                    for j in range(len(avgs[0]))
                ]
            )
            out1 = np.sum(
                [
                    avgs[1][j] * coef if tunable[1][j] else avgs[1][j]
                    for j in range(len(avgs[1]))
                ]
            )
            return out0 / out1 - target

        res = fsolve(func, 1, args=(ratios[2], tunable, target))
        if isinstance(res, np.ndarray):
            res = res[0]
        curve = GraphSIMS.getCurve(self, msmtid, element, silent=True)
        if curve is None:
            return False
        old = curve.attr("_SIMSYieldCoef")
        curve.update({"_SIMSYieldCoef": res * old})
        if not silent:
            msg = "SIMS adjust yield to reach {} ratio {}: {} yield {} (old value {})."
            print(
                msg.format(
                    ratioInit,
                    target,
                    curve.attr("_SIMSelement"),
                    curve.attr("_SIMSYieldCoef"),
                    old,
                )
            )
        return GraphSIMS.getRatioGUI(
            self, ROI, ratioInit, msmtid=msmtid, ifROIisIdx=True
        )

    def targetRatioSetYieldGUI(self, ROI, ratioCurves, element, target, msmtid=None):
        """Adjust the relative SIMS yield of the Curve of a given `element`, such that
        the Curve ratio `ratioCurves` has the chosen value `target` within the selected
        range of interest `ROI`.

        :param ROI: range of interest, in units of input x data. [x_min, x_max].
        :param ratioCurves: which Curve ratio is to be adjusted
        :param element: which element the yield has to be adjusted
        :param target: the known integrated ratio we want to reach by adjusting the
               element yield
        :param msmtid: identifier for the Curves to consider.
        """
        ROI = GraphSIMS.ROI_GUI2idx(self, msmtid, ROI)
        return GraphSIMS.targetRatioSetYield(
            self, msmtid, ROI, ratioCurves, element, target
        )

    def getRatioGUI(self, ROI, ratioCurves, msmtid=None, ifROIisIdx=False):
        """Computes a ratio of curves within a ROI.

        :param ROI: range of interest, in units of input x data. [x_min, x_max]
        :param ratioCurves:  which Curve ratio is to be adjusted
        :param msmtid:  identifier for the Curves to consider.
        :param ifROIisIdx: if True, the ROI is given in units of index. If False, in
               terms of (calibrated) depth.
        """

        if msmtid is None:
            print("ERROR GraphSIMS getRatioGUI: msmtid was not provided.")
            return False
        ratioName = str(ratioCurves)
        # GUI function: ROI is assumed is be function of sputtering time
        if not ifROIisIdx:
            ROI = GraphSIMS.ROI_GUI2idx(self, msmtid, ROI)
        GGIs = GraphSIMS.getRatio(self, msmtid, ROI, ratioCurves, ifReturnCoefs=True)
        msg = "{}: {} (avg of local ratios: {}). Coefficients: {}, {}."
        out = msg.format(
            ratioName,
            "{:1.4f}".format(GGIs[0]),
            "{:1.4f}".format(GGIs[1]),
            str(roundSignificant(GGIs[2][0], 4)),
            str(roundSignificant(GGIs[2][1], 4)),
        )
        return out

    def ROI_GUI2idx(self, msmtid, ROI):
        c = None
        for curve in range(len(self)):
            if self[curve].attr("_SIMSmsmt") == msmtid:
                c = curve
                break
        if c is None:
            msg = "GraphSIMS getRatioGUI cannot find curve with correct msmtid ({})."
            print(msg.format(msmtid))
        if ROI == "":
            return np.arange(0, self[c].shape(1) - 1)
        return np.arange(
            np.argmin(np.abs(self[c].x() - min(ROI))),
            np.argmin(np.abs(self[c].x() - max(ROI))),
        )

    @classmethod
    def check_saturation(cls, curve):
        if np.max(curve.y()) > cls.THRESHOLD_SATURATION:
            msg = (
                "WARNING: please check for possible saturation of curve {} ({}). "
                "Expect saturation if signal > {}."
            )
            print(
                msg.format(
                    curve.attr("label"),
                    curve.attr("_simselement"),
                    cls.THRESHOLD_SATURATION,
                )
            )
