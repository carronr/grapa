# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np
import copy

from grapa.curve import Curve
from grapa.mathModule import roundSignificant
from grapa.utils.funcgui import FuncListGUIHelper, AlterListItem


def findXAtValuePolynom(datax, datay, target, ifPlot=False):
    if len(datax) < 3 or len(datay) < 3:
        print(
            "ERROR CurveSIMS findXAtValuePolynom: not enough data to fit",
            "3rd degree polynom, returned 0.",
        )
        return [0, 0]
    z = np.polyfit(datax, datay, 3, full=True)[0]
    roots = np.roots((np.poly1d(z) - target))
    rootsReal = np.real(roots[np.isreal(roots)])
    rootsInROI = rootsReal[(rootsReal >= min(datax)) * (rootsReal <= max(datax))]
    if len(rootsInROI) == 0:
        rootsInROI = rootsReal
    if len(rootsInROI) == 0:
        print(
            "Error findXAtValuePolynom cannot find suitable value.",
            target,
            datax,
            datay,
        )
    # to handle cases where the polynom has several solutions
    root = np.average(rootsInROI)
    idx = np.argmin(np.abs(datax - root))
    if ifPlot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(datax, datay, ".-b")
        plt.plot(datax, np.polyval(z, datax), "k")
    return [root, idx]


def selectDataEdge(curve, ROI, targetRel=None, threshRel=None, ifTrailingEdge=False):
    """
    Identify transition from low to high level
    """
    if targetRel is None:
        targetRel = 0.5
    if threshRel is None:
        threshRel = [targetRel * 0.6, targetRel * 0.6 + 1 * 0.4]
    y = copy.deepcopy(curve.y(index=ROI))
    iMin, iMax = y.argmin(), y.argmax()
    targetAbs = y[iMin] + targetRel * (y[iMax] - y[iMin])
    # data normalization in [0,1] interval
    y = (y - y[iMin]) / y[iMax]
    if not ifTrailingEdge:
        rangeThres = [ROI[0] + (y > t).argmax() for t in threshRel]
    else:
        rangeThres = [ROI[0] + iMax + (y[iMax:] < t).argmax() for t in threshRel]
    rangeThres.sort()
    rangeThres = range(rangeThres[0], rangeThres[1] + 1)
    datax = curve.x(index=rangeThres)
    datay = curve.y(index=rangeThres)
    # idx = ROI[0] + iMax + (y[iMax:] < 0.5 * y[iMax]).argmax()
    return [datax, datay, targetAbs, rangeThres]


class CurveSIMS(Curve):
    """CurveSIMS offer some basic processing of SIMS data.
    One can notably compute ratios of curves, for which relative yields have to be
    provided or can be calculated using known integral ratios.

    Curves ratios are noted with keywords such as "ggi" (notation shortcuts), or
    explicitly as "[['^71Ga+'], ['^71Ga+', '^113In+']]", to be interpreted as ratio of
    sums of curve values mutliplied by their individual yields.
    An empty top or bottom list is replaced by 1."
    """

    CURVE = "Curve SIMS"

    ALTER_SIMS_TIME_DEPTH = "SIMSdepth"

    AXISLABELS_X = {
        "": ["Sputter time", "t", "s"],
        ALTER_SIMS_TIME_DEPTH: ["Depth", "", "nm"],
    }

    FORMAT_AUTOLABEL = ["${_simselement}", "${sample} ${_simselement}"]

    def __init__(self, *args, **opts):
        Curve.__init__(self, *args, **opts)
        self.update({"Curve": CurveSIMS.CURVE})
        # SIMS data files are a bit special, header start with empty cell
        if self.attr("label") == "" and self.attr("empty0") != "":
            self.update({"label": self.attr("empty0"), "empty0": ""})
        # some additional attributes, which must not override data saved in the data file
        add = {
            "_SIMSelement": self.attr("label"),
            "_SIMSmsmt": "".join(
                self.attr("filename").split("/")[-1].split("\\")[-1].split(".")[:-1]
            ).replace("_", ""),
            "_SIMStotal": 0,
            "_SIMSYieldCoef": 1,
        }
        for key in add:
            if self.attr(key) == "":
                self.update({key: add[key]})
        # check and clean first 2 empty lines (total and header lines might have been mistaken)
        if np.isnan(self.x(index=1)) and np.isnan(self.y(index=1)):
            self.update({"_SIMStotal": self.y(index=0)})
            self.data = np.array([self.x()[2:], self.y()[2:]])

    # RELATED TO GUI
    def funcListGUI(self, **kwargs):
        from grapa.datatypes.graphSIMS import GraphSIMS

        out = Curve.funcListGUI(self, **kwargs)
        # auto-label
        out.append(
            [
                self.label_auto,
                "Auto label",
                ["template"],
                [self.FORMAT_AUTOLABEL[0]],
                {},
                [{"field": "Combobox", "values": self.FORMAT_AUTOLABEL}],
            ]
        )
        # specific to SIMS data
        msmtidDict = {"msmtid": self.attr("_SIMSmsmt")}
        ratios = [k["aliases"][0] for k in GraphSIMS.SHORTCUTS] + [
            "other ratio"
        ]  #  + ['ratio as ["elem0"],["elem1","elem2"]']
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...] (, [default1, default2, ...], {'hiddenVar1':value1}) ]
        out.append(
            [
                self.localizeEdgeFromGUI,
                "Find edge",
                ["ROI", "target", "threshold", "edge"],
                [[self.x(0), self.x(self.shape(1) - 1)], 0.5, [0.3, 0.7], "leading"],
                {},
                [
                    {},
                    {},
                    {},
                    {
                        "field": "Combobox",
                        "values": ["leading", "trailing"],
                        "width": 7,
                    },
                ],
            ]
        )
        ROI = (
            [self.x(0), self.x(self.shape(1) - 1)]
            if self.attr("_SIMSLayerBoundaries") == ""
            else self.attr("_SIMSLayerBoundaries")
        )
        depth = (
            (ROI[1] - ROI[0]) * self.attr("_SIMSdepth_mult")
            if self.attr("_SIMSdepth_mult") != ""
            else 2
        )
        out.append(
            [
                GraphSIMS.setLayerBoundariesDepthParametersGUI,
                "Calibrate depth",
                ["Depth", "ROI"],
                [depth, roundSignificant(ROI, 5)],
                msmtidDict,
            ]
        )
        labels = ["^113In+"]
        if "graph" in kwargs:
            labels = []
            for c in kwargs["graph"].curves("_SIMSmsmt", msmtidDict["msmtid"]):
                labels.append(c.attr("_SIMSelement"))
            if len(labels) == 0:
                labels = ["^113In+"]
        out.append(
            [
                GraphSIMS.targetRatioSetYieldGUI,
                "Adjust compos.",
                ["ROI", "Ratio", "Elem.", "Compos."],
                [ROI, "GGI", ("^113In+" if "^113In+" in labels else labels[0]), 0.35],
                msmtidDict,
                [
                    {},
                    {"field": "Combobox", "values": ratios, "width": 6},
                    {"field": "Combobox", "values": labels, "width": 8},
                    {"width": 6},
                ],
            ]
        )
        out.append(
            [
                GraphSIMS.getRatioGUI,
                "Compute ratio",
                ["ROI", "Ratio"],
                [ROI, "GGI"],
                msmtidDict,
                [{}, {"field": "Combobox", "values": ratios}],
            ]
        )
        out.append(
            [
                GraphSIMS.appendReplaceCurveRatioGUISmt,
                "Create curve ratio",
                ["Ratio", "Curve name", "Smooth S-G w", "d"],
                ["GGI", "GGI", 1, 1],
                msmtidDict,
                [{"field": "Combobox", "values": ratios, "width": 6}, {}, {}, {}],
            ]
        )
        out.append([self.cropDataROI, "Crop trace inside ROI", ["ROI"], [ROI]])
        out.append([self.print_help, "Help!", [], []])

        lookup_x = {self.ALTER_SIMS_TIME_DEPTH: "nm"}
        out += FuncListGUIHelper.graph_axislabels(self, lookup_x=lookup_x, **kwargs)

        self._funclistgui_memorize(out)
        return out

    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        doc = "sputter time as default x coordinate"
        out.append(AlterListItem("Semilogy", ["", "idle"], "semilogy", doc))
        doc = "depth as x axis, provided depth was calibrated (see analysis functions)."
        item = AlterListItem(
            "Raw <-> Depth", [self.ALTER_SIMS_TIME_DEPTH, ""], "semilogy", doc
        )
        out.append(item)
        return out

    # alter: s to z, normalized (in toggle switch?)

    def setDepthParameters(self, offset, mult):
        self.update({"_SIMSdepth_offset": offset, "_SIMSdepth_mult": mult})

    # other methods
    def localizeEdge(
        self, ROI, targetRel, threshRel=None, ifTrailingEdge=False, ROIrefinement=0
    ):
        # check if ROI is a actual range or only extrema
        if (
            isinstance(ROI, (list, tuple))
            and len(ROI) == 2
            and np.abs(ROI[1] - ROI[0]) > 1
        ):
            ROI.sort()
            ROI = range(int(ROI[0]), int(ROI[1]))
        #            print ('Localize edge ROI correction (', min(ROI), max(ROI), ')')
        else:
            pass
        # first localize max in ROI, then compute max*target and select data subset [max*threshold[0], max*threshold[1]]
        [datax, datay, targetAbs, rangeThresholds] = selectDataEdge(
            self, ROI, targetRel, ifTrailingEdge=ifTrailingEdge, threshRel=threshRel
        )
        # then interpolate data subset with polynom, and find intersection with target*max
        [root, idx] = findXAtValuePolynom(datax, datay, targetAbs)
        idx += rangeThresholds[0]
        if ROIrefinement > 0:
            delta = ROIrefinement * (
                np.array([min(rangeThresholds), max(rangeThresholds)]) - idx
            )
            # check at least 1 point enlargment
            delta[0], delta[1] = idx + min(delta[0], idx - 1), idx + max(
                delta[1], idx + 1
            )
            # check indices are valid
            delta[0], delta[1] = max(0, delta[0]), min(delta[1], len(self.y()) - 1)
            # do not want a larger ROI than before
            delta[0], delta[1] = max(min(ROI), delta[0]), min(delta[1], max(ROI))
            [root, idx] = self.localizeEdge(
                range(delta[0], delta[1]),
                targetRel,
                threshRel=threshRel,
                ifTrailingEdge=ifTrailingEdge,
            )
        return [root, idx]

    def localizeEdgeFromGUI(self, ROI, targetRel, threshRel, ifTrailingEdge):
        """Auto-detect edges in the selected curve.

        :param ROI: range of interest, in units of input x data. [x_min, x_max]
        :param targetRel: value relative to max within ROI the edge is defined.
               Default 0.5
        :param threshRel: threshold for data to consider relative to maximum,
               default [0.3, 0.7].
        :param ifTrailingEdge: 0 if leading edge, 1 if trailing edge.
        """
        # assumes ROI is in the form [sputterTimeMin, sputterTimeMax]
        if ifTrailingEdge == "trailing":
            ifTrailingEdge = True
        elif ifTrailingEdge == "leading":
            ifTrailingEdge = False
        if len(list(ROI)) > 2:
            ROI = [min(ROI), max(ROI)]
        ROI = [
            np.argmin(np.abs(self.x() - ROI[0])),
            np.argmin(np.abs(self.x() - ROI[1])),
        ]
        res = self.localizeEdge(
            ROI, targetRel, threshRel=threshRel, ifTrailingEdge=ifTrailingEdge
        )
        return (
            "Edge position for curve "
            + self.attr("_SIMSelement")
            + " at: "
            + str(roundSignificant(res[0], 5))
        )

    # find indices of
    def findLayerBorders(
        self,
        ROI=None,
        targetRel=0.5,
        threshRel=[0.3, 0.7],
        ROIrefinement=0,
        returnIdx=True,
    ):
        # targetRel = None#0.5
        # threshRel = None # [0.3, 0.7]
        if ROI is None:  # [range(0,10), range(10,20)]
            ROI = [
                range(0, int(self.shape(1) / 2)),
                range(int(self.shape(1) / 2), int(self.shape(1))),
            ]
        posInterfGaIn = [np.nan, np.nan]
        for i in range(2):
            [root, idx] = self.localizeEdge(
                ROI[i],
                targetRel,
                threshRel=threshRel,
                ifTrailingEdge=i,
                ROIrefinement=ROIrefinement,
            )
            posInterfGaIn[i] = idx if returnIdx else root
        return posInterfGaIn

    def cropDataROI(self, ROI):
        """Delete datapoints with x value outside specified range.

        :param ROI: Range of interest, in units of x data. [x_min, x_max].
        """
        x, y = self.x(), self.y()
        mask = (x >= np.min(ROI)) * (x <= np.max(ROI))
        self.data = np.array([x[mask], y[mask]])

    def print_help(self):
        super().print_help()
        from grapa.datatypes.graphSIMS import GraphSIMS

        print("\nList of existing ratioCurves keywords:")
        for short in GraphSIMS.SHORTCUTS:
            print(" - {}: {}".format(short["aliases"][0], short["short"]))

        """
        print("*** *** ***")
        print("CurveSIMS offer some basic processing of SIMS data.")
        print(
            "Each can notably compute ratios of curves, for which a relative yield has to be provided."
        )
        print(
            "Curves ratios are noted with keywords such as \"ggi\", or as \"[['^71Ga+'],['^71Ga+', '^113In+']]\". For 1st and 2nd list the curves multiplied by their yield are summed, and the ratio is finally computed. An empty top or bottom list is replaced by 1."
        )
        """

        """
        print("Curve transforms:")
        print("- Semilogy, with etching time as default x coordinate,")
        print(
            "- Raw <-> Depth: depth as x axis, provided depth was calibrated (see analysis functions)."
        )
        print("Analysis functions:")
        print("- Find edge: auto-detect edges in the selected curve. Parameters:")
        print("  ROI: range of interest, in units of input x data,")
        print("  Target: default 0.5,")
        print("  Threshold: default [0.3,0.7]")
        print("  ifTrailing: 0 if leading edge, 1 if trailing edge.")
        print(
            "- Calibrate depth: calibrate depth by linear scaling of input x axis. Parameters:"
        )
        print("  Depth: know depth of ROI,")
        print("  ROI: range of interest, in units of input x data.")
        print(
            "- Adjust composition: adjust the relative SIMS yield of a curve such that the chosen"
        )
        print(
            "  curve ratio has the chosen value within the selected range of interest. Parameters:"
        )
        print("  ROI: range of interest, in units of input x data,")
        print("  Ratio: which Curve ratio is to be adjusted,")
        print("  Element: which element the yield has to be adjusted,")
        print(
            "  Compos: the know composition we want to reach by adjusting the element yield."
        )
        print("- Compute ratio: computes a ratio of curves within a ROI.")
        print("  ROI: range of interest, in units of input x data,")
        print("  Ratio: which Curve ratio is to be computed.")
        print("- Create curve ratio: creates a Curve computed as a ratio of Curves.")
        print("  Ratio: which Curve ratio is to be computed,")
        print("  Curve name: give it a beautiful name.")
        print(
            "  Smooth S-G w, d: a Savitsky-Golay smoothening can be applied to the elemental traces"
        )
        print(
            "  prior to combination, using window and degree. No effect if w 1 d 1, possibly w 21 d 3."
        )
        self.printHelpFunc(self.cropDataROI)
        self.printHelpFunc(self.autoLabel)
        print("   ")
        """
        return True
