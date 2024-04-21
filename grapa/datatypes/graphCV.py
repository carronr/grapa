# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

from copy import deepcopy
import warnings
import numpy as np

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.graphIO import GraphIO


class GraphCV(Graph):
    FILEIO_GRAPHTYPE = "C-V curve"

    AXISLABELS = [["Voltage", "V", "V"], ["Capacitance", "C", "nF"]]

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1="", line2="", line3="", **kwargs):
        if (
            fileExt == ".txt"
            and line1.strip()[:12] == "Sample name:"
            and fileName[0:4] == "C-V_"
        ):
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        len0 = len(self)
        GraphIO.readDataFromFileGeneric(self, attributes)
        self.castCurve("Curve CV", len0, silentSuccess=True)
        # label based on file name, maybe want to base it on file content
        label = (
            self[len0]
            .attr("label")
            .replace("C-V ", "")
            .replace(" [nF]", "")
            .replace(" Capacitance", "")
            .replace("T=", "")
        )
        self[len0].update({"label": label, "label_initial": label})
        # cell name, sample name
        cellname = self[len0].attr("cell name")
        if self[len0].attr("cell") == "" and cellname != "":
            self[len0].update({"cell": cellname})
        samplename = self[len0].attr("sample name")
        if self[len0].attr("sample") == "" and samplename != "":
            self[len0].update({"sample": samplename})
        # set [nF] units
        self[len0].setY(1e9 * self[len0].y())
        # compute phase angle if required
        nb_add = 0
        if self.attr("_CVLoadPhase", False) is not False:
            f = 1e5  # [Hz]
            C = self[len0].y() * 1e-9  # in [F], not [nF]
            conductance = None
            # retrieve resistance curve, not always same place
            for c in range(len0 + 1, len(self)):
                lbl = self[c].attr("Voltage")
                if isinstance(lbl, str) and lbl[:1] == "R":
                    conductance = 1 / self[c].y()
                    break
            if conductance is None:
                msg = (
                    "ERROR GraphCV Read"
                    + self.filename
                    + "as CurveCf with phase: cannot find R."
                )
                print(msg)
                warnings.warn(msg)
            else:
                phase_angle = np.arctan(f * 2 * np.pi * C / conductance) * 180.0 / np.pi
                #                phase_angle = np.arctan(C / conductance) * 180. / np.pi
                self.append(
                    Curve([self[len0].x(), phase_angle], self[len0].getAttributes())
                )
                nb_add += 1
        # delete unneeded Curves
        for c in range(len(self) - 1 - nb_add, len0, -1):
            self.deleteCurve(c)
        # cosmetics
        axisLabels = deepcopy(GraphCV.AXISLABELS)
        # normalize with area C
        area = self[len0].attr("cell area (cm2)", None)
        if area is None:
            area = self[len0].attr("cell area", None)
        if area is None:
            area = self[len0].attr("area", None)
        if area is not None:
            self[len0].setY(self[len0].y() / area)
            self[len0].update({"cell area (cm2)": area})
            axisLabels[1][2] = axisLabels[1][2].replace("F", "F cm$^{-2}$")
            if not self.silent:
                print("Capacitance normalized to area", self[len0].getArea(), "cm2.")
        # graph cosmetics
        self.update(
            {
                "xlabel": self.formatAxisLabel(axisLabels[0]),
                "ylabel": self.formatAxisLabel(axisLabels[1]),
            }
        )  # default
