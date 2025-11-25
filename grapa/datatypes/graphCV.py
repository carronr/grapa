# -*- coding: utf-8 -*-
"""
To parse files contaiing capacitance versus voltage C-V data.
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

from copy import deepcopy
import logging

import numpy as np

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.utils.parser_dispatcher import FileParserDispatcher
from grapa.datatypes.curveCV import CurveCV
from grapa.utils.error_management import issue_warning

logger = logging.getLogger(__name__)


class GraphCV(Graph):
    """To parse files contaiing capacitance versus voltage C-V data."""

    FILEIO_GRAPHTYPE = "C-V curve"

    AXISLABELS = [CurveCV.AXISLABELS_X[""], CurveCV.AXISLABELS_Y[""]]

    @classmethod
    def isFileReadable(cls, filename, fileext, line1="", **_kwargs):
        line1filtered = line1.encode("ascii", errors="ignore").decode()
        if (
            fileext == ".txt"
            and line1filtered.strip().startswith("Sample name:")
            and filename.startswith("C-V_")
        ):
            return True
        return False

    def readDataFromFile(self, attributes, **_kwargs):
        len0 = len(self)
        FileParserDispatcher.readDataFromFileGeneric(self, attributes)
        self.castCurve("Curve CV", len0, silentSuccess=True)
        # label based on file name, maybe want to base it on file content
        label = (
            self[len0]
            .attr("label")
            .replace("C-V ", "")
            .replace("citance[F]", "citance [F]")
            .replace(" [F]", "")
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
        units = ["V", "nF"]
        self[len0].setY(1e9 * self[len0].y())
        if np.max(self[len0].y()) > 1e5:
            msg = (
                "WARNING: GraphCV, expected raw data in F, performing internal"
                "conversion into nF. Please check if units are correct (min %s, max %s)"
            )
            print(msg % (np.min(self[len0].y()), np.max(self[len0].y())))
        # compute phase angle if required
        nb_add = 0
        if "_cvloadphase" in attributes and attributes["_cvloadphase"]:
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
                msg = "ERROR GraphCV Read %s as CurveCf with phase: cannot find R."
                issue_warning(logger, msg, self.filename)
            else:
                phase_angle = np.arctan(f * 2 * np.pi * C / conductance) * 180.0 / np.pi
                #                phase_angle = np.arctan(C / conductance) * 180. / np.pi
                attr = self[len0].get_attributes()
                self.append(Curve([self[len0].x(), phase_angle], attr))
                nb_add += 1
        # delete unneeded Curves
        for c in range(len(self) - 1 - nb_add, len0, -1):
            self.curve_delete(c)
        # cosmetics
        axislabels = deepcopy(GraphCV.AXISLABELS)
        # normalize with area C
        area = self[len0].attr("cell area (cm2)", None)
        if area is None:
            area = self[len0].attr("cell area", None)
        if area is None:
            area = self[len0].attr("area", None)
        if area is not None:
            self[len0].setY(self[len0].y() / area)
            self[len0].update({"cell area (cm2)": area})
            axislabels[1][2] = axislabels[1][2].replace("F", "F cm$^{-2}$")
            units[1] = "nF cm-2"
            if not self.silent:
                print("Capacitance normalized to area", self[len0].getArea(), "cm2.")
        self[len0].data_units(*units)
        # graph cosmetics
        self.update(
            {
                "xlabel": self.formatAxisLabel(axislabels[0]),
                "ylabel": self.formatAxisLabel(axislabels[1]),
            }
        )  # default
