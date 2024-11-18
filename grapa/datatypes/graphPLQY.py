# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2024, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import numpy as np

from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.curve import Curve
from grapa.datatypes.curveJscVoc import CurveJscVoc


class GraphPLQY(Graph):
    """
    Read PLQY of home-made setup Abt207 Empa
    """

    FILEIO_GRAPHTYPE = "PLQY absolute PL measurement"

    KEY_AVGAMP = "Average amplitude [A]"
    KEY_AVGPHASE = "Average phase [deg]"
    KEY_AVGPLQY = "Average PLQY [-]"

    @classmethod
    def isFileReadable(cls, filename, fileext, line1="", line2="", line3="", **kwargs):
        if fileext == ".txt":
            if line1.startswith(
                "Output file of Abt207 PLQY setup"
            ) and filename.startswith("PLQY"):
                return True
            if line1.startswith(
                "Output file of Abt207 home-made PLQY"
            ) and filename.startswith("PLPowerDep_"):
                return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        filename = os.path.basename(self.filename)
        if filename.startswith("PLQY"):
            return GraphPLQY.readDataFromFilePLQY(self, attributes, **kwargs)
        elif filename.startswith("PLPowerDep_"):
            return GraphPLQY.readDataFromFilePLPowDep(self, attributes, **kwargs)
        print(
            "GraphPLQY BUG - Sorry, was a mistake, do not know how to open the file..."
        )

    def readDataFromFilePLPowDep(self, attributes, **kwargs):
        """to open Power dependency module"""
        le = len(self)
        GraphIO.readDataFromFileGeneric(self, attributes, **kwargs)
        # remove unnecessary Curves
        while len(self) > le + 1:
            del self[le + 1]
        # cosmetics and functional modifications
        self[-1].update(
            {
                "linespec": "o-",
                "temperature": self[-1].attr("fit temperature [k]"),
                "label": self[-1].attr("sample"),
            }
        )
        self.castCurve(CurveJscVoc.CURVE, -1, silentSuccess=True)
        self.update(
            {
                "alter": ["", "abs"],
                "typeplot": "semilogy",
                "xlabel": ["Calculated QFLS", "", "meV"],
                "ylabel": ["Excitation power as current density", "", "mA cm$^{-2}$"],
            }
        )

    def readDataFromFilePLQY(self, attributes, **kwargs):
        """To open the rest - output of measurements with lockin"""
        fname, _ = os.path.splitext(os.path.basename(self.filename))
        split = fname.split("_")
        label = ""
        if len(split) > 1:
            label = " ".join(split[1:])
        # read data
        le = len(self)
        GraphIO.readDataFromFileGeneric(self, attributes, **kwargs)
        # add R vs time (not R average) - before modifying other curves
        curve_time_r = None
        if len(self) > le + 1:
            convert_i_plqy = np.nan
            try:
                convert_i_plqy = float(self[le].attr("Sample factor I to PLQY"))
            except ValueError:
                pass
            time = self[le].x()
            r = np.sqrt(self[le].y() ** 2 + self[le + 1].y() ** 2)
            plqy = r * convert_i_plqy
            curve_time_r = Curve([time, plqy], {})
        # tune X vs time into Y vs X
        self[le].setX(self[le].y())
        self[le].setY(self[le + 1].y())
        if label == "":
            label = self[le].attr("label")
        if fname.startswith("PLQYcalibration"):
            label += " calibration"
        if fname.startswith("PLQYzero"):
            label += " zero"
        if len(self) > le + 4:
            label += " PLQY {:.1e}".format(self[le + 4].y()[-1])
        self[le].update({"label": label, "time": ""})
        if self[le].attr("sample") == "":
            self[le].update({"sample": self[le].attr("sample name")})
        # retrieve info, then delete y vs time, rho avg, theta avg, PLQY avg
        if len(self) > le + 2:
            self[le].update({GraphPLQY.KEY_AVGAMP: self[le + 2].y()[-1]})
        if len(self) > le + 3:
            self[le].update({GraphPLQY.KEY_AVGPHASE: self[le + 3].y()[-1]})
        if len(self) > le + 4:
            self[le].update({GraphPLQY.KEY_AVGPLQY: self[le + 4].y()[-1]})
        # delete unneeded
        while len(self) > le + 1:
            del self[le + 1]
        #
        if curve_time_r is not None:
            curve_time_r.update(self[le].getAttributes())
            curve_time_r.update({"label": label + " PLQY vs time"})
            curve_time_r.swapShowHide()
            self.append(curve_time_r)
        # graph cosmetics
        self.update(
            {
                "xlabel": ["Signal x", "", "A"],
                "ylabel": ["Signal y", "", "A"],
                "axhline": [0, {"linewidth": 0.5}],
                "axvline": [0, {"linewidth": 0.5}],
                "collabels": "",
            }
        )
