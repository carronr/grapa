# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import numpy as np

from grapa.graph import Graph
from grapa.utils.parser_dispatcher import FileParserDispatcher
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

    AXISLABEL_A_X = ["Signal x", "", "A"]
    AXISLABEL_A_Y = ["Signal y", "", "A"]
    AXISLABEL_TIME = ["Time", "", "s"]
    AXISLABEL_PLQY = ["PLQY", "", "-"]

    @classmethod
    def isFileReadable(cls, filename, fileext, line1="", **_kwargs):
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
        filename = os.path.basename(str(self.filename))
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
        FileParserDispatcher.readDataFromFileGeneric(self, attributes, **kwargs)
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
        fname, _ = os.path.splitext(os.path.basename(str(self.filename)))
        split = fname.split("_")
        label = ""
        if len(split) > 1:
            label = " ".join(split[1:])
        # read data
        le = len(self)
        FileParserDispatcher.readDataFromFileGeneric(self, attributes, **kwargs)
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
        self[le].update({"label": label + " Y vs X", "time": ""})
        self[le].data_units(unit_x="A", unit_y="A")
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
        # add PLQY vs time, graph cosmetics
        xlabel = GraphPLQY.AXISLABEL_A_X
        ylabel = GraphPLQY.AXISLABEL_A_Y
        axvline = [0, {"linewidth": 0.5}]
        axhline = [0, {"linewidth": 0.5}]
        typeplot = ""
        self[le].update({Curve.KEY_AXISLABEL_X: xlabel, Curve.KEY_AXISLABEL_Y: ylabel})
        if curve_time_r is not None:
            xlabel = GraphPLQY.AXISLABEL_TIME
            ylabel = GraphPLQY.AXISLABEL_PLQY
            axvline, axhline = "", ""
            typeplot = "semilogy"
            curve_time_r.update(dict(self[le].get_attributes()))
            curve_time_r.update(
                {
                    "label": label + " PLQY_avg vs time",
                    Curve.KEY_AXISLABEL_X: xlabel,
                    Curve.KEY_AXISLABEL_Y: ylabel,
                }
            )
            curve_time_r.data_units(unit_x="s", unit_y="")
            self.append(curve_time_r)
            self[le].visible(False)
        self.update(
            {
                "xlabel": xlabel,
                "ylabel": ylabel,
                "axhline": axhline,
                "axvline": axvline,
                "typeplot": typeplot,
                "subplots_adjust": [0.20, 0.15],
            }
        )
