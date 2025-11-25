# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 10:57:14 2017

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
from re import findall as refindall

import numpy as np

from grapa.graph import Graph
from grapa.curve import Curve


class GraphMCAfit(Graph):
    """to open html output of pymca XRF software"""

    FILEIO_GRAPHTYPE = "XRF fit areas"

    AXISLABELS = [["Data", "", None], ["Value", "", None]]

    @classmethod
    def isFileReadable(cls, _filename, fileext, line1="", **_kwargs):
        if fileext == ".html" and line1[0:24] == "<HTML><HEAD><TITLE>PyMCA":
            return True
        return False

    def readDataFromFile(self, attributes, **_kwargs):
        filename = str(self.filename)
        filenam_, _ = os.path.splitext(filename)
        sample = filenam_.split("/")[-1].split(".mca")[0]
        with open(filename, "r") as f:
            content = f.read()
        tmp = np.array([])
        content = (
            content.replace(' align="left"', "")
            .replace(' align="right"', "")
            .replace(" bgcolor=#E6F0F9", "")
            .replace(" bgcolor=#E6F0F9", "")
            .replace(" bgcolor=#FFFFFF", "")
            .replace(" bgcolor=#FFFACD", "")
            .replace("  ", " ")
            .replace("<td >", "<td>")
            .replace("<tr", "\n<tr")
            .replace("<TR", "\n<TR")
            .replace("<table", "\n<table")
            .replace("\n\n", "\n")
        )
        # Cu
        expr = r"<tr><td>Cu</td><td>K</td><td>([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)</td><td>"
        split = refindall(expr, content)
        try:
            tmp = np.append(tmp, float(split[0][0]))
        except IndexError:
            tmp = np.append(tmp, np.nan)
        # In
        expr = r"<tr><td>In</td><td>K</td><td>([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)</td><td>"
        split = refindall(expr, content)
        try:
            tmp = np.append(tmp, float(split[0][0]))
        except IndexError:
            tmp = np.append(tmp, np.nan)
        # Ga
        expr = r"<tr><td>Ga</td><td>Ka</td><td>([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)</td><td>"
        split = refindall(expr, content)
        try:
            tmp = np.append(tmp, float(split[0][0]))
        except IndexError:
            tmp = np.append(tmp, np.nan)
        # Se
        expr = r"<tr><td>Se</td><td>K</td><td>([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)</td><td>"
        split = refindall(expr, content)
        try:
            tmp = np.append(tmp, float(split[0][0]))
        except IndexError:
            tmp = np.append(tmp, np.nan)
        # time - assumed here, no value reading
        tmp = np.append(tmp, 180)

        ##  XRF calibration: MIGHT WANT TO UPDATE THAT!!!
        val_cu = tmp[0] / 43434 * 22.56
        val_in = tmp[1] / 10992 * 13.66
        val_ga = tmp[2] / 29578 * 11.41
        val_se = tmp[3] / 195460 * 52.40
        ##  END OF XRF calibration: MIGHT WANT TO UPDATE THAT!!!

        val_sum = val_cu + val_in + val_ga + val_se
        val_cu /= val_sum
        val_in /= val_sum
        val_ga /= val_sum
        val_se /= val_sum
        tmp = np.append(tmp, val_ga / (val_ga + val_in))  # x calculation
        tmp = np.append(tmp, val_cu / (val_ga + val_in))  # y calculation
        tmp = np.append(tmp, tmp[0] * 450 / 10000000)  # D calculation

        curve = Curve(np.append(np.arange(tmp.size), tmp).reshape((2, tmp.size)), {})
        curve.update(attributes)
        attr = {
            "XRF fit Cu [fitarea/s]": float(tmp[0]) / float(tmp[4]),
            "XRF fit In [fitarea/s]": float(tmp[1]) / float(tmp[4]),
            "XRF fit Ga [fitarea/s]": float(tmp[2]) / float(tmp[4]),
            "XRF fit Se [fitarea/s]": float(tmp[3]) / float(tmp[4]),
            "XRF fit time [s]": float(tmp[4]),
            "XRF fit x [ ]": float(tmp[5]),
            "XRF fit y [ ]": float(tmp[6]),
            "XRF fit D [um]": float(tmp[7]),
            "sample": sample,
            "label": curve.attr("label").split(".")[0],
            "linespec": "d",
            "_collabels": ["Element [ ]", "Fit area / value [a.u.]"],
        }
        curve.update(attr)
        self.append(curve)

        # graph cosmetics
        self.update(
            {
                "xlabel": self.formatAxisLabel(GraphMCAfit.AXISLABELS[0]),
                "ylabel": self.formatAxisLabel(GraphMCAfit.AXISLABELS[1]),
                "xtickslabels": [
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    ["Cu", "In", "Ga", "Se", "Acq. time", "GGI", "CGI", "D"],
                ],
            }
        )

        # Print composition calibration "x=0.3"
        msg = "  XRF composition: GGI {:1.3f}, CGI {:1.3f}, D {:1.3f} (calibration x=0.3)."
        print(msg.format(tmp[5], tmp[6], tmp[7]))
        # Print composition alternative calibration Se ratios
        gase = 1.63 * tmp[2] / tmp[3]
        cuse = 2.0661 * tmp[0] / tmp[3]
        ggi_ = 2 * gase / (1 + 1 / 3 * (1 - 2 * cuse))
        cgi_ = 2 * cuse / (1 + 1 / 3 * (1 - 2 * cuse))
        d_ = 9.50e-06 * tmp[3]
        attr2 = {
            "XRF fit x (Se ratios) [ ]": float(ggi_),
            "XRF fit y (Se ratios) [ ]": float(cgi_),
            "XRF fit D (Se ratios) [um]": float(d_),
        }
        curve.update(attr2)
        msg = "  XRF composition: GGI {:1.3f}, CGI {:1.3f}, D {:1.3f} (calibration Ga/Se, Cu/Se)."
        print(msg.format(ggi_, cgi_, d_))
