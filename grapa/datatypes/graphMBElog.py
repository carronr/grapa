# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 10:44:14 2017

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

import os
import numpy as np


from grapa.graph import Graph
from grapa.curve import Curve


class GraphMBElog(Graph):
    FILEIO_GRAPHTYPE = "Small MBE log file"

    AXISLABELS = [["Time", "t", "min"], ["Substrate heating", "", "a.u."]]

    @classmethod
    def isFileReadable(cls, _filename, fileext, line1="", **_kwargs):
        if fileext == ".txt" and line1 == "Time	Value":
            return True
        return False

    def readDataFromFile(self, attributes, **_kwargs):
        """Reads MBE log file. Curve is standard xyCurve."""
        fileName, fileext = os.path.splitext(self.filename)
        sample = (fileName.split("/")[-1]).split("_")[0]
        data = np.genfromtxt(
            self.filename,
            skip_header=1,
            delimiter="\t",
            usecols=[0, 1],
            invalid_raise=False,
        )
        self.append(Curve(np.transpose(data), attributes))
        self[-1].update(
            {
                "_collabels": ["Time [min]", "Substrate heating [a.u.]"],
                "label": sample,
                "sample": sample,
            }
        )
        self.update(
            {
                "xlabel": self.formatAxisLabel(GraphMBElog.AXISLABELS[0]),
                "ylabel": self.formatAxisLabel(GraphMBElog.AXISLABELS[1]),
            }
        )
