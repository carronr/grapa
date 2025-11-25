# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 12:52:52 2017

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np
from grapa.graph import Graph
from grapa.datatypes.curveEQE import CurveEQE


class GraphEQE_CSU(Graph):
    FILEIO_GRAPHTYPE = "EQE curve"

    AXISLABELS = [["Wavelength", r"\lambda", "nm"], ["Cell EQE", "", "%"]]

    @classmethod
    def isFileReadable(cls, _filename, fileext, line1="", line3="", **_kwargs):
        if fileext == ".txt" and line1 == "CSU" and line3.startswith("Bias Light"):
            return True
        return False

    def readDataFromFile(self, attributes, **_kwargs):
        attrs = {}
        # interpret headers
        with open(self.filename, "r") as f:
            attrs["acquisition location"] = f.readline().strip()
            attrs["sample"] = f.readline().strip()
            attrs["label"] = attrs["sample"]
            attrs["acquisition bias light"] = (
                f.readline().replace("Bias Light ", "").strip()
            )
            attrs["acquisition cell area"] = float(f.readline().strip())
            attrs["acquisition Eg"] = float(f.readline().strip().replace("Eg = ", ""))
            f.readline()  # do nothing with it
            nlines = int(f.readline().strip())
            attrs["acquisition Jsc"] = float(f.readline().strip())
            attrs["_collabels"] = f.readline().strip().split("\t")

        # read data
        data = np.array([np.nan, np.nan])
        kw = {
            "delimiter": "\t",
            "invalid_raise": False,
            "skip_header": 9,
            "usecols": [0, 1],
        }
        try:
            data = np.transpose(np.genfromtxt(self.filename, **kw))
        except Exception as e:
            if not self.silent:
                print("WARNING GraphEQE_CSU cannot read file", self.filename)
                print(type(e), e)

        # create data
        self.append(CurveEQE(data, attributes))
        self[-1].update(attrs)  # file content may override default attributes
        if nlines != len(self[-1].x()):
            print("WARNING GraphEQE_CSU: number of lines differ from expected")

        # cosmetics
        if len(data.shape) > 1:
            self[-1].update({"muloffset": 100})
        # some default settings
        self.update(
            {
                "ylim": [0, 100],
                "xlim": [300, np.nan],
                "xlabel": self.formatAxisLabel(GraphEQE_CSU.AXISLABELS[0]),
                "ylabel": self.formatAxisLabel(GraphEQE_CSU.AXISLABELS[1]),
            }
        )
