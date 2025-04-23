# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:25:49 2018

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

# -*- coding: utf-8 -*-

import numpy as np

from grapa.graph import Graph
from grapa.curve import Curve


class GraphXPS(Graph):
    """
    reads XPS data of system
    """

    FILEIO_GRAPHTYPE = "XPS data"

    AXISLABELS = [["Binding energy", "", "eV"], ["Intensity", "", "counts/s"]]

    @classmethod
    def isFileReadable(cls, _filename, fileext, line1="", **_kwargs):
        if fileext == ".csv" and line1.startswith("Point"):
            return True
        return False

    def readDataFromFile(self, attributes, **_kwargs):
        # parse headers
        attrs = {}
        f = open(self.filename, "r")
        attrs["acquisition location"] = f.readline().strip()
        attrs["acquisition sample"] = f.readline().strip()
        attrs["acquisition type"] = f.readline().strip()
        f.close()
        attrs["muloffset"] = 1
        # reads data
        data = np.array([np.nan, np.nan])
        kw = {
            "delimiter": ",",
            "invalid_raise": False,
            "skip_header": 4,
            "usecols": [0, 1],
        }
        try:
            data = np.transpose(np.genfromtxt(self.filename, **kw))
        except Exception as e:
            if not self.silent:
                print("WARNING GraphXPS cannot read file", self.filename)
                print(type(e), e)
                return False
        try:
            self.append(Curve(data, attributes))
        except Exception as e:
            print("WARNING GraphXPS cannot read file", self.filename)
            print(type(e), e)
            return False
        self[-1].update(attrs)  # file content may override default attributes
        self[-1].update({"sample": self.attr("label")})
        # graph cosmetics
        self.update(
            {
                "xlabel": self.formatAxisLabel(GraphXPS.AXISLABELS[0]),
                "ylabel": self.formatAxisLabel(GraphXPS.AXISLABELS[1]),
            }
        )
        self.update({"xlim": [max(self[-1].x()), min(self[-1].x())]})
