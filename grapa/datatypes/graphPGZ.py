# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:30:22 2023

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""


import numpy as np
from grapa.graph import Graph
from grapa.curve import Curve


class GraphPGZ(Graph):
    FILEIO_GRAPHTYPE = "PGZ path file"

    AXISLABELS = [["Position x", "", "mm"], ["Position y", "", "mm"]]

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1="", **kwargs):
        if fileExt == ".pgz":
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        kw = {
            "delimiter": "\t",
            "invalid_raise": False,
            "skip_header": 0,
            "usecols": [0, 1, 2],
        }
        data = np.genfromtxt(self.filename, **kw)

        # transform data format
        x = [[], []]
        y = [[], []]
        previous = -1
        for row in data:
            onOff = 1 if row[2] == 1 else 0
            if row[2] != previous and len(x[1 - onOff]) > 0:
                x[onOff].append(x[1 - onOff][-1])
                y[onOff].append(y[1 - onOff][-1])
                x[1 - onOff].append(np.nan)
                y[1 - onOff].append(np.nan)
                #  print('   added from other')
            x[onOff].append(row[0])
            y[onOff].append(row[1])
            # print('   added to', onOff)
            previous = row[2] * 1

        self.append(Curve([x[0], y[0]], attributes))
        self.append(Curve([x[1], y[1]], attributes))
        lbl = self[-2].attr("label")
        self[-1].update({"label": lbl + " off", "labelhide": 1, "alpha": 0.20})

        colors = {
            "P1": [0.306, 0.627, 0],
            "P2": [0.992, 0.388, 0.106],
            "P3": [0.565, 0.376, 0.651],
            "PT": [0, 0.651, 0.612],
        }
        color = ""
        for key in colors:
            if key in lbl:
                color = colors[key]
                break
        if color != "":
            self[-2].update({"color": color})
            self[-1].update({"color": color})

        # some default settings
        self.update(
            {
                "figsize": [5, 5],
                "subplots_adjust": [0.15, 0.15, 0.95, 0.95],
                "xlabel": self.formatAxisLabel(GraphPGZ.AXISLABELS[0]),
                "ylabel": self.formatAxisLabel(GraphPGZ.AXISLABELS[1]),
            }
        )
