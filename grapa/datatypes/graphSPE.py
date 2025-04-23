# -*- coding: utf-8 -*-
"""
An very primitive data parser for .spe data files containing XRF data
of a given manufacturer.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np

from grapa.graph import Graph
from grapa.datatypes.curveMCA import CurveMCA


class GraphSPE(Graph):
    """
    An very primitive data parser for .spe data files containing XRF data
    of a given manufacturer.
    """

    FILEIO_GRAPHTYPE = "XRF SPE raw data"

    AXISLABELS = [["XRF detector channel", "", " "], ["Counts", "", "cts"]]

    USERWARNING = True

    @classmethod
    def isFileReadable(
        cls, _filename, fileext, line1="", line2="", line3="", **_kwargs
    ):
        if fileext == ".spe":
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        readatstart = (
            348 - 10 * 2
        )  # can be changed. Change _mca_ctokev_offset accordingly
        datatype = np.float32
        xdim, ydim = 1024, 1

        def read_at(file, pos, size, datatype):
            file.seek(pos)
            return np.fromfile(file, datatype, size)

        with open(self.filename, "rb") as file:
            img = read_at(file, readatstart, xdim * ydim, datatype)
            # fid.close()

        datay = img.flatten()
        datax = list(range(len(datay)))
        self.append(CurveMCA([datax, datay], attributes))
        self[-1].update({"_mca_ctokev_mult": 0.0326, "_mca_ctokev_offset": 7.0})  # 2.0
        if self[-1].attr("sample") == "":
            self[-1].update({"sample": self[-1].attr("label")})
        # graph cosmetics
        self.update(
            {"xlabel": GraphSPE.AXISLABELS[0], "ylabel": GraphSPE.AXISLABELS[1]}
        )
        # user warning - under development
        if GraphSPE.USERWARNING:
            print(
                "WARNING GraphSPE: parsing of the .spe file very crude, and",
                " will absolutely not work for all .spe versions",
            )
            print("THe parsing of metadata is currently not implemented.")
            print("This warning message will only show once.")
            GraphSPE.USERWARNING = False
