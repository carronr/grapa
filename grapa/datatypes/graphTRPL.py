# -*- coding: utf-8 -*-
"""
Parse files containing time-resolved photoluminscence (TRPL) decays

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

from os import path as ospath

from grapa.graph import Graph
from grapa.utils.parser_dispatcher import FileParserDispatcher
from grapa.datatypes.curveTRPL import CurveTRPL


class GraphTRPL(Graph):
    """Parse files containing time-resolved photoluminscence (TRPL) decays"""

    FILEIO_GRAPHTYPE = "TRPL decay"

    AXISLABELS = [CurveTRPL.AXISLABELS_X[""], CurveTRPL.AXISLABELS_Y[""]]

    @classmethod
    def isFileReadable(
        cls, _filename, fileext, line1="", line2="", line3="", **_kwargs
    ):
        if fileext == ".dat" and line1 == "Time[ns]	crv[0] [Cnts.]":
            return True  # TRPL file
        elif (
            fileext == ".dat"
            and line1 == "Parameters:"
            and (
                (
                    line2.strip().startswith("Sample")
                    and line3.strip().startswith("Solvent")
                )
                or (
                    line2.strip().split(" : ")[0] in ["Exc_Wavelength"]
                    and line3.strip().split(" : ")[0] in ["Exc_Bandpass"]
                )
            )
        ):  # spectroscopy file
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        """Read a TRPL decay."""
        len0 = len(self)
        kw = {}
        if "line1" in kwargs and kwargs["line1"] == "Parameters:":
            kw.update({"delimiterHeaders": " : "})
        FileParserDispatcher.readDataFromFileGeneric(self, attributes, **kw)
        self.castCurve(CurveTRPL.CURVE, len0, silentSuccess=True)
        # label management
        filenam_, _fileext = ospath.splitext(self.filename)  # , fileExt
        # self[len0].update({'label': filenam_.split('/')[-1].split('\\')[-1]})
        lbl = filenam_.split("/")[-1].split("\\")[-1].replace("_", " ").split(" ")
        smp = str(self[len0].attr("sample"))
        try:
            if float(int(float(smp))) == float(smp):
                smp = str(int(float(smp)))
        except Exception:
            pass
        smp = smp.replace("_", " ").split(" ")
        new = lbl
        if len(smp) > 0:
            new = [l for l in lbl if l not in smp] + smp
        # print('label', self.attr('label'), [l for l in lbl if l not in smp], smp)
        self[len0].update({"label": " ".join(new)})
        # clean values
        for key in self[len0].get_attributes():
            val = self[len0].attr(key)
            if isinstance(val, str) and "Â°" in val:
                self[len0].update({key: val.replace("Â°", "°")})
        # graph properties
        xlabel = (
            self.attr("xlabel").replace("[", " [").replace("  ", " ").capitalize()
        )  # ] ]
        if xlabel in ["", " "]:
            xlabel = GraphTRPL.AXISLABELS[0]
        self.update(
            {
                "typeplot": "semilogy",
                "alter": ["", "idle"],
                "xlabel": self.formatAxisLabel(xlabel),
                "ylabel": self.formatAxisLabel(GraphTRPL.AXISLABELS[1]),
            }
        )
        self.update({"subplots_adjust": [0.2, 0.15]})
        # cleaning
        if "line1" in kwargs and kwargs["line1"] == "Parameters:":
            attr = self[len0].get_attributes()
            keys = list(attr.keys())
            for key in keys:
                val = attr[key]
                if isinstance(val, str) and val.startswith("\t"):
                    self[len0].update({key: ""})
