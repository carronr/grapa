"""
@author: Romain Carron
Copyright (c) 2023, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import os
import numpy as np

from grapa.graph import Graph
from grapa.datatypes.curveEQE import CurveEQE


class GraphEQE(Graph):

    FILEIO_GRAPHTYPE = "EQE curve"
    FILEIO_GRAPHTYPE_OLD = "EQE curve (old)"

    AXISLABELS = [["Wavelength", r"\lambda", "nm"], ["Cell EQE", "", "%"]]

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1="", **kwargs):
        # new QE files
        if fileExt == ".sr" and line1 == "This is a QE measurment":
            return True
        # old setup QE files (2013(?) or older)
        if fileExt == ".sr" and line1[0:16] == "Reference Cell: ":
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        ifOld = False
        # retrieve sample name - code should work for old and new file format
        f = open(self.filename, "r")
        line = f.readline()  # retrieve sample, should be stored on the 2ndline
        if line[0:16] == "Reference Cell: ":
            ifOld = True
        line = list(
            filter(len, f.readline().replace(": ", "\t").strip(" \r\n\t").split("\t"))
        )
        # look for acquisition attributes
        attributesFile = {}
        interests = [
            ["Amplification settings", "amplification"],
            ["Lockin settings", "lockin"],
            ["Reference File", "referencefile"],
        ]
        for li in f:
            for interest in interests:
                if li.startswith(interest[0]):
                    split = [s.strip() for s in li.strip().split("\t")]
                    for param in split:
                        if ":" in param and ":\\" not in param:
                            # several parameters on same line
                            tmp = param.split(":")
                            attributesFile.update(
                                {interest[1] + tmp[0].replace(" ", ""): tmp[1]}
                            )
                        elif ":\\" in param:
                            # some file...
                            val = os.path.normpath(param).replace("\\", "/")
                            attributesFile.update({interest[1]: val})
            if len(li) == 1 and li[0] in ["\n"]:
                break
        f.close()
        data = np.array([np.nan, np.nan])
        try:
            kw = {"delimiter": "\t", "invalid_raise": False}
            if ifOld:
                kw.update({"skip_header": 14, "usecols": [0, 6]})
            else:
                kw.update({"skip_header": 15, "usecols": [0, 5]})
            data = np.transpose(np.genfromtxt(self.filename, **kw))
        except Exception:
            if not self.silent:
                print("readDataFromFileEQE cannot read file", self.filename)
        # normalize data, units
        self.append(CurveEQE(data, attributes))
        self[-1].update(attributesFile)
        if len(data.shape) > 1:
            self[-1].update({"mulOffset": 100})
        self[-1].update({"_units": ["nm", ""]})
        # update label with information stored inside the file
        lbl = line[-1].replace("_", " ")
        if lbl == "Ref":
            filenam_, fileext = os.path.splitext(self.attr("filename"))
            lbl = (filenam_.split("/")[-1]).split("\\")[-1]
        self[-1].update({"label": lbl, "sample": lbl})
        # some default settings
        self.update(
            {
                "xlabel": GraphEQE.AXISLABELS[0],
                "ylabel": GraphEQE.AXISLABELS[1],
                "ylim": [0, 100],
                "xlim": [300, np.nan],
            }
        )
        self.headers.update({"collabels": ["Wavelength [nm]", "EQE [%]"]})
        if ifOld:
            self.headers.update({"meastype": GraphEQE.FILEIO_GRAPHTYPE_OLD})
