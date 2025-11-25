# -*- coding: utf-8 -*-
"""
To parse files containing curent-voltage JV curves

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import numpy as np

from grapa.graph import Graph
from grapa.utils.parser_dispatcher import FileParserDispatcher
from grapa.datatypes.curveJV import CurveJV


class GraphJV(Graph):
    """To parse files containing curent-voltage JV curves"""

    FILEIO_GRAPHTYPE = "J-V curve"
    FILEIO_GRAPHTYPE_TIV = "TIV curve"
    FILEIO_GRAPHTYPE_IV_HLS = "I-V curve (H-L soaking)"

    AXISLABELS = [["Bias voltage", "V", "V"], ["Current density", "J", "mA cm$^{-2}$"]]

    @classmethod
    def isFileReadable(
        cls, _filename, fileext, line1="", line2="", line3="", **_kwargs
    ):
        """Can this module open the file?"""
        if fileext == ".txt" and line1[-11:] == "_ParamDescr":
            return True  # JV
        if (
            fileext == ".txt"
            and line1.strip().startswith("Sample name:")
            and line2.strip().startswith("Cell name:")
            and line3.strip().startswith("Cell area [cm")
        ):
            return True  # TIV
        if (
            fileext == ".csv"
            and len(line1) > 40
            and line1.strip()[:21] == '"Time";"Temperature ['
            and line1.strip()[-38:] == '";"Illumination [mV]";"Humidity [%RH]"'
        ):
            return True  # J-V from Heat-Light soaking setup
        return False

    def readDataFromFile(self, attributes, **kwargs):
        """Decides how to open the file"""
        line1 = kwargs["line1"] if "line1" in kwargs else ""
        fileName = kwargs["fileName"] if "fileName" in kwargs else ""
        if (
            len(line1) > 40
            and line1.strip()[:21] == '"Time";"Temperature ['
            and line1.strip()[-38:] == '";"Illumination [mV]";"Humidity [%RH]"'
        ):
            GraphJV.readDataFromFileIV_HLS(self, attributes)
        elif line1.strip().startswith("Sample name:") and (
            not fileName.startswith("I-V_")
        ):
            GraphJV.readDataFromFileTIV(self, attributes)
        else:
            GraphJV.readDataFromFileJV(self, attributes)
        # set 'sample' attribute
        if self[-1].attr("sample") == "":
            self[-1].update({"sample": self[-1].attr("label")})

    def readDataFromFileJV(self, attributes):
        """File format of the in-house JV setup"""
        # extract data analysis from acquisition software
        # - requires a priori knowledge of file structure
        # especially want to know cell area before creation of the CurveJV object
        kw = {"skip_header": 1, "delimiter": "\t", "invalid_raise": False, "dtype": str}
        jvsoft = np.genfromtxt(self.filename, usecols=[3, 4], **kw)
        test_fileformat = "".join([line[0] for line in jvsoft])
        if len(test_fileformat) == 0:
            jvsoft = np.genfromtxt(self.filename, usecols=[2, 4], **kw)

        key_replace = {
            "approx. cell temperature": "temperature",
            "measurement date and time": "datetime",
        }
        for value, key_ in jvsoft:
            if len(value) > 0:  # is a str by construction
                key = key_.lower().strip()
                if " (" in key:
                    key = key[: key.find(" (")]
                if key in key_replace:
                    key = key_replace[key]
                key = "acquis soft " + key
                try:
                    value = float(value)
                except ValueError:
                    pass  # keep as str
                attributes.update({key: value})
        # until 08.05.2025 - update of IV acquisition software
        # key = ["Acquis soft Voc", "Acquis soft Jsc", "Acquis soft FF",
        #        "Acquis soft Eff", "Acquis soft Cell area", "Acquis soft Vmpp",
        #        "Acquis soft Jmpp", "Acquis soft Pmpp", "Acquis soft Rp",
        #        "Acquis soft Rs", "Acquis soft datetime", "Acquis soft Temperature",
        #        "Acquis soft Illumination factor", "Acquis soft (Cell Comment)",
        #        "Acquis soft (Global Comment)", "Acquis soft Lightsource",
        #       ]
        # for i in range(len(key)):
        #     val = JVsoft[i]
        #     if len(val) > 0:  # is a str by construction
        #         try:
        #             val = float(val)
        #         except ValueError:
        #             pass  # keep as str
        #         attributes.update({key[i]: val})

        if "label" in attributes:
            attributes["label"] = (
                attributes["label"].replace("I-V ", "").replace("_", " ")
            )
        # create Curve object
        data = np.genfromtxt(
            self.filename,
            skip_header=1,
            delimiter="\t",
            usecols=[0, 1],
            invalid_raise=False,
        )
        curve = CurveJV(
            np.transpose(data),
            attributes,
            units=["V", "mAcm-2"],
            ifCalc=True,
            silent=self.silent,
        )
        self.append(curve)
        self[-1].update({"_collabels": ["Voltage [V]", "Current density [mA cm-2]"]})
        self.update(
            {
                "xlabel": self.formatAxisLabel(GraphJV.AXISLABELS[0]),
                "ylabel": self.formatAxisLabel(GraphJV.AXISLABELS[1]),
                "axhline": [0, {"linewidth": 0.5}],
                "axvline": [0, {"linewidth": 0.5}],
            }
        )

    def readDataFromFileTIV(self, attributes):
        """File format of in-house TIV setup"""
        FileParserDispatcher.readDataFromFileGeneric(self, attributes)
        # check if we can actually read some TIV data, otherwise this might be
        # a processed I-V_ database (summary of cells properties)
        if len(self) == 0:
            FileParserDispatcher.readDataFromFileTxt(self, attributes)
            return
        # back to TIV data reading
        self.update(
            {
                "xlabel": self.formatAxisLabel(GraphJV.AXISLABELS[0]),
                "ylabel": self.formatAxisLabel(GraphJV.AXISLABELS[1]),
            }
        )
        # delete columns of temperature in newer versions
        if len(self) == 3:
            if self[-1].attr("voltage (v)") == "T sample (K)":
                self.curve_delete(-1)
            if self[-1].attr("voltage (v)") == "T stage (K)":
                self.curve_delete(-1)
        self[0].update({"voltage (v)": ""})  # artifact from file reading
        filebasename, _fileextfileext = os.path.splitext(
            os.path.basename(self.filename)
        )
        self[0].update({"label": filebasename.replace("_", " ")})
        # rename attributes to match these of JV setup
        key = [
            "sample name",
            "cell name",
            "voc (v)",
            "jsc (ma/cm2)",
            "ff (%)",
            "eff (%)",
            "cell area [cm2]",
            "vmpp (v)",
            "jmpp (ma/cm2)",
            "pmpp (mw/cm2)",
            "rp (ohmcm2)",
            "rs (ohmcm2)",
            "temperature [k]",
            "temperature",
            "Acquis soft Illumination factor",
        ]  # , 'Acquis soft datatime'
        new = [
            "sample",
            "cell",
            "Acquis soft Voc",
            "Acquis soft Jsc",
            "Acquis soft FF",
            "Acquis soft Eff",
            "Acquis soft Cell area",
            "Acquis soft Vmpp",
            "Acquis soft Jmpp",
            "Acquis soft Pmpp",
            "Acquis soft Rp",
            "Acquis soft Rs",
            "Acquis soft Temperature",
            "Acquis soft Temperature",
            "Acquis soft Illumination factor",
        ]  # ,  'Acquis soft datatime'
        c = self[0]
        for i in range(len(key)):
            if c.attr(key[i]) != "":  # risk of duplicates in list above...
                c.update({new[i]: c.attr(key[i])})
                c.update({key[i]: ""})
        # we believe this information is reliable
        c.update({"temperature": c.attr("Acquis soft Temperature")})
        c.update({"measId": str(c.attr("temperature"))})
        # recreate curve as a CurveJV
        self[0] = CurveJV(c.data, attributes=c.get_attributes(), silent=True)
        self.update({"meastype": GraphJV.FILEIO_GRAPHTYPE_TIV})

    def readDataFromFileIV_HLS(self, attributes):
        """File format of in-house HLS setup"""
        le = len(self)
        FileParserDispatcher.readDataFromFileGeneric(self, attributes, delimiter=";")
        self[le].update({"area": 1.0})
        if max(abs(self[le].x())) > 10:  # want units in [V], not [mV]
            self[le].setX(self[le].x() * 0.001)
        self.update({"xlabel": self.attr("xlabel").replace("[mv]", "[mV]")})
        self.update(
            {
                "xlabel": self.formatAxisLabel(self.attr("xlabel")),
                "ylabel": self.formatAxisLabel(self.attr("ylabel")),
            }
        )
        self.castCurve("Curve JV", le, silentSuccess=True)
        self.update({"meastype": GraphJV.FILEIO_GRAPHTYPE_IV_HLS})
