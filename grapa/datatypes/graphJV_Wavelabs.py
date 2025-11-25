# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:53:00 2023

@author: Matthias Diethelm
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import numpy as np
from datetime import datetime

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.datatypes.curveJV import CurveJV

try:
    import pylightxl as xl
except ImportError:
    pass


def asfloat(value):
    val = value
    try:
        val = float(value)
    except ValueError:
        pass
    if val == "#NV":
        val = np.nan
    return val


def cleanupdata(data, curveindex):
    # TODO: evaluate if following algorithm for removal of faulty first
    # datapoint is safe to use. Sometimes the first datapoint seems very low
    diff = np.abs(data[:-1, 1] - data[1:, 1])
    if diff[0] > 3 * np.max(diff[1 : min(10, len(diff))]):
        msg = "WARNING CurveJV WAVELABS curve {}: suspicious current value at index 0, removed first datapoint. {}"
        print(msg.format(curveindex, data[:5, 1]))
        data = data[1:, :]
    # TODO: assess is following tests are reasonable:
    # - removal of initial datapoint, and
    # - removal of saturated last datapoints
    # check if x is sorted - maybe faulty initial datapoint, maybe saturated at the end
    if not np.all(data[:-1, 0] <= data[1:, 0]):
        msg = "WARNING CurveJV WAVELABS curve {}: voltage data not sorted."
        print(msg.format(curveindex))
        issues = np.where(data[:-1, 0] > data[1:, 0])
        for idx in issues[0][::-1]:  # scan in reverse order
            msgtmp = "  {} {}".format(
                idx, data[max(0, idx - 1) : min(idx + 2, data.shape[0]), 0]
            )
            if idx == 0:
                msgtmp += ". Removed datapoint."
                data = data[1:, :]
            # TODO: revise conditions?
            if idx > 10 and idx > data.shape[1] - 5:
                # only if amongst very last datapoints
                if data[idx, 1] > 0.99 * np.max(data[:, 1]):
                    # should be saturated datapoints
                    lasty = data[idx:, 1]
                    if (np.max(lasty) - np.min(lasty)) / np.max(lasty) < 0.01:
                        # only if very similar y values
                        msgadd = ". Removed datapoint and subsequent ones ({})."
                        msgtmp += msgadd.format(len(lasty))
                        data = data[:idx, :]
            print(msgtmp)
    return data


class GraphJV_Wavelabs(Graph):
    """
    Reads JV .xlsx files exported from Wavelabs
    """

    FILEIO_GRAPHTYPE = "J-V curve"

    @classmethod
    def isFileReadable(cls, filename, fileext, line1="", **_kwargs):
        if fileext == ".xlsx" and (
            filename.startswith("IV Measurement")
            or filename.startswith("MPPT Measurement")
        ):
            try:
                xl  # if import pylightxl worked successfully
                return True
            except NameError:
                msg = (
                    "ImportError package pylightxl not found, file import Wavelabs "
                    "may not work."
                )
                print(msg)
        return False

    def readDataFromFile(self, attributes, **_kwargs):
        if "IV Measurement" in self.filename:
            GraphJV_Wavelabs.parse_iv(self, attributes)
        elif "MPPT Measurement" in self.filename:
            GraphJV_Wavelabs.parse_mpp(self, attributes)

    def parse_iv(self, attributes):
        """IV Measurement"""
        kwargscurve = {"units": ["V", "mAcm-2"], "ifCalc": True, "silent": self.silent}

        cols_attr = {
            3: "Acquis soft Isc [mA]",
            4: "Acquis soft Voc [V]",
            5: "Acquis soft Vmpp [V]",
            6: "Acquis soft Impp [mA]",
            7: "Acquis soft Pmpp [mW]",
            8: "Acquis soft FF",
            9: "Acquis soft Rp [kOhm]",
            10: "Acquis soft Rs [Ohm]",
            11: "Acquis soft Eta [%]",
            12: "Acquis soft Jsc [mA/cm2]",
            13: "Acquis soft Flashtime [ms]",
        }
        cols_attr_rs = {
            17: "Acquis soft Isc 2 [mA]",
            18: "Acquis soft Vmpp 2 [V]",
            19: "Acquis soft Impp 2 [mA]",
            20: "Acquis soft Pmpp 2 [mW]",
        }

        db = xl.readxl(fn=self.filename)
        # determine starting indices of measurements within one file
        col4 = db.ws(ws="sheet1").col(col=4)
        start_ind = [index + 1 for (index, item) in enumerate(col4) if not item == ""]
        if start_ind[0] == 1:
            del start_ind[0]
        else:
            msg = "WARNING: failed to parse labels of header line, presumably ok. {}"
            print(msg.format(self.filename))
            print(" ", db.ws(ws="sheet1").range("A1:E3"))
        start_ind.append(len(col4) + 1)  # to know where to stop last acquisition
        for i in range(len(start_ind) - 1):
            attributes_row = db.ws(ws="sheet1").row(row=start_ind[i])
            # main IV curve
            for idx, key in cols_attr.items():
                val = asfloat(attributes_row[idx])
                attributes.update({key: val})
            # TODO: should we keep round() ?
            key_area = "Acquis soft Cell area"
            cellarea = round(
                attributes["Acquis soft Isc [mA]"]
                / attributes["Acquis soft Jsc [mA/cm2]"],
                10,
            )
            if cellarea == 0 or np.isnan(cellarea):
                cellarea = 1
            attributes.update({key_area: cellarea})
            # TODO: should we worry about illumPower to feed to CurveJV ?
            #    Risk of wrong number for for illumination != AM1.5G
            at = GraphJV_Wavelabs.interpret_filename(attributes["filename"])
            if len(start_ind) - 1 > 1 and "label" in at:
                at["label"] = "{} {}".format(at["label"], i)
            attributes.update(at)
            # Measurement data
            address = "A" + str(start_ind[i]) + ":B" + str(start_ind[i + 1] - 1)
            data = np.array(db.ws(ws="sheet1").range(address=address), dtype=float)
            data[:, 1] = -data[:, 1] / cellarea
            data = cleanupdata(data, i)
            # create Curve object
            curve = CurveJV(np.transpose(data[:, 0:2]), attributes, **kwargscurve)
            if curve.darkOrIllum():  # dark: revise label
                label = curve.attr("label") + " " + curve.darkOrIllum(ifText=True)
                curve.update({"label": label})
            curve.update({"_collabels": ["Voltage [V]", "Current density [mA cm-2]"]})
            self.append(curve)

            # RS curve
            attributes_rs = {}
            key_rs = [key_area, "filename"]
            for key in key_rs:
                val = asfloat(attributes[key])
                attributes_rs.update({key: val})
            label = attributes["label"] + " Rs half intensity"
            attributes_rs.update({"label": label})
            for idx, key in cols_attr_rs.items():
                val = asfloat(attributes_row[idx])
                attributes.update({key: val})
            # retrieve JV data
            address = "O" + str(start_ind[i]) + ":P" + str(start_ind[i + 1] - 1)
            try:
                data_rs = db.ws(ws="sheet1").range(address=address)
                data_rs = np.array(data_rs, dtype=float)
            except ValueError:
                # if scan at half intensity was not saved, then content is "" -> error
                continue
            data_rs[:, 1] = -data_rs[:, 1] / cellarea
            data = cleanupdata(data, i)
            # create Curve object
            curve = CurveJV(np.transpose(data_rs[:, 0:2]), attributes_rs, **kwargscurve)
            curve.update({"_collabels": ["Voltage [V]", ""]})
            curve.visible(False)
            self.append(curve)

        # end of parse data - graph cosmetics - JV
        self.update(
            {
                "xlabel": ["Bias voltage", "V", "V"],
                "ylabel": ["Current density", "J", "mA cm$^{-2}$"],
                "axhline": [0, {"linewidth": 0.5}],
                "axvline": [0, {"linewidth": 0.5}],
            }
        )

    def parse_mpp(self, attributes):
        """MPPT Measurement"""
        # TODO: how to deal with cell area?
        # create Curve object
        at = GraphJV_Wavelabs.interpret_filename(attributes["filename"])
        attributes.update(at)

        # Measurement data
        db = xl.readxl(fn=self.filename)
        sheet_obj = db.ws(ws="sheet1")
        data = []
        for i in range(2, sheet_obj.maxrow + 1):
            row = sheet_obj.row(i)
            data.append(row)
        data = np.array(data)
        header = sheet_obj.row(1)
        if header[0] == "":
            print("WARNING: well, parse of header failed, we are down to guesswork")
            header = ["Time [s]", "Umpp [mV]", "Impp [mA]", "Pmpp [mW]"]
        # data[:, 2] = -data[:, 2] / attributes["Acquis soft Cell area"]  # NOPE
        # data[:, 3] = data[:, 3] / attributes["Acquis soft Cell area"]  # NOPE
        self.append(Curve(np.transpose(data[:, [0, 3]]), attributes))
        label = self[-1].attr("label")
        self[-1].update({"label": label + " " + header[3]})
        self.append(Curve(np.transpose(data[:, [0, 1]]), attributes))
        self[-1].update({"label": label + " " + header[1]})
        self[-1].visible(False)
        self.append(Curve(np.transpose(data[:, [0, 2]]), attributes))
        self[-1].update({"label": label + " " + header[2]})
        self[-1].visible(False)
        self[-1].update({"_collabels": header})  # not sure this is correct anymore
        # graph cosmetics
        self.update({"xlabel": header[0], "ylabel": header[3]})

    @staticmethod
    def interpret_filename(filename):
        out = {}
        filebase = os.path.splitext(os.path.basename(filename))[0]
        split = filebase.split("_")
        if len(split) > 2:
            sample = split[1]
            date_str = " ".join(split[-2:])
            try:  # to make isoformat out of filename information
                date = "20" + date_str[:-2] + ":" + date_str[-2:] + ":00"
                date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                date_str = date.isoformat()
            except ValueError:
                pass
            label = sample
            if len(label) == 0:
                label = filebase
            out.update({"sample": sample, "measurement date": date_str, "label": label})
        return out
