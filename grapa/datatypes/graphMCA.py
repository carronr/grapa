# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2026, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

from typing import Optional
from re import findall as refindall
import numpy as np

from grapa.graph import Graph
from grapa.datatypes.curveMCA import CurveMCA
from grapa.shared.string_manipulations import strToVar
from grapa.shared.maths import is_number
from grapa.parse.parser_generic import file_lines


def parse_from_list_of_lines(lines, attributes: Optional[dict] = None):
    """Parses MCA lines of file into channel counts and metadata attributes."""
    if attributes is None:
        attributes = {}

    def categ(line, old):
        if line.startswith("<<"):
            return line.replace("<", "").replace(">", "")
        return old

    y = []
    category = ""
    section = 0
    for line in lines:
        if section == 0:
            if line == "<<DATA>>":
                section = 1
            category = categ(line, category)
            split = refindall("([^-]+) - ([^-]+)", line)
            if len(split) > 0 and len(split[0]) > 1:
                at = category + "." + split[0][0]
                attributes.update({at: strToVar(split[0][1])})
        elif section == 1:
            if is_number(line):
                y.append(float(line))
            else:
                category = categ(line, category)
                section = 2
        elif section == 2:
            category = categ(line, category)
            expr = r"([^0-9]+): ([-+]?[0-9]*\.?[0-9]+) *([^0-9]*)"
            split = refindall(expr, line)
            if len(split) == 0:
                expr = r"([^0-9]+): (\w+)()"
                split = refindall(expr, line)
            if len(split) > 0 and len(split[0]) > 2:
                tmp = split[0][2]
                if len(split[0][2]) > 0:
                    tmp = " [" + split[0][2] + "]"
                at = category + "." + split[0][0] + tmp
                attributes.update({at: strToVar(split[0][1])})
            else:
                pass  # not interesting input (section title, etc.)
    y = np.array(y, dtype=float)
    data = np.append(np.arange(y.size), y).reshape((2, y.size))
    return attributes, data


class GraphMCA(Graph):
    FILEIO_GRAPHTYPE = "XRF MCA raw data"

    AXISLABELS = [CurveMCA.AXISLABELS_X[""], CurveMCA.AXISLABELS_Y[""]]

    @classmethod
    def isFileReadable(cls, _filename, fileext, line1="", **_kwargs):
        if fileext == ".mca" and line1 == "<<PMCA SPECTRUM>>":
            return True
        return False

    def readDataFromFile(self, attributes, **_kwargs):
        # parse file
        lines, _enc = file_lines(self.filename)
        attributes, data = parse_from_list_of_lines(lines, attributes=attributes)

        # format data, create Curve
        at = "DPP CONFIGURATION.Preset"
        if at + " [sec]" not in attributes and at + " [min]" in attributes:
            attributes[at + " [sec]"] = attributes[at + " [min]"] * 60
        self.append(CurveMCA(data, attributes))
        self[-1].update({"muloffset": "1/" + str(self.attr(at + " [sec]"))})
        if self[-1].attr("sample") == "":
            self[-1].update({"sample": self[-1].attr("label")})
        self[-1].data_units(unit_y="counts")
        self[-1].update({"_collabels": ["Channel [ ]", "Counts [ ]"]})

        # graph cosmetics
        ylabel = list(GraphMCA.AXISLABELS[1])
        if self[-1].get_muloffset()[1] != 1:
            ylabel[2] = ylabel[2] + " s$^{-1}$"
        self.update(
            {
                "xlabel": self.format_axis_label(GraphMCA.AXISLABELS[0]),
                "ylabel": self.format_axis_label(ylabel),
            }
        )
        if data.shape[1] == 1024:
            self.update({"xlim": [0, 1024]})
        print("Opened MCA raw data file. To know composition, open .html file instead.")
